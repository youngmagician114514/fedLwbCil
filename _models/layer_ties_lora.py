import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from _models import register_model
from typing import List
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool

import math
from typing import Optional, List


def zero_pad(x, lora_ind):
    result = x.new_zeros((len(lora_ind), *x.shape[1:]))
    result[lora_ind] = x
    return result


def merge_AB(A, B, lora_ind):
    def T(w):
        # return w.transpose(0, 1) if self.fan_in_fan_out else w
        return w

    delta_w = F.conv1d(A.unsqueeze(0), B.unsqueeze(-1), groups=3).squeeze(
        0
    )  # groups = 3 because we are using q,k and v, shape [768, 2304]
    return T(zero_pad(delta_w, lora_ind))


@register_model("layer_ties_lora")
class TiesLora(BaseModel):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 3e-4,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1.0,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        ties_keep_ratio: float = 0.5,
        ties_lambda: float = 1.0,
    ) -> None:
        # for LoRA, we keep the mean of the LoRA modules of the old tasks
        super().__init__(fabric, network, device, optimizer, lr, wd_reg)
        self.lora_alpha = lora_alpha
        self.r = r
        self.cl_merge = cl_merge
        
                # === TIES 超参数 ===
        # 保留每个“任务向量”（client 的 LoRA 参数）中幅值最大的 top-k%，默认 k=0.2 (20%)
        self.ties_keep_ratio = ties_keep_ratio
        # 缩放系数 λ，最终 delta = λ * merged_task_vector
        self.ties_lambda = ties_lambda
        
        self.lora_keys = []
        self.lora_params = {}
        self.optimizer_str = optimizer
        self.lr = lr
        self.clip_grad = clip_grad
        self.wd_reg = wd_reg
        self.lora_head = lora_head
        self.head_keys = []
        self.old_delta = {}
        self.cur_B = {}
        self.cur_A = {}
        self.init_lora_params(network, r)
        self.optimization_dict = {}
        if not self.lora_head:
            self.head = {
                key: nn.Parameter(torch.tensor(self.network.state_dict()[key].clone().detach()), requires_grad=True).to(
                    self.device
                )
                for key in self.head_keys
            }
        self.old_tasks_A = None
        self.old_tasks_B = None
        if self.cl_merge == "individual":
            self.old_tasks_A = {}
            self.old_tasks_B = {}
        self.avg_type = avg_type
        self.pre_B = {}
        self.pre_A = {}
        self.pre_head = None
        self.pre_network = None
        # self.network.eval()


    def ties_merge_param_with_w0(
        self,
        param_list,
        weight_init,
        keep_ratio: float = None,
    ):
        #*修剪时也参考预训练权重的大小，但是删去时只删去多余的部分,这是基于层进行ties的代码，后续看情况要不要改成完全整体修剪
        assert len(param_list) > 0
                    
        stacked = torch.stack(param_list, dim=0)  # (N, ...)
        device = stacked.device
        dtype = stacked.dtype        
        N = stacked.size(0)

        if keep_ratio is None:
            keep_ratio = self.ties_keep_ratio
        keep_ratio = float(keep_ratio)
        
        stacked_with_w0 = stacked + weight_init.unsqueeze(0)
        
        # ---------- 1) Trim：每个任务向量自身做 top-k% 剪枝 ----------
        flat = stacked_with_w0.reshape(N, -1)              # (N, D)
        D = flat.size(1)

        if keep_ratio <= 0.0:
            trimmed_flat = torch.zeros_like(flat)
        elif keep_ratio >= 1.0:
            trimmed_flat = flat.clone()
        else:
            k = max(1, int(D * keep_ratio))
            abs_flat = flat.abs()
            # 对每个任务向量单独取 top-k
            _, topk_idx = torch.topk(abs_flat, k, dim=1, largest=True, sorted=False)
            keep_mask = torch.zeros_like(flat, dtype=torch.bool)
            keep_mask.scatter_(1, topk_idx, True)
            trimmed_flat = flat * keep_mask
        
        trimmed = trimmed_flat.view_as(stacked_with_w0)    # (N, ...)

        # === Step 3. 符号方向判断（由约束决定） ===
        mean_BA = stacked.mean(dim=0)  # 平均 (B@A)
        positive_condition = mean_BA > (-weight_init)
        keep_mask_condition = torch.where(
        positive_condition,
        stacked > (-weight_init),
        stacked < (-weight_init),
        ) 
        
        trimmed = torch.where(keep_mask_condition, trimmed, torch.zeros_like(trimmed))#*(trimmed是N*矩阵，为0的地方是0，不为0的地方是W0+B@A)

        sum_vals = trimmed.sum(dim=0)#*若不为0，则是k*W0+B1A1+B2A2.....+BKAK
        count = (trimmed != 0).sum(dim=0)#*count本身，上一行的k
        delta = torch.zeros_like(sum_vals) 
        nonzero_mask = count > 0
        delta[nonzero_mask] = sum_vals[nonzero_mask] / count[nonzero_mask]- weight_init[nonzero_mask] 
               
        return delta


    def ties_merge_param(
        self,
        param_list,
        norm_weights=None,
        keep_ratio: float = None,
        lam: float = None,
    ):
        """
        TIES 融合：
        1) Trim：对每个任务向量（每个客户端的参数）做 per-task 的 top-k% 幅值剪枝；
        2) Elect Sign：对每个参数，汇总修剪后值的总和，取总和符号作为最终符号；
        3) Disjoint Merge：只平均与最终符号一致的任务向量值，再乘缩放系数 λ。

        param_list: List[Tensor]，每个 tensor 形状相同（比如一个 LoRA A 或 B）
        norm_weights: List[float] 或 None，用于联邦场景下加权平均（样本数权重）
        keep_ratio: 保留比例 k，None 时使用 self.ties_keep_ratio
        lam: 缩放系数 λ，None 时使用 self.ties_lambda
        """
        assert len(param_list) > 0
        stacked = torch.stack(param_list, dim=0)  # (N, ...)
        device = stacked.device
        dtype = stacked.dtype
        N = stacked.size(0)

        if keep_ratio is None:
            keep_ratio = self.ties_keep_ratio
        keep_ratio = float(keep_ratio)

        if lam is None:
            lam = self.ties_lambda
        lam = float(lam)

        # ---------- 1) Trim：每个任务向量自身做 top-k% 剪枝 ----------
        flat = stacked.reshape(N, -1)              # (N, D)
        D = flat.size(1)

        if keep_ratio <= 0.0:
            trimmed_flat = torch.zeros_like(flat)
        elif keep_ratio >= 1.0:
            trimmed_flat = flat.clone()
        else:
            k = max(1, int(D * keep_ratio))
            abs_flat = flat.abs()
            # 对每个任务向量单独取 top-k
            _, topk_idx = torch.topk(abs_flat, k, dim=1, largest=True, sorted=False)
            keep_mask = torch.zeros_like(flat, dtype=torch.bool)
            keep_mask.scatter_(1, topk_idx, True)
            trimmed_flat = flat * keep_mask

        trimmed = trimmed_flat.view_as(stacked)    # (N, ...)

        # ---------- 2) Elect Sign：根据修剪后总和选举符号 ----------
        sum_trimmed = trimmed.sum(dim=0)           # (...)
        final_sign = sum_trimmed.sign()            # -1, 0, 1

        # ---------- 3) Disjoint Merge：只平均与最终符号一致的值 ----------
        sign_trimmed = trimmed.sign()
        # 只允许：非零 & 符号与 final_sign 相同
        fs = final_sign.unsqueeze(0)               # (1, ...)
        agree_mask = (sign_trimmed == fs) & trimmed.ne(0)

        if norm_weights is not None:
            w = torch.tensor(norm_weights, device=device, dtype=dtype)  # (N,)
            w = w.view(N, *([1] * (stacked.dim() - 1)))                 # (N, 1, 1, ...)
            weighted_vals = trimmed * w * agree_mask
            weighted_sum = weighted_vals.sum(dim=0)                     # (...)
            weight_sum = (agree_mask * w).sum(dim=0)                    # (...)

            merged = torch.zeros_like(final_sign, dtype=dtype)
            nonzero_mask = weight_sum > 0
            merged[nonzero_mask] = weighted_sum[nonzero_mask] / weight_sum[nonzero_mask]
        else:
            vals = trimmed * agree_mask
            sum_vals = vals.sum(dim=0)
            count = agree_mask.sum(dim=0)

            merged = torch.zeros_like(final_sign, dtype=dtype)
            nonzero_mask = count > 0
            merged[nonzero_mask] = sum_vals[nonzero_mask] / count[nonzero_mask]

        # 缩放系数 λ
        if lam != 1.0:
            merged = merged * lam

        return merged


    def init_lora_params(self, network, r):
        for name, param in network.named_parameters():
            param.requires_grad = False  # freeze all the parameters
            if not self.lora_head and "head" in name:
                self.head_keys.append(name)
            if ("qkv" in name and "weight" in name) or (
                ("mlp" in name and "weight" in name)
                or ("proj" in name and "weight" in name and "attn" in name)
                or (self.lora_head and "head" in name and "weight" in name)
            ):
                self.lora_keys.append(name)
                self.lora_params[name] = {name: [param.shape[1], param.shape[0]]}
                self.old_delta[name] = nn.Parameter(
                    torch.zeros(param.shape[0], param.shape[1]), requires_grad=False
                ).to(self.device)
                self.cur_B[name] = nn.Parameter(torch.zeros(param.shape[0], r), requires_grad=True).to(self.device)
                self.cur_A[name] = nn.Parameter(torch.zeros(r, param.shape[1]), requires_grad=True).to(self.device)
                nn.init.kaiming_uniform_(self.cur_A[name], a=math.sqrt(5))


    def init_matrices(self, reverse=False):
        for key in self.lora_keys:
            self.cur_B[key] = nn.Parameter(torch.zeros_like(self.cur_B[key]), requires_grad=True).to(self.device)
            self.cur_A[key] = nn.Parameter(torch.zeros_like(self.cur_A[key]), requires_grad=True).to(self.device)
            if not reverse:
                nn.init.kaiming_uniform_(self.cur_A[key], a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.cur_B[key], a=math.sqrt(5))

    # used for testing, using a functional_call() to call the network with self.optimization_dict parameters
    def set_optimization(self):
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        for key in self.optimization_dict.keys():
            self.optimization_dict[key] = self.optimization_dict[key].to(self.device)
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = False
            self.cur_A[key].requires_grad = False
            self.cur_A[key] = self.cur_A[key].to(self.device)
            self.cur_B[key] = self.cur_B[key].to(self.device)
        if "run_sum" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    self.optimization_dict[key] += self.old_delta[key]
                self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        elif "run_mean" in self.cl_merge:
            for key in self.lora_keys:
                if self.cur_task > 0:
                    tmp = (self.old_delta[key] * self.cur_task) + self.cur_B[key] @ self.cur_A[key]
                    self.optimization_dict[key] += tmp / (self.cur_task + 1)
                else:
                    self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        elif "individual" in self.cl_merge:
            if "sum" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.cur_B[key] @ self.cur_A[key]
                        self.optimization_dict[key] += tmp
                    else:
                        self.optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
            elif "mean" in self.cl_merge:
                for key in self.lora_keys:
                    if self.cur_task > 0:
                        tmp = (self.old_delta[key] * self.cur_task) + self.cur_B[key].detach() @ self.cur_A[
                            key
                        ].detach()
                        self.optimization_dict[key] += tmp / (self.cur_task + 1)
                    else:
                        self.optimization_dict[key] += self.cur_B[key].detach() @ self.cur_A[key].detach()

        else:
            raise ValueError("Invalid cl_merge type")

    def debug_matrices_create(self):
        for key in self.lora_keys:
            self.pre_B[key] = self.cur_B[key].detach().clone()
            self.pre_A[key] = self.cur_A[key].detach().clone()
        self.pre_head = deepcopy(self.network.model.head.state_dict())

    def debug_matrices_compare(self):
        num_equal_B = 0
        num_equal_A = 0
        for key in self.lora_keys:
            if torch.allclose(self.pre_B[key], self.cur_B[key]):
                num_equal_B += 1
                print(f"B equal with key {key}")
            if torch.allclose(self.pre_A[key], self.cur_A[key]):
                num_equal_A += 1
                print(f"A equal with key {key}")
        print(f"Number of equal B matrices: {num_equal_B} out of {len(self.lora_keys)}")
        print(f"Number of equal A matrices: {num_equal_A} out of {len(self.lora_keys)}")
        if not self.lora_head:
            num_equal_head = 0
            for key in self.pre_head.keys():
                if torch.allclose(self.pre_head[key], self.network.model.head.state_dict()[key]):
                    num_equal_head += 1
                    print(f"Head equal with key {key}")
                else:
                    print(
                        f"Head not equal with key {key}, distance: {torch.dist(self.pre_head[key], self.network.model.head.state_dict()[key])}"
                    )
            print(f"Number of equal head matrices: {num_equal_head} out of {len(self.head_keys)}")

    def get_optimization_dict(self, fabric=True):
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key].requires_grad = True
                optimization_dict[key] = self.head[key]
        for key in self.lora_keys:
            self.old_delta[key].requires_grad = False
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            if self.cur_task > 0 and not "individual" in self.cl_merge:
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict

    def get_dummy_optimization_dict(self):
        return deepcopy(dict(self.network.named_parameters()))

    def begin_task(self, n_classes_per_task: int):
        super().begin_task(n_classes_per_task)
        if self.cur_task > 0:
            if self.cl_merge == "run_sum":
                for key in self.lora_keys:
                    self.old_delta[key] += self.cur_B[key].detach() @ self.cur_A[key].detach()
            elif self.cl_merge == "run_mean" or "individual" in self.cl_merge:
                for key in self.lora_keys:
                    self.old_delta[key] = (
                        self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                    ) / self.cur_task
            else:
                raise ValueError("Invalid cl_merge type")
            self.init_matrices()

    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
        self.old_delta = deepcopy(server_info["old_delta"])
        if not self.lora_head:
            self.network.model.head.load_state_dict(server_info["head"])
            # for p in self.network.model.head.parameters():
            #    p.requires_grad = True
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }

        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        if not self.lora_head:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()) + list(self.head.values()),
                lr=self.lr,
                weight_decay=self.wd_reg,
            )
        else:
            self.optimizer = OptimizerClass(
                list(self.cur_B.values()) + list(self.cur_A.values()), lr=self.lr, weight_decay=self.wd_reg
            )
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, update: bool = True) -> float:
        self.optimizer.zero_grad()
        optimization_dict = self.get_optimization_dict()
        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = functional_call(self.network, optimization_dict, inputs)[
                :, self.cur_offset : self.cur_offset + self.cpt
            ]
            loss = self.loss(outputs, labels - self.cur_offset)

        if update:
            self.fabric.backward(loss)
            # torch.nn.utils.clip_grad_norm_(list(self.cur_B.values()) + list(self.cur_A.values()), 1.0)
            if self.clip_grad:
                try:
                    self.fabric.clip_gradients(self.network, self.optimizer, max_norm=1.0, norm_type=2)
                except:
                    pass
            self.optimizer.step()
        return loss.item()

    def forward(self, x, fabric=True):
        if fabric:
            return functional_call(self.network, self.optimization_dict, x)
        return functional_call(self.network.module, self.optimization_dict, x)

    def forward_2(self, x):
        return self.network.module(x)

    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.old_delta[key] = self.old_delta[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            client_info["head"] = deepcopy(self.network.model.head.state_dict())
        return client_info

    def get_server_info(self):
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        server_info = {
            "cur_A": deepcopy(self.cur_A),
            "cur_B": deepcopy(self.cur_B),
        }
        if getattr(self, "old_delta", None) is not None:
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
            server_info["old_delta"] = self.old_delta
        if not self.lora_head:
            server_info["head"] = deepcopy(self.network.model.head.state_dict())
        return server_info

    def end_round_client(self, dataloader: DataLoader):
        if not self.lora_head:
            sd = self.network.state_dict()
            for key in self.head_keys:
                sd[key] = self.head[key]
            self.network.load_state_dict(sd)

    def end_round_server(self, client_info: List[dict]):
        if self.avg_type == "weighted":
            total_samples = sum([client["num_train_samples"] for client in client_info])
            norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
        else:
            weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
            norm_weights = [w / sum(weights) for w in weights]
        cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
        cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
        if not self.lora_head:
            heads = [client["head"] for client in client_info]  # list of head matrices for all clients
            head_sd = self.network.model.head.state_dict()
            for key in head_sd.keys():
                head_sd[key] = torch.stack(
                    [head[key] * norm_weight for head, norm_weight in zip(heads, norm_weights)]
                ).sum(0)
            self.network.model.head.load_state_dict(head_sd)

        if len(client_info) > 0:
            for key in self.lora_keys:
                # 每个客户端在该 key 上的 LoRA 参数（任务向量）
                B_list = [client_B[key] for client_B in cl_B]
                A_list = [client_A[key] for client_A in cl_A]

                merged_B = self.ties_merge_param(
                    B_list,
                    norm_weights=norm_weights,          # 联邦权重
                    keep_ratio=self.ties_keep_ratio,    # Trim 的 top-k%
                    lam=self.ties_lambda,               # 缩放 λ
                )
                merged_A = self.ties_merge_param(
                    A_list,
                    norm_weights=norm_weights,
                    keep_ratio=self.ties_keep_ratio,
                    lam=self.ties_lambda,
                )

                self.cur_B[key] = nn.Parameter(merged_B)
                self.cur_A[key] = nn.Parameter(merged_A)

            self.set_optimization()

    def end_task_server(self, client_info: List[dict] = None):
        with torch.no_grad():
            optimization_dict = deepcopy(self.network.state_dict())#*这就是拷贝的原始预训练权重
            if getattr(self, "run_weights", None) is None:
                self.run_weights_A = {
                    key:[] for key in self.lora_keys
                    if "weight" in key and not "head" in key
                }
                self.run_weights_B = {
                    key:[] for key in self.lora_keys
                    if "weight" in key and not "head" in key
                }
                self.run_weights = {
                    key:[] for key in self.lora_keys
                    if "weight" in key and not "head" in key
                }
            for key in self.lora_keys:
                if "weight" in key and not "head" in key:
                    B = self.cur_B[key].to(self.device)
                    A = self.cur_A[key].to(self.device)            
                    # self.run_weights_A[key].append(A) 
                    # self.run_weights_B[key].append(B)
                    task_weight_init= optimization_dict[key]
                    self.run_weights[key].append(B@A) 

                    
                    if self.cur_task > 0:
                        # merge_task_weight_A= self.ties_merge_param(
                        #     self.run_weights_A[key],
                        #     norm_weights=None,#*是否加权？
                        #     keep_ratio=self.ties_keep_ratio,
                        #     lam=self.ties_lambda,
                        # )
                        # merge_task_weight_B= self.ties_merge_param(
                        #     self.run_weights_B[key],
                        #     norm_weights=None,#*是否加权？
                        #     keep_ratio=self.ties_keep_ratio,
                        #     lam=self.ties_lambda,
                        # )  
                        merge_task_weight= self.ties_merge_param_with_w0(
                            param_list=self.run_weights[key],
                            weight_init=task_weight_init,
                            keep_ratio=self.ties_keep_ratio,
                            
                        )                                                
                        optimization_dict[key] += merge_task_weight#*预训练模型+合并后的AB矩阵
                        
            if self.cur_task > 0:
                self.optimization_dict = optimization_dict
        
        print("my task merge")


    def to(self, device="cpu"):
        self.network.to(device)
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].to(device)
            self.cur_A[key] = self.cur_A[key].to(device)
            self.old_delta[key] = self.old_delta[key].to(device)
        if not self.lora_head:
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
        return self
