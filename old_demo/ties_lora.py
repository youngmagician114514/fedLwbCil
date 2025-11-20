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
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora_pre import Lora
from tqdm import tqdm
import math
from transformers import AutoModelForSequenceClassification, T5ForSequenceClassification, T5Model


@register_model("ties_lora")
class TiesLora(Lora):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 0.0003,
        clip_grad: str_to_bool = False,
        wd_reg: float = 0.1,
        avg_type: str = "weighted",
        lora_alpha: float = 1,
        r: int = 16,
        lora_head: str_to_bool = False,
        cl_merge: str = "individual_mean",
        lr_B: float = -1,
        lr_A: float = -1,
        only_square: int = 0,
        train_bias: str = "all",
    ) -> None:
        self.is_server = False
        lr_back = lr_B if lr_B > 0 else lr
        Lora.__init__(
            self,
            fabric,
            network,
            device,
            optimizer,
            lr,
            clip_grad,
            wd_reg,
            avg_type,
            lora_alpha,
            r,
            lora_head,
            cl_merge,
        )
        self.cur_round = 0
        self.optimization_dict = deepcopy(dict(self.network.state_dict()))
        lr_back = []
        lr_back.append(lr)
        self.lr_back = lr_back
        del self.old_delta
        
        
    #*用于处理模型中的参数，分离Lora的头部和身体
    def split_backbone_head(self, split=False):
        Bs = []
        As = []
        head_params = []
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
            Bs.append(self.cur_B[key])
            As.append(self.cur_A[key])
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
            self.head[key].requires_grad = True
            head_params.append(self.head[key])
        if split:
            return Bs, As, head_params
        else:
            return Bs + As, head_params        
        
    #*把lora参数合并到本体上
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
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        return optimization_dict 
    
    #*这个函数详细演示了计算损失过程
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
    
    
    #*每个任务开始前初始化AB矩阵，并积累之前的矩阵
    def begin_task(self, n_classes_per_task: int):
        BaseModel.begin_task(self, n_classes_per_task)
        # server
        if self.cur_task > 0 and self.is_server:
            self.to("cpu")
            for key in self.lora_keys:
                # filler in order to test something meaningful after each comm round (not the last one)
                self.old_delta[key] = (
                    self.old_delta[key] * (self.cur_task - 1) + self.cur_B[key].detach() @ self.cur_A[key].detach()
                ) / self.cur_task
            self.to(self.device)
            self.init_matrices(freeze_A=False)
        else:
            if self.cur_task > 0:
                self.init_matrices(freeze_A=False)
        self.cur_round = 0
        
    #*每个cilent端在开始之前加载sever端的参数,更改选择同时训练AB矩阵   
    def begin_round_client(self, dataloader: DataLoader, server_info: dict):
        self.network.train()
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        for key in self.lora_keys:
            self.cur_B[key].requires_grad = True
            self.cur_A[key].requires_grad = True
        # self.optimization_dict = deepcopy(server_info["old_delta"])
        self.old_delta = server_info["old_delta"]
        if not self.lora_head:
            if isinstance(self.network.module.model, T5Model):
                self.network.head.load_state_dict(server_info["head"])
            else:
                self.network.model.head.load_state_dict(server_info["head"])
            # for p in self.network.model.head.parameters():
            #    p.requires_grad = True
            self.head = {
                key: nn.Parameter(self.network.state_dict()[key].clone().detach(), requires_grad=True).to(self.device)
                for key in self.head_keys
            }
        BAs, head_params = self.split_backbone_head(split=False)
        back = BAs
        back_lr = self.lr_back[0] 
        params = [{"params": back, "lr": back_lr}, {"params": head_params}]
        OptimizerClass = getattr(torch.optim, self.optimizer_str)
        self.optimizer = OptimizerClass(params, lr=self.lr, weight_decay=self.wd_reg)
        self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        self.cur_round += 1

    #*如果没有old_delta,则初始化old_delta 
    def begin_round_server(self):
        if getattr(self, "old_delta", None) is None:
            self.old_delta = {}
            for key in self.lora_keys:
                self.old_delta[key] = torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1])
        self.cur_round += 1     
        

    def end_round_client(self, dataloader: DataLoader):
        # self.network.eval()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            self.optimizer = None
        Lora.end_round_client(self, dataloader)
        self.set_optimization()
        for key in self.head_keys:
            self.optimization_dict[key] = self.head[key]
        self.to("cpu", only_trainable=False)
     
     #*如果有梯度用梯度，没梯度用参数绝对值
    @staticmethod
    def compute_importance(matrix: torch.Tensor, client_grad: torch.Tensor | None) -> torch.Tensor:
        """
        返回与 matrix 同形状的非负重要度权重。
        - 有梯度：用 |grad| 的均值/合并到与 matrix 同形状
        - 无梯度：退化为 |matrix|
        """
        if client_grad is None:
            return matrix.abs()

        # client_grad 可能来自累计或多 batch；把它规整到与 matrix 同形状
        g = client_grad
        # 若维度可广播，直接取绝对值；否则按最后一维聚合
        try:
            imp = g.abs()
            if imp.shape != matrix.shape:
                # 典型场景：batch 维或样本维 -> 做均值到与 matrix 可广播/相同形状
                while imp.dim() > matrix.dim():
                    imp = imp.mean(dim=0)
                # 逐维对齐：若维度不等但能整除，做均值压缩
                for d in range(imp.dim()):
                    if imp.shape[d] != matrix.shape[d]:
                        imp = imp.mean(dim=d, keepdim=True)
                imp = imp.expand_as(matrix)
            return imp
        except Exception:
            # 兜底
            return matrix.abs()
     
    def end_round_server(self, client_info: List[dict]):
        self.is_server = True
        
        with torch.no_grad():
            if self.avg_type == "weighted":
                total_samples = sum([client["num_train_samples"] for client in client_info])
                norm_weights = [client["num_train_samples"] / total_samples for client in client_info]
            else:
                weights = [1 if client["num_train_samples"] > 0 else 0 for client in client_info]
                norm_weights = [w / sum(weights) for w in weights]
            cl_B = [client["cur_B"] for client in client_info]  # list of B matrices for all clients
            cl_A = [client["cur_A"] for client in client_info]  # list of A matrices for all clients
            self.to("cpu")     
 
             # import gc
            dtype = torch.float64 
            # from time import time
            # start = time()
            total_mem = torch.cuda.get_device_properties(0).total_memory
            
            sparsity_ratio = 0.2#*后面记得加进参数里
            
            
            #*后续改成在客户端本地就完成稀疏化
            for key in self.lora_keys:
                # 收集所有客户端的 A 矩阵和对应的梯度（假设客户端信息包含梯度）
                client_As = [cl_A[i][key].to(dtype) for i in range(len(cl_A))]
                client_grads = [client_info[i].get("grads_A", {}).get(key, None) for i in range(len(client_info))]
                
                # 计算每个客户端参数的重要性
                importances = [self.compute_importance(A, grad) for A, grad in zip(client_As, client_grads)]
                
                # 稀疏化：对每个客户端，保留 top-K 重要参数
                sparse_As = []
                for A, imp in zip(client_As, importances):
                    # 计算阈值（按重要性排序，取 top sparsity_ratio）
                    flat_imp = imp.flatten()
                    k = max(1, int(len(flat_imp) * sparsity_ratio))
                    threshold = torch.topk(flat_imp, k).values[-1]
                    # 低于阈值的参数置零
                    mask = imp >= threshold
                    sparse_A = A * mask
                    sparse_As.append(sparse_A)
                
                # #* 实现ties融合，稀疏化后的 A 矩阵，
                # weights_sum_A = 0
                # for sparse_A in sparse_As:
                #     weights_sum_A += sparse_A
                
                weights_sum_A = torch.stack(sparse_As, dim=0)
                
                sum_pos_A = torch.clamp(weights_sum_A, min=0).sum(dim=0)  # 同号加总
                sum_neg_A = torch.clamp(weights_sum_A, max=0).sum(dim=0)  # 同号加总(≤0)
                
                w = torch.tensor(norm_weights, dtype=weights_sum_A.dtype, device=weights_sum_A.device)\
                    .view(-1, *([1] * (weights_sum_A.dim() - 1)))
                pos_mask = (weights_sum_A > 0).to(weights_sum_A.dtype)
                neg_mask = (weights_sum_A < 0).to(weights_sum_A.dtype)

                W_pos = (w * pos_mask).sum(dim=0)
                W_neg = (w * neg_mask).sum(dim=0)

                pos_val = sum_pos_A / (W_pos + 1e-12)
                neg_val = sum_neg_A / (W_neg + 1e-12)
                
                choose_pos = (pos_val.abs() >= neg_val.abs())
                merged_A = torch.where(choose_pos, pos_val, neg_val)            
                
                # 写回当前全局 A
                self.cur_A[key] = nn.Parameter(merged_A.to(torch.float32).to("cpu"))
                
                
                
            for key in self.lora_keys:
                client_Bs = [cl_B[i][key].to(dtype) for i in range(len(cl_B))]
                client_grads = [client_info[i].get("grads_B", {}).get(key, None) for i in range(len(client_info))]
                
                importances = [self.compute_importance(B, grad) for B, grad in zip(client_Bs, client_grads)]
                
                sparse_Bs = []
                for B, imp in zip(client_Bs, importances):
                    flat_imp = imp.flatten()
                    k = max(1, int(len(flat_imp) * sparsity_ratio))
                    threshold = torch.topk(flat_imp, k).values[-1]
                    mask = imp >= threshold
                    sparse_B = B * mask
                    sparse_Bs.append(sparse_B)
                
                #* 实现ties融合，稀疏化后的 B 矩阵，
                # weights_sum_B = 0
                # for sparse_B in sparse_Bs:
                #     weights_sum_B += sparse_B
                    
                weights_sum_B = torch.stack(sparse_Bs, dim=0)    
                                                        
                sum_pos_B = torch.clamp(weights_sum_B, min=0).sum(dim=0)  # 同号加总
                sum_neg_B = torch.clamp(weights_sum_B, max=0).sum(dim=0)  # 同号加总(≤0)
                
                # 仅对“同号参与者”做平均
                w = torch.tensor(norm_weights, dtype=weights_sum_B.dtype, device=weights_sum_B.device)\
                    .view(-1, *([1] * (weights_sum_B.dim() - 1)))
                pos_mask = (weights_sum_B > 0).to(weights_sum_B.dtype)
                neg_mask = (weights_sum_B < 0).to(weights_sum_B.dtype)

                W_pos = (w * pos_mask).sum(dim=0)
                W_neg = (w * neg_mask).sum(dim=0)

                pos_val = sum_pos_B / (W_pos + 1e-12)
                neg_val = sum_neg_B / (W_neg + 1e-12)
                
                choose_pos = (pos_val.abs() >= neg_val.abs())
                merged_B = torch.where(choose_pos, pos_val, neg_val)            
                
                # 写回当前全局 B
                self.cur_B[key] = nn.Parameter(merged_B.to(torch.float32).to("cpu"))

        #*处理头部参数
        keys = list(self.network.state_dict().keys())
        sd = self.network.state_dict()
        for key in keys:
                sd[key] = torch.stack(
                    [client["state_dict"][key] * norm_weight for client, norm_weight in zip(client_info, norm_weights)]
                ).sum(0)
        self.network.load_state_dict(sd)
        
        
        if getattr(self, "old_delta", None) is None:
            self.old_delta = {key: torch.zeros(self.cur_B[key].shape[0], self.cur_A[key].shape[1], requires_grad=False) 
                             for key in self.lora_keys}
        self.detach()
        self.set_optimization()
        for key in self.head_keys:
            self.optimization_dict[key] = self.network.state_dict()[key].to(self.device)
        self.to(self.device)
        torch.cuda.empty_cache()        
            
    #*这里应该要用知识蒸馏之类的处理
    def end_task_server(self, client_info: List[dict] = None):
        print("end_task_server")

            
    def get_client_info(self, dataloader: DataLoader):
        for key in self.lora_keys:
            self.cur_B[key] = self.cur_B[key].detach()
            self.cur_A[key] = self.cur_A[key].detach()
        client_info = {
            "cur_A": self.cur_A,
            "cur_B": self.cur_B,
            "num_train_samples": len(dataloader.dataset.data),
        }
        if not self.lora_head:
            if isinstance(self.network.module.model, T5Model):
                client_info["head"] = self.network.head.state_dict()
            else:
                client_info["head"] = self.network.model.head.state_dict()
        client_info["state_dict"] = self.network.state_dict()
        return client_info     

    def end_task_client(self, dataloader: DataLoader, server_info: dict):
        self.cur_B = deepcopy(server_info["cur_B"])
        self.cur_A = deepcopy(server_info["cur_A"])
        if isinstance(self.network.module.model, T5Model):
            self.network.head.load_state_dict(server_info["head"])
        else:
            self.network.model.head.load_state_dict(server_info["head"])
        self.set_optimization()
        self.to("cpu", only_trainable=False)

    def set_optimization_cur_task(self, fabric=True):
        self.detach()
        self.to(self.device)
        sd = self.network.state_dict()
        if fabric:
            optimization_dict = deepcopy(dict(self.network.state_dict()))
        else:
            optimization_dict = deepcopy(dict(self.network.module.state_dict()))
        for key in self.lora_keys:
            if self.cur_task > 0 and not "individual" in self.cl_merge and not "fisher" in self.cl_merge:
                self.old_delta[key] = self.old_delta[key].to(self.device)
                optimization_dict[key] += self.old_delta[key]
            optimization_dict[key] += self.cur_B[key] @ self.cur_A[key]
        self.optimization_dict = optimization_dict


    def detach(self):
        for key in self.lora_keys:
            self.cur_A[key] = self.cur_A[key].detach()
            self.cur_B[key] = self.cur_B[key].detach()
        for key in self.head_keys:
            self.head[key] = self.head[key].detach()
        if getattr(self, "old_delta", None) is not None:
            for key in self.lora_keys:
                self.old_delta[key] = self.old_delta[key].detach()
    
        
    def to(self, device="cpu", only_trainable=True):
        if "cpu" in device or not only_trainable:  # we move everything to the cpu
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
            for key in self.head_keys:
                self.head[key] = self.head[key].to(device)
            if getattr(self, "old_delta", None) is not None:
                for key in self.lora_keys:
                    self.old_delta[key] = self.old_delta[key].to(device)
        else:  # we move only the trainable parameters to the device
            self.network = self.network.to(device)
            for key in self.lora_keys:
                self.cur_A[key] = self.cur_A[key].to(device)
                self.cur_B[key] = self.cur_B[key].to(device)
        return self