import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import NUM_CLASSES, DEVICE
from utils import compute_class_weights


class DareTiesOptimizer:
    def __init__(self, base_arch, experts, val_loader, y_true, device, p_drop=0.5, lr=0.1, epochs=50):
        self.device = device
        self.val_loader = val_loader
        self.y_true = y_true
        self.epochs = epochs
        self.lr = lr
        self.base_state_dict = copy.deepcopy(experts[0].state_dict())
        for key in self.base_state_dict:
            for i in range(1, len(experts)):
                self.base_state_dict[key] = (
                    self.base_state_dict[key] + experts[i].state_dict()[key]
                )
            self.base_state_dict[key] = self.base_state_dict[key] / len(experts)

        total_delta = {k: torch.zeros_like(v) for k, v in self.base_state_dict.items()}
        temp_tasks = []

        rescale = 1.0 / (1.0 - p_drop)   

        for exp in experts:
            task_dict = {}
            est = exp.state_dict()
            for k, v in est.items():
                tau = v - self.base_state_dict[k]
                if "norm" in k:
                    tau_hat = tau
                else:
                    drop_mask = torch.bernoulli(
                        torch.full_like(tau, p_drop)
                    )
                    tau_hat = rescale * (1.0 - drop_mask) * tau
                task_dict[k] = tau_hat
                total_delta[k] = total_delta[k] + tau_hat
            temp_tasks.append(task_dict)

        self.final_task_vectors = []
        for i in range(len(experts)):
            final_task_dict = {}
            for k in temp_tasks[i]:
                tau_hat     = temp_tasks[i][k]
                majority_sign = torch.sign(total_delta[k])
                agreement   = torch.sign(tau_hat) == majority_sign
                tau_tilde   = torch.where(
                    agreement, tau_hat, torch.zeros_like(tau_hat)
                )
                final_task_dict[k] = tau_tilde
            self.final_task_vectors.append(final_task_dict)

        self.scaling_factors = []
        for i in range(len(experts)):
            s_dict = {}
            for k in self.base_state_dict.keys():
                s_dict[k] = nn.Parameter(
                    torch.tensor(1.0, device=self.device)
                )
            self.scaling_factors.append(s_dict)

        self.model = copy.deepcopy(base_arch)


    def _functional_forward(self, x, state_dict):
        
        x = nn.functional.linear(
            x, state_dict['layer1.weight'], state_dict['layer1.bias']
        )
        x = nn.functional.layer_norm(
            x, (128,),
            weight=state_dict['norm1.weight'], bias=state_dict['norm1.bias']
        )
        x = nn.functional.gelu(x)

        x = nn.functional.linear(
            x, state_dict['layer2.weight'], state_dict['layer2.bias']
        )
        x = nn.functional.layer_norm(
            x, (64,),
            weight=state_dict['norm2.weight'], bias=state_dict['norm2.bias']
        )
        x = nn.functional.gelu(x)

        x = nn.functional.linear(
            x, state_dict['layer3.weight'], state_dict['layer3.bias']
        )
        x = nn.functional.layer_norm(
            x, (32,),
            weight=state_dict['norm3.weight'], bias=state_dict['norm3.bias']
        )
        x = nn.functional.gelu(x)

        return nn.functional.linear(
            x, state_dict['output_layer.weight'], state_dict['output_layer.bias']
        )
        '''
        
        x = nn.functional.linear(
            x, state_dict['layer1.weight'], state_dict['layer1.bias']
        )
        x = nn.functional.layer_norm(
            x, (30,),
            weight=state_dict['norm1.weight'], bias=state_dict['norm1.bias']
        )
        x = nn.functional.gelu(x)
        return nn.functional.linear(
            x, state_dict['output_layer.weight'], state_dict['output_layer.bias']
        )
        '''
        
    def _merge_state_dict(self):
        merged = {}
        for k, base_param in self.base_state_dict.items():
            param = base_param.clone()
            for i in range(len(self.scaling_factors)):
                param = param + self.scaling_factors[i][k] * self.final_task_vectors[i][k]
            merged[k] = param
        return merged


    def optimize(self):
        all_params = [p for s_dict in self.scaling_factors for p in s_dict.values()]
        optimizer  = optim.Adam(all_params, lr=self.lr)

        class_weights = compute_class_weights(self.y_true, NUM_CLASSES)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        print(f"   [DARE-TIES] Optimising scaling factors for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            for Xb, yb in self.val_loader:
                optimizer.zero_grad()
                curr_state = self._merge_state_dict()
                out  = self._functional_forward(Xb, curr_state)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for p in all_params:
                        p.clamp_(0.0, 1.0)

        with torch.no_grad():
            final_state = self._merge_state_dict()
        self.model.load_state_dict(final_state)
        return self.model
