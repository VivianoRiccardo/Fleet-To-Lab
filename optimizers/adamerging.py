import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call 

from config import NUM_CLASSES, DEVICE
from utils import compute_class_weights


class AdaMergingOptimizer:


    def __init__(self, base_arch, shared_init, experts, val_loader, y_true,
                 device, lr=0.003, epochs=130, init_coeff=0.3):
        self.device = device
        self.val_loader = val_loader
        self.y_true = y_true
        self.epochs = epochs
        self.lr = lr
        self.K = len(experts)

        self.model = copy.deepcopy(base_arch).to(device).eval()

        base_sd = shared_init.state_dict()
        self.keys = list(base_sd.keys())
        self.base_state = {k: base_sd[k].detach().clone().to(device) for k in self.keys}

        self.task_vectors = []
        for exp in experts:
            est = exp.state_dict()
            self.task_vectors.append(
                {k: est[k].detach().to(device) - self.base_state[k] for k in self.keys}
            )

        self.coeffs = [
            {k: nn.Parameter(torch.tensor(float(init_coeff), device=device))
             for k in self.keys}
            for _ in range(self.K)
        ]

    def _merge_state_dict(self):
        merged = {}
        for k in self.keys:
            param = self.base_state[k].clone()
            for i in range(self.K):
                param = param + self.coeffs[i][k] * self.task_vectors[i][k]
            merged[k] = param
        return merged

    def _forward(self, x, sd):
        return functional_call(self.model, sd, (x,))

    def optimize(self):
        params = [p for c in self.coeffs for p in c.values()]
        optimizer = optim.Adam(params, lr=self.lr)
        class_w = compute_class_weights(self.y_true, NUM_CLASSES).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_w)

        print(f"   [AdaMerging] Learning {len(params)} layer-wise coefficients "
              f"for {self.epochs} epochs...")

        for _ in range(self.epochs):
            for Xb, yb in self.val_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self._forward(Xb, self._merge_state_dict())
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            self.model.load_state_dict(self._merge_state_dict())
        return self.model
