import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import NUM_CLASSES, DEVICE
from utils import compute_class_weights


class DistillationOptimizer:

    def __init__(self, base_arch, teachers, val_loader, y_true, device,
                 lr=0.003, epochs=130, temperature=4.0, alpha=0.5):
        self.student = copy.deepcopy(base_arch)
        self.teachers = [copy.deepcopy(t) for t in teachers]
        for t in self.teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad = False

        self.val_loader = val_loader
        self.y_true = y_true
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.T = temperature
        self.alpha = alpha

    def _teacher_soft_targets(self, x):
        with torch.no_grad():
            probs = [F.softmax(t(x) / self.T, dim=1) for t in self.teachers]
            return torch.stack(probs, dim=0).mean(dim=0)

    def optimize(self):
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr)
        hard_criterion = nn.CrossEntropyLoss(
            weight=compute_class_weights(self.y_true, NUM_CLASSES)
        )

        print(f"   [Distillation] Distilling {len(self.teachers)} teachers into student "
              f"for {self.epochs} epochs (T={self.T}, alpha={self.alpha})...")

        for _ in range(self.epochs):
            self.student.train()
            for Xb, yb in self.val_loader:
                optimizer.zero_grad()

                soft_targets = self._teacher_soft_targets(Xb)             
                student_logits = self.student(Xb)

                student_log_T = F.log_softmax(student_logits / self.T, dim=1)
                soft_loss = F.kl_div(student_log_T, soft_targets,
                                     reduction='batchmean') * (self.T * self.T)
                hard_loss = hard_criterion(student_logits, yb)          

                loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
                loss.backward()
                optimizer.step()

        self.student.eval()
        return self.student
