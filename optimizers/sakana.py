import copy
import numpy as np
import torch
import torch.nn as nn

from config import NUM_CLASSES, DEVICE
from utils import model_to_array, get_predictions, compute_fitness_cross_entropy


class SimpleCMAES:
    def __init__(self, n_params, sigma_init=1.0 / 6.0, pop_size=None):
        self.N = n_params
        self.sigma = sigma_init
        self.mu = np.full(self.N, 0.5)

        if pop_size is None:
            self.pop_size = int(4 + 3 * np.log(self.N))
        else:
            self.pop_size = pop_size

        self.mu_eff_ratio = 0.5
        self.n_survivors   = max(1, int(self.pop_size * self.mu_eff_ratio))
        weights = np.log(self.n_survivors + 0.5) - np.log(
            np.arange(1, self.n_survivors + 1)
        )
        self.weights = weights / np.sum(weights)
        self.mueff   = 1.0 / np.sum(self.weights ** 2)

        self.cc    = (4 + self.mueff / self.N) / (self.N + 4 + 2 * self.mueff / self.N)
        self.cs    = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1    = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu   = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs

        self.p_c   = np.zeros(self.N)
        self.p_sigma = np.zeros(self.N)
        self.C     = np.eye(self.N)
        self.chiN  = np.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N ** 2))

    def ask(self):
        self.C = (self.C + self.C.T) / 2
        try:
            B = np.linalg.cholesky(self.C)
            self.solutions = self.mu + self.sigma * (
                B @ np.random.randn(self.N, self.pop_size)
            ).T
        except np.linalg.LinAlgError:
            self.C = np.eye(self.N) * 0.1
            self.solutions = np.random.multivariate_normal(
                self.mu, (self.sigma ** 2) * self.C, self.pop_size
            )
        return self.solutions

    def tell(self, solutions, scores):
        idx      = np.argsort(scores)[::-1]
        best_idx = idx[: self.n_survivors]
        best_pop = solutions[best_idx]

        mu_old  = self.mu.copy()
        self.mu = np.dot(self.weights, best_pop)

        y = (self.mu - mu_old) / self.sigma

        self.p_sigma = (1 - self.cs) * self.p_sigma + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * y

        norm_ps = np.linalg.norm(self.p_sigma)
        h_sigma = 1 if norm_ps / self.chiN < 1.4 + 2 / (self.N + 1) else 0

        self.p_c = (1 - self.cc) * self.p_c + h_sigma * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y

        rank_one = np.outer(self.p_c, self.p_c)
        rank_mu  = np.zeros((self.N, self.N))
        for i in range(self.n_survivors):
            diff = (best_pop[i] - mu_old) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * rank_one
            + self.cmu * rank_mu
        )

        self.sigma = self.sigma * np.exp(
            (self.cs / self.damps) * (norm_ps / self.chiN - 1)
        )
        self.sigma = max(1e-5, min(self.sigma, 1e2))


class SakanaPSOptimizer:

    def __init__(self, base_arch, experts, val_loader, y_true, device, generations=20):
        self.base_arch   = base_arch
        self.experts     = experts
        self.val_loader  = val_loader
        self.y_true      = y_true
        self.device      = device
        self.generations = generations

        self.base_state = copy.deepcopy(experts[0].state_dict())
        for k in self.base_state:
            for i in range(1, len(experts)):
                self.base_state[k] = self.base_state[k] + experts[i].state_dict()[k]
            self.base_state[k] = self.base_state[k] / len(experts)

        self.task_vectors = []
        for exp in experts:
            tv = {}
            for k, v in exp.state_dict().items():
                if v.dtype == torch.float32:
                    tv[k] = v - self.base_state[k]
            self.task_vectors.append(tv)

        self.keys = list(self.task_vectors[0].keys())
        self.K    = len(experts)

        n_params  = 2 * self.K
        self.cma  = SimpleCMAES(n_params=n_params, sigma_init=1.0 / 6.0)

    def build_model(self, genome):
        genome = np.clip(genome, 0.0, 1.0)
        s_vals = genome[: self.K]          
        p_vals = genome[self.K :]         

        final_state = copy.deepcopy(self.base_state)

        for k in self.keys:
            sum_masked    = torch.zeros_like(self.base_state[k])
            masked_vectors = []

            for i in range(self.K):
                tau    = self.task_vectors[i][k]
                p_i    = float(p_vals[i])

                if "norm" in k or p_i >= 1.0:
                    tau_hat = tau
                else:
                    drop_mask = torch.bernoulli(torch.full_like(tau, p_i))
                    rescale   = 1.0 / (1.0 - p_i)
                    tau_hat   = rescale * (1.0 - drop_mask) * tau

                masked_vectors.append(tau_hat)
                sum_masked = sum_masked + tau_hat

            majority_sign  = torch.sign(sum_masked)
            merged_delta   = torch.zeros_like(self.base_state[k])
            for i in range(self.K):
                tau_hat   = masked_vectors[i]
                agreement = torch.sign(tau_hat) == majority_sign
                tau_tilde = torch.where(agreement, tau_hat, torch.zeros_like(tau_hat))
                merged_delta = merged_delta + s_vals[i] * tau_tilde

            final_state[k] = final_state[k] + merged_delta

        model = copy.deepcopy(self.base_arch)
        model.load_state_dict(final_state)
        return model

    def run(self):
        print(f"   [Sakana PS] Optimising parameter space ({self.generations} gens)...")
        best_score  = -np.inf
        best_genome = None

        for g in range(self.generations):
            pop    = self.cma.ask()
            scores = []
            for genome in pop:
                m = self.build_model(genome)
                _, probs = get_predictions(m, self.val_loader)
                score = compute_fitness_cross_entropy(probs, self.y_true)
                scores.append(score)
                if score > best_score:
                    best_score  = score
                    best_genome = genome.copy()

            self.cma.tell(pop, np.array(scores))
            if (g + 1) % 5 == 0:
                print(f"    Gen {g+1}: best score {best_score:.4f}")

        return self.build_model(best_genome)

class SakanaHybridModel(nn.Module):

    def __init__(self, models, weights, device):
        super().__init__()
        self.models  = nn.ModuleList(models)
        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32).to(device),
            requires_grad=False,
        )

    def forward(self, x):
        probs      = torch.stack([torch.softmax(m(x), dim=1) for m in self.models])
        w_norm     = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
        weighted   = (probs * w_norm).sum(dim=0)
        return torch.log(weighted + 1e-8)


class SakanaDFSOptimizer:
    def __init__(self, pool_models, val_loader, y_true, device, generations=30):
        self.pool        = pool_models
        self.val_loader  = val_loader
        self.y_true      = y_true
        self.device      = device
        self.generations = generations
        self.K           = len(pool_models)
        self.cma         = SimpleCMAES(n_params=self.K, pop_size=48)

    def run(self):
        print(f"   [Sakana DFS] Optimising ensemble weights ({self.generations} gens)...")

        precomputed_probs = []
        for m in self.pool:
            _, p = get_predictions(m, self.val_loader)
            precomputed_probs.append(p)
        precomputed_probs = np.array(precomputed_probs)  

        best_score   = -np.inf
        best_weights = None

        for g in range(self.generations):
            pop    = self.cma.ask()
            scores = []

            for w_genome in pop:
                w_exp   = np.exp(w_genome - np.max(w_genome))
                w_norm  = w_exp / np.sum(w_exp)             # (K,)
                ensemble_p = np.einsum('k,knc->nc', w_norm, precomputed_probs)
                score   = compute_fitness_cross_entropy(ensemble_p, self.y_true)
                scores.append(score)
                if score > best_score:
                    best_score   = score
                    best_weights = w_genome.copy()

            self.cma.tell(pop, np.array(scores))
            if (g + 1) % 5 == 0:
                print(f"    Gen {g+1}: best score {best_score:.4f}")

        return SakanaHybridModel(self.pool, best_weights, self.device)
