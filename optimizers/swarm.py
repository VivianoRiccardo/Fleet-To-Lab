import copy
import numpy as np

from config import (
    SWARM_SIZE, SWARM_ITERATIONS, STEP_LAMBDA, LAMBDA_DECAY,
    INERTIA, COGNITIVE, SOCIAL, REPEL, PATIENCE, RESTART_PATIENCE,
)
from utils import model_to_array, array_to_model, get_predictions, compute_fitness_cross_entropy


class ModelSwarmOptimizer:

    def __init__(self, base_model, initial_experts, validation_loader, y_true_val, device):
        self.base_model   = base_model
        self.initial_experts = [model_to_array(e) for e in initial_experts]
        self.val_loader   = validation_loader
        self.y_true       = y_true_val
        self.device       = device

        self.particles_x           = []
        self.particles_v           = []
        self.personal_best_p       = []
        self.personal_best_scores  = []
        self.stagnation_counters   = []   

        self.global_best_g         = None
        self.global_best_score     = -np.inf
        self.global_worst_gw       = None
        self.global_worst_score    = np.inf

    def _evaluate(self, weights):
        temp_model = array_to_model(copy.deepcopy(self.base_model), weights)
        _, probs   = get_predictions(temp_model, self.val_loader)
        return compute_fitness_cross_entropy(probs, self.y_true)

    def _update_global_trackers(self, weights, score):
        improved = False
        if score > self.global_best_score:
            self.global_best_score = score
            self.global_best_g     = weights.copy()
            improved = True
        if score < self.global_worst_score:
            self.global_worst_score = score
            self.global_worst_gw    = weights.copy()
        return improved
    def initialize_swarm(self):
        print(f"   [Model Swarms] Initialising {SWARM_SIZE} particles...")

        self.particles_x = list(self.initial_experts)

        while len(self.particles_x) < SWARM_SIZE:
            idx_a, idx_b = np.random.choice(len(self.initial_experts), 2, replace=False)
            t = np.random.uniform(0.0, 1.0)
            crossover = t * self.initial_experts[idx_a] + (1.0 - t) * self.initial_experts[idx_b]
            self.particles_x.append(crossover)

        for i in range(SWARM_SIZE):
            xi    = np.array(self.particles_x[i])
            score = self._evaluate(xi)

            rand_idx = np.random.randint(0, SWARM_SIZE)
            vi       = np.array(self.particles_x[rand_idx]) - xi

            self.particles_x[i]          = xi
            self.particles_v.append(vi)
            self.personal_best_p.append(xi.copy())
            self.personal_best_scores.append(score)
            self.stagnation_counters.append(0)

            self._update_global_trackers(xi, score)

        print(f"   [Model Swarms] Init global best: {self.global_best_score:.4f}")

    def run(self):
        current_lambda          = STEP_LAMBDA
        global_patience_counter = 0

        for k in range(SWARM_ITERATIONS):

            if global_patience_counter >= PATIENCE:
                print(
                    f"   [Model Swarms] Global patience ({PATIENCE} iters) reached "
                    f"at iteration {k}. Stopping."
                )
                break

            global_improved_this_iter = False 

            for i in range(SWARM_SIZE):
                xi = self.particles_x[i]
                vi = self.particles_v[i]
                pi = self.personal_best_p[i]

                rv, rp, rg, rw = np.random.rand(4)

                term_inertia   = rv * INERTIA   * vi
                term_cognitive = rp * COGNITIVE * (pi - xi)
                term_social    = rg * SOCIAL    * (self.global_best_g  - xi)
                term_repel     = rw * REPEL     * (self.global_worst_gw - xi)

                C      = rv * INERTIA + rp * COGNITIVE + rg * SOCIAL + rw * REPEL
                C      = max(C, 1e-8)   
                new_vi = (term_inertia + term_cognitive + term_social - term_repel) / C

                new_xi = xi + current_lambda * new_vi

                self.particles_v[i] = new_vi
                self.particles_x[i] = new_xi

                score = self._evaluate(new_xi)

                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_p[i]       = new_xi.copy()
                    self.stagnation_counters[i]   = 0
                else:
                    self.stagnation_counters[i] += 1

                if self._update_global_trackers(new_xi, score):
                    global_improved_this_iter = True

                if self.stagnation_counters[i] >= RESTART_PATIENCE:
                    self.particles_x[i]          = self.personal_best_p[i].copy()
                    self.particles_v[i]           = np.zeros_like(vi)
                    self.stagnation_counters[i]  = 0

            if global_improved_this_iter:
                global_patience_counter = 0
            else:
                global_patience_counter += 1

            current_lambda *= LAMBDA_DECAY

            if (k + 1) % 10 == 0:
                print(
                    f"   [Model Swarms] Iter {k+1}/{SWARM_ITERATIONS} | "
                    f"Best: {self.global_best_score:.4f} | "
                    f"Global patience: {global_patience_counter}/{PATIENCE}"
                )

        print(f"   [Model Swarms] Finished. Global best: {self.global_best_score:.4f}")
        return array_to_model(copy.deepcopy(self.base_model), self.global_best_g)
