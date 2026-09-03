import copy
import numpy as np

from config import (
    SWARM_SIZE, SWARM_ITERATIONS, PATIENCE,
)
from utils import model_to_array, array_to_model, get_predictions, compute_fitness_cross_entropy


class IWOAOptimizer:
    """
    Improved Whale Optimization Algorithm (IWOA) over model weight space.

    This is the IWOA-ALONE ablation of AcoMerge: it runs Phase 2's whale
    dynamics as a standalone global search over full flattened weight vectors,
    WITHOUT the MMAS pheromone phase. Fitness is therefore evaluated directly
    on the calibration set (weighted cross-entropy), exactly as for the other
    weight-space baselines, rather than being read from pheromone.

    Interface mirrors ModelSwarmOptimizer so it is a drop-in in main.py:
        opt = IWOAOptimizer(base_model, experts, val_loader, y_true, device)
        opt.initialize_swarm()
        model = opt.run()

    IWOA update rules (matching the paper's Phase 2):
      * shrinking encircling  (p < 0.5, |A| < 1) : exploit around best
      * spiral update         (p >= 0.5)         : logarithmic spiral to best
      * exploration           (p < 0.5, |A| >= 1): move toward a random whale
      * adaptive inertia      w_inertia = 1 - 2*(t/T)^3
      * a decreases linearly from 2 to 0 over iterations
    """

    B_SPIRAL = 1.0   # logarithmic-spiral shape constant b

    def __init__(self, base_model, initial_experts, validation_loader,
                 y_true_val, device):
        self.base_model      = base_model
        self.initial_experts = [model_to_array(e) for e in initial_experts]
        self.val_loader      = validation_loader
        self.y_true          = y_true_val
        self.device          = device

        self.whales_x = []          # population of full weight vectors
        self.best_x   = None        # global-best weight vector
        self.best_score = -np.inf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(self, weights):
        """Scalar utility for a weight vector (weighted cross-entropy fitness)."""
        temp_model = array_to_model(copy.deepcopy(self.base_model), weights)
        _, probs   = get_predictions(temp_model, self.val_loader)
        return compute_fitness_cross_entropy(probs, self.y_true)

    def _update_best(self, weights, score):
        if score > self.best_score:
            self.best_score = score
            self.best_x     = weights.copy()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_swarm(self):
        """
        Populate SWARM_SIZE whales: the experts as-is, remaining slots filled
        by pairwise linear interpolation (same seeding as ModelSwarm, so the
        two share an identical starting population for a fair IWOA-vs-PSO
        comparison).
        """
        print(f"   [IWOA] Initialising {SWARM_SIZE} whales...")
        self.whales_x = list(self.initial_experts)

        while len(self.whales_x) < SWARM_SIZE:
            a, b = np.random.choice(len(self.initial_experts), 2, replace=False)
            t = np.random.uniform(0.0, 1.0)
            self.whales_x.append(t * self.initial_experts[a] +
                                 (1.0 - t) * self.initial_experts[b])

        self.whales_x = [np.array(w) for w in self.whales_x]

        for w in self.whales_x:
            self._update_best(w, self._evaluate(w))

        print(f"   [IWOA] Init global best: {self.best_score:.4f}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Run IWOA and return the best-found model.

        Stopping: SWARM_ITERATIONS iterations, or PATIENCE consecutive
        iterations without global-best improvement.
        """
        T = SWARM_ITERATIONS
        patience = 0

        for t in range(T):
            if patience >= PATIENCE:
                print(f"   [IWOA] Patience ({PATIENCE}) reached at iter {t}. Stopping.")
                break

            # a decreases linearly 2 -> 0; adaptive inertia w = 1 - 2(t/T)^3
            a = 2.0 * (1.0 - t / T)
            w_inertia = 1.0 - 2.0 * (t / T) ** 3

            improved = False

            for i in range(SWARM_SIZE):
                xi = self.whales_x[i]
                p  = np.random.rand()

                if p < 0.5:
                    A = a * (2.0 * np.random.rand() - 1.0)
                    C = 2.0 * np.random.rand()
                    if abs(A) < 1.0:
                        # Shrinking encircling (exploitation around best)
                        D = np.abs(C * self.best_x - xi)
                        new_xi = self.best_x - w_inertia * A * D
                    else:
                        # Exploration: move toward a random whale
                        rand_idx = np.random.randint(0, SWARM_SIZE)
                        x_rand = self.whales_x[rand_idx]
                        D = np.abs(C * x_rand - xi)
                        new_xi = x_rand - w_inertia * A * D
                else:
                    # Spiral update toward best
                    Dp = np.abs(self.best_x - xi)
                    l  = np.random.uniform(-1.0, 1.0)
                    new_xi = (w_inertia * Dp * np.exp(self.B_SPIRAL * l) *
                              np.cos(2.0 * np.pi * l) + self.best_x)

                self.whales_x[i] = new_xi
                score = self._evaluate(new_xi)
                before = self.best_score
                self._update_best(new_xi, score)
                if self.best_score > before:
                    improved = True

            patience = 0 if improved else patience + 1

            if (t + 1) % 10 == 0:
                print(f"   [IWOA] Iter {t+1}/{T} | Best: {self.best_score:.4f} | "
                      f"patience: {patience}/{PATIENCE}")

        print(f"   [IWOA] Finished. Global best: {self.best_score:.4f}")
        return array_to_model(copy.deepcopy(self.base_model), self.best_x)
