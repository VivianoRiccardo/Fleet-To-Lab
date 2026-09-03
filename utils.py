import random
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    f1_score
)
from scipy.stats import t
import pyezsparkc

from config import DEVICE, NUM_CLASSES, MASTER_SEED


# =============================
# SEEDING
# =============================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pyezsparkc.get_randomness_with_seed(MASTER_SEED)


# =============================
# METRICS
# =============================
def compute_metrics_all(y_true, y_pred):
    """Returns the full classification_report as a dictionary."""
    d = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    d['bal_acc'] = bal_acc
    return d

def compute_class_weights(y, num_classes):
    """Balanced class weights: N / (C * count_c)."""
    counts = np.bincount(y, minlength=num_classes)
    weights = len(y) / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# =============================
# FITNESS FUNCTIONS
# =============================
def compute_fitness_cross_entropy(child_probs, y_true):
    eps = 1e-8
    probs = np.clip(child_probs, eps, 1.0)
    counts = np.bincount(y_true, minlength=NUM_CLASSES)
    weights = len(y_true) / (NUM_CLASSES * np.maximum(counts, 1))
    sample_weights = weights[y_true]
    nll = -np.sum(
        sample_weights * np.log(probs[np.arange(len(y_true)), y_true])
    ) / np.sum(sample_weights)
    return 20.0 - nll


def compute_fitness_accuracy_entropy(child_probs, y_true, beta=0.02):
    acc = accuracy_score(y_true, np.argmax(child_probs, axis=1))
    entropy = -np.mean(np.sum(child_probs * np.log(child_probs + 1e-8), axis=1))
    SHIFT = beta * np.log(NUM_CLASSES) + 1e-6
    return acc - beta * entropy + SHIFT


def compute_fitness_combined(child_probs, y_true, penalty_weight):
    """Maximizes: Balanced Accuracy - (Penalty * Weighted Cross Entropy)."""
    y_pred = np.argmax(child_probs, axis=1)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    eps = 1e-12
    probs = np.clip(child_probs, eps, 1.0)
    counts = np.bincount(y_true, minlength=NUM_CLASSES)
    class_weights = len(y_true) / (NUM_CLASSES * np.maximum(counts, 1))
    sample_weights = class_weights[y_true]
    true_class_probs = probs[np.arange(len(y_true)), y_true]
    nll = -np.sum(sample_weights * np.log(true_class_probs)) / np.sum(sample_weights)

    return bal_acc - (penalty_weight * nll) + 20


# =============================
# MODEL <-> ARRAY CONVERSIONS
# =============================
def model_to_array(model):
    params_list = [param.detach().cpu().view(-1) for param in model.parameters()]
    return torch.cat(params_list).numpy()


def array_to_model(model, flat_array):
    flat_tensor = torch.tensor(flat_array, dtype=torch.float32).to(
        next(model.parameters()).device
    )
    idx = 0
    with torch.no_grad():
        for param in model.parameters():
            num_param = param.numel()
            chunk = flat_tensor[idx : idx + num_param]
            param.copy_(chunk.view(param.shape))
            idx += num_param
    return model


# =============================
# PREDICTIONS
# =============================
def get_predictions(model, loader):
    model.eval()
    preds = []
    probs = []
    with torch.no_grad():
        for Xb, _ in loader:
            out = model(Xb)
            prob = torch.softmax(out, dim=1)
            _, p = torch.max(prob, 1)
            preds.extend(p.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    return np.array(preds), np.array(probs)


# =============================
# EXPERT CLOUD GENERATION
# =============================
def generate_expert_cloud(w, n=6, sigma=0.06):
    cloud = []
    for _ in range(n):
        noise = np.random.normal(0, sigma, size=w.shape)
        cloud.append(w + noise)
    return cloud


# =============================
# PRINTING / DIAGNOSTICS
# =============================
def print_class_distribution(y, name="Dataset"):
    counts = count_classes(y)
    total = len(y)
    print(f"\n{name} class distribution:")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt} ({cnt/total:.2%})")


def count_classes(y, num_classes=None):
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    counts = np.bincount(y, minlength=num_classes if num_classes else 0)
    return {i: int(counts[i]) for i in range(len(counts))}


# =============================
# STATISTICAL TESTS
# =============================
def corrected_resampled_t_test(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected t-statistic for MCCV."""
    n = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    correction = (1 / n) + (n_train / n_test)
    stderr = np.sqrt(correction * var_diff)
    t_stat = mean_diff / (stderr + 1e-8)
    df = n - 1
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    return t_stat, p_value


# =============================
# TEST SET EVALUATION (for per-generation tracking)
# =============================
def compute_test_metrics(model, test_loader, y_test):
    """
    Evaluate a model on the test set and return balanced accuracy,
    macro F1, and weighted cross-entropy loss.
    """
    preds, probs = get_predictions(model, test_loader)

    bal_acc = balanced_accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    # Weighted cross-entropy loss (same weighting as training)
    eps = 1e-8
    probs_clipped = np.clip(probs, eps, 1.0)
    counts = np.bincount(y_test, minlength=NUM_CLASSES)
    class_weights = len(y_test) / (NUM_CLASSES * np.maximum(counts, 1))
    sample_weights = class_weights[y_test]
    true_class_probs = probs_clipped[np.arange(len(y_test)), y_test]
    all_probs = np.clip(true_class_probs,1e-15,1.0)
    loss = -np.mean(np.log(all_probs))

    return bal_acc, macro_f1, loss
