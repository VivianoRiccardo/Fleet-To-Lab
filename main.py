import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from config import (
    FILES_GROUP_1, FILES_GROUP_2, FILES_GROUP_3, FILES_GROUP_4,
    BATCH_SIZE, BATCH_SIZE_ADAPT, LR_BASE, LR_ADAPT, LR_ADAPT_ONLY,
    EPOCHS_BASE, EPOCHS_ADAPT,
    ADAPT_PERCENT_FIRST_CLASS, ADAPT_PERCENT_SECOND_CLASS, ADAPT_PERCENT_THIRD_CLASS,
    EXPERT_CLOUD, RANDOM_CLOUD, EXPERT_MULTIPLIER,
    INPUT_DIM_EXPERT, INPUT_DIM_CONDITIONAL, NUM_CLASSES, DEVICE,
    MASTER_SEED, N_REPEATS, GA_GENERATIONS
)
from utils import (
    set_global_seed, compute_metrics_all, compute_class_weights,
    model_to_array, array_to_model, get_predictions,
    generate_expert_cloud, print_class_distribution,
    corrected_resampled_t_test
)
from datasets import RoverDataset, load_raw_12_features, add_context_flags
from networks import AdaptiveSlippageNet, SlippageNet
from optimizers.swarm import ModelSwarmOptimizer
from optimizers.dare_ties import DareTiesOptimizer
from optimizers.sakana import SakanaPSOptimizer, SakanaDFSOptimizer
from optimizers.aco import run_aco_merging
from optimizers.adamerging import AdaMergingOptimizer
from optimizers.distillation import DistillationOptimizer
from optimizers.iwoa import IWOAOptimizer

from git_rebasin import align_experts_to_reference


def train_model(model, loader, criterion, optimizer, epochs):
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()


def train_film_adaptation(model, loader, class_weights, optimizer, epochs):
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            probs = torch.softmax(model(Xb), dim=1)
            log_probs = torch.log(probs + 1e-8)
            sample_weights = class_weights[yb].to(DEVICE)
            loss = -torch.mean(
                sample_weights *
                torch.sum(
                    torch.nn.functional.one_hot(yb, NUM_CLASSES).float() * log_probs,
                    dim=1
                )
            )
            loss.backward()
            optimizer.step()

def train_film_adaptation_tracked(model, loader, class_weights, optimizer, epochs,
                                   test_loader=None, y_test=None, label=""):
    history = []

    for epoch in range(epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            probs = torch.softmax(model(Xb), dim=1)
            log_probs = torch.log(probs + 1e-8)
            sample_weights = class_weights[yb].to(DEVICE)
            loss = -torch.mean(
                sample_weights *
                torch.sum(
                    torch.nn.functional.one_hot(yb, NUM_CLASSES).float() * log_probs,
                    dim=1
                )
            )
            loss.backward()
            optimizer.step()

        if test_loader is not None and y_test is not None:
            model.eval()
            with torch.no_grad():
                preds, _ = get_predictions(model, test_loader)
            report = compute_metrics_all(y_test, preds)
            history.append({
                'epoch': epoch + 1,
                'bal_acc': report['bal_acc'],
                'macro_f1': report['macro avg']['f1-score']
            })

    if label and history:
        print(f"\n[{label}] Per-epoch test metrics:")
        print(f"  {'Epoch':<8} | {'Bal Acc':<10} | {'Macro F1':<10}")
        print(f"  {'-'*35}")
        for row in history:
            print(f"  {row['epoch']:<8} | {row['bal_acc']:.4f}     | {row['macro_f1']:.4f}")

    return history

def _build_cloud_pool(centroid_arrays, expert_cloud, random_cloud, expert_multiplier, sigma=0.005):

    expanded = []

    smaller = []

    for w in centroid_arrays:
        expanded.append(w)
        smaller.append(w)
        smaller.extend(generate_expert_cloud(w, n=expert_multiplier, sigma=sigma))
        expanded.extend(generate_expert_cloud(w, n=expert_cloud, sigma=sigma))
        for _ in range(random_cloud):
            expanded.append(model_to_array(SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE)))

    full_pool = [array_to_model(SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE), e) for e in expanded]
    small_pool = [array_to_model(SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE), e) for e in smaller]

    return full_pool, small_pool


def build_expert_pools(raw_experts, expert_cloud, random_cloud, expert_multiplier):
    expert_arrays = [model_to_array(e) for e in raw_experts]
    return _build_cloud_pool(expert_arrays, expert_cloud, random_cloud, expert_multiplier)


def build_non_expert_pools(n_centroids, expert_cloud, random_cloud, expert_multiplier):

    random_centroid_arrays = [
        model_to_array(SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE))
        for _ in range(n_centroids)
    ]
    return _build_cloud_pool(random_centroid_arrays, expert_cloud, random_cloud, expert_multiplier)


def create_adapt_test_split(X_raw, y, adapt_ratios, seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    indices = np.arange(len(y))
    adapt_indices_list = []

    for cls_label, ratio in adapt_ratios.items():
        cls_idxs = indices[y == cls_label]
        n_samples = max(1, int(len(cls_idxs) * ratio)) if len(cls_idxs) > 0 else 0
        selected = np.random.choice(cls_idxs, n_samples, replace=False)
        adapt_indices_list.extend(selected)

    adapt_indices = np.array(adapt_indices_list)
    mask = np.ones(len(y), dtype=bool)
    mask[adapt_indices] = False
    test_indices = indices[mask]

    X_adapt, y_adapt = X_raw[adapt_indices], y[adapt_indices]
    X_test, y_test = X_raw[test_indices], y[test_indices]

    perm = np.random.permutation(len(y_adapt))
    return X_adapt[perm], y_adapt[perm], X_test, y_test

def main():
    set_global_seed(MASTER_SEED)
    rng = np.random.default_rng(MASTER_SEED)
    seeds = rng.integers(low=0, high=2**31 - 1, size=N_REPEATS).tolist()
    torch.use_deterministic_algorithms(True)


    print("=== STEP 1: LOAD RAW DATA (12 Features) ===")
    X1_raw, y1 = load_raw_12_features(FILES_GROUP_1)
    X2_raw, y2 = load_raw_12_features(FILES_GROUP_2)
    X3_raw, y3 = load_raw_12_features(FILES_GROUP_3)
    X4_raw, y4 = load_raw_12_features(FILES_GROUP_4)

    print_class_distribution(y1, "Earth Set")
    print_class_distribution(y2, "Moon Set1")
    print_class_distribution(y3, "Moon Set2")
    print_class_distribution(y4, "Novel Set")


    X1_cond = add_context_flags(X1_raw, 'earth', 1)
    X2_cond = add_context_flags(X2_raw, 'moon', 2)
    X3_cond = add_context_flags(X3_raw, 'moon', 3)


    print("\n=== TRAINING CONDITIONAL NETWORK (17 Inputs) ===")
    X_train_cond = np.vstack((X1_cond, X2_cond, X3_cond))
    y_train_cond = np.concatenate((y1, y2, y3))

    cond_loader = DataLoader(
        RoverDataset(X_train_cond, y_train_cond),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seeds[0])
    )

    model_cond = AdaptiveSlippageNet(INPUT_DIM_CONDITIONAL, NUM_CLASSES).to(DEVICE)
    model_cond.set_mode("base")

    crit = nn.CrossEntropyLoss(weight=compute_class_weights(y_train_cond, NUM_CLASSES))
    opt = optim.Adam(filter(lambda p: p.requires_grad, model_cond.parameters()), lr=LR_BASE)
    train_model(model_cond, cond_loader, crit, opt, EPOCHS_BASE)

    print("\n=== TRAINING EXPERTS (12 Inputs) ===")
    raw_experts = []
    expert_datasets = [(X1_raw, y1), (X2_raw, y2), (X3_raw, y3)]
    model_tc = SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE)
    shared_init = copy.deepcopy(model_tc) # for adamerge
    
    for Xd, yd in expert_datasets:
        model = copy.deepcopy(model_tc)
        loader = DataLoader(
            RoverDataset(Xd, yd), batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(seeds[1])
        )
        crit = nn.CrossEntropyLoss(weight=compute_class_weights(yd, NUM_CLASSES))
        opt = optim.Adam(model.parameters(), lr=LR_BASE)
        train_model(model, loader, crit, opt, EPOCHS_BASE)
        if len(raw_experts)== 0:
            raw_experts.append(model)
        else:
            raw_experts.append(raw_experts[0])
    experts_shared_base = [copy.deepcopy(e) for e in raw_experts]# for adamerge
    

    print("\n=== ALIGNING EXPERTS VIA GIT RE-BASIN ===")
    raw_experts = [raw_experts[0]] + align_experts_to_reference(
        reference_model=raw_experts[0],
        expert_models=raw_experts[1:],  
        max_iters=100
    )


    print("\n=== TRAINING EARTH EXPERT ===")
    model_x1 = AdaptiveSlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE)
    model_x1.set_mode("base")

    loader_x1 = DataLoader(
        RoverDataset(X1_raw, y1), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seeds[2])
    )
    crit = nn.CrossEntropyLoss(weight=compute_class_weights(y1, NUM_CLASSES))
    opt = optim.Adam(filter(lambda p: p.requires_grad, model_x1.parameters()), lr=LR_BASE)
    train_model(model_x1, loader_x1, crit, opt, EPOCHS_BASE)

    full_pool, small_pool = build_expert_pools(raw_experts, EXPERT_CLOUD, RANDOM_CLOUD, EXPERT_MULTIPLIER)
    non_expert_full_pool, non_expert_small_pool = build_non_expert_pools(
        n_centroids=len(raw_experts),  
        expert_cloud=EXPERT_CLOUD,
        random_cloud=RANDOM_CLOUD,
        expert_multiplier=EXPERT_MULTIPLIER
    )

    model_names = [
        "Cond_Net", "Swarm_Exp", "Swarm_NonExp", "ACO_Exp", "ACO_NonExp",
        "Earth_Adapt", "Dare_Ties_Exp", "Dare_Ties_NonExp",
        "Dare_Ties_Sakana_Exp", "Dare_Ties_Sakana_NonExp", "model_only_adapt_set", "AdaMerge", "Distill"
    ]

    results_per_class = {
        name: {c: {'prec': [], 'rec': [], 'f1': []} for c in range(NUM_CLASSES)}
        for name in model_names
    }
    results_global = {name: {'acc': [], 'macro_f1': []} for name in model_names}

    aco_exp_gen_metrics = []      
    aco_nonexp_gen_metrics = []

    adapt_ratios = {
        0: ADAPT_PERCENT_FIRST_CLASS,
        1: ADAPT_PERCENT_SECOND_CLASS,
        2: ADAPT_PERCENT_THIRD_CLASS
    }
    cond_histories = []
    earth_histories = []
    nn_repeats = 27
    sum_repeats = 3
    for i in range(N_REPEATS):
        print(str(i)+"/"+str(nn_repeats+sum_repeats))
        seed = seeds[i]

        X4_adapt_raw, y4_adapt, X4_test_raw, y4_test = create_adapt_test_split(
            X4_raw, y4, adapt_ratios, seed
        )
        print_class_distribution(y4_adapt, "Adapt Set")
        print_class_distribution(y4_test, "Test Set")

        adapt_class_weights = compute_class_weights(y4_adapt, NUM_CLASSES)

        X4_adapt_cond = add_context_flags(X4_adapt_raw, 'moon', 1)
        X4_test_cond = add_context_flags(X4_test_raw, 'moon', 1)

        print("\n=== TRAINING ONLY MOON BASELINE (12 Inputs) ===")
        model_only_adapt_set = SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE)
        ss = model_to_array(model_only_adapt_set)
        
        loader_adapt_only = DataLoader(
            RoverDataset(X4_adapt_raw, y4_adapt), batch_size=BATCH_SIZE_ADAPT, shuffle=True,
            generator=torch.Generator().manual_seed(seeds[1])
        )
        crit = nn.CrossEntropyLoss(weight=compute_class_weights(y4_adapt, NUM_CLASSES))
        opt = optim.Adam(model_only_adapt_set.parameters(), lr=LR_ADAPT_ONLY)
        train_model(model_only_adapt_set, loader_adapt_only, crit, opt, EPOCHS_ADAPT)
        
        '''
        # --- Conditional Net FiLM Adaptation ---
        
        print("Adapting Conditional Network via FiLM...")
        
        adapt_loader_cond = DataLoader(
            RoverDataset(X4_adapt_cond, y4_adapt), batch_size=BATCH_SIZE_ADAPT, shuffle=True,
            generator=torch.Generator().manual_seed(seeds[3])
        )
        
        test_loader_eval = DataLoader(
            RoverDataset(X4_test_cond, y4_test), batch_size=BATCH_SIZE_ADAPT, shuffle=False
        )
        model_cond_copy = copy.deepcopy(model_cond)
        model_cond_copy.set_mode("adapt")
        opt_adapt = optim.Adam(filter(lambda p: p.requires_grad, model_cond_copy.parameters()), lr=LR_ADAPT)
        cond_history = train_film_adaptation_tracked(
            model_cond_copy, adapt_loader_cond, adapt_class_weights, opt_adapt, EPOCHS_ADAPT,
            test_loader=test_loader_eval, y_test=y4_test, label=f"Cond_Net Run {i}"
        )
        '''
        
        print("Adapting Conditional Network...")
        model_cond_copy = copy.deepcopy(model_cond)
        adapt_loader_cond = DataLoader(
            RoverDataset(X4_adapt_cond, y4_adapt), batch_size=BATCH_SIZE_ADAPT, shuffle=True,
            generator=torch.Generator().manual_seed(seeds[3])
        )
        model_cond_copy.set_mode("base")
        opt_adapt = optim.Adam(filter(lambda p: p.requires_grad, model_cond_copy.parameters()), lr=LR_ADAPT)
        train_film_adaptation(model_cond_copy, adapt_loader_cond, adapt_class_weights, opt_adapt, EPOCHS_ADAPT)
        
        val_loader_merge = DataLoader(
            RoverDataset(X4_adapt_raw, y4_adapt), batch_size=BATCH_SIZE_ADAPT, shuffle=False
        )

        test_loader_aco = DataLoader(
            RoverDataset(X4_test_raw, y4_test), batch_size=64, shuffle=False
        )

        experts_swarm = [copy.deepcopy(e) for e in full_pool]
        experts_aco = [copy.deepcopy(e) for e in full_pool]
        experts_dare_ties = [copy.deepcopy(e) for e in small_pool]
        experts_dare_ties_sakana = [copy.deepcopy(e) for e in small_pool]

        not_experts_swarm = [copy.deepcopy(e) for e in non_expert_full_pool]
        not_experts_aco = [copy.deepcopy(e) for e in non_expert_full_pool]
        not_experts_dare_ties = [copy.deepcopy(e) for e in non_expert_small_pool]
        not_experts_dare_ties_sakana = [copy.deepcopy(e) for e in non_expert_small_pool]
        

        print("\n--- 1. Model Swarms (EXPERTS) ---")
        swarm_opt = ModelSwarmOptimizer(
            SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            experts_swarm, val_loader_merge, y4_adapt, DEVICE
        )
        swarm_opt.initialize_swarm()
        final_swarm_expert = swarm_opt.run()

        
        
        
        print("\n--- 2. ACO (EXPERTS) ---")
        final_aco_expert, gen_metrics_exp = run_aco_merging(
            experts_aco, val_loader_merge, y4_adapt,
            SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            test_loader=test_loader_aco, y_test=y4_test
        )
        aco_exp_gen_metrics.append(gen_metrics_exp)
        

        print("\n--- AdaMerging (EXPERTS) ---")
        final_adamerge = AdaMergingOptimizer(
            base_arch=SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            shared_init=shared_init,
            experts=experts_shared_base,
            val_loader=val_loader_merge,
            y_true=y4_adapt,
            device=DEVICE,
            lr=LR_ADAPT,
            epochs=EPOCHS_ADAPT,
            init_coeff=0.3,
        ).optimize()

        print("\n--- Knowledge Distillation (EXPERTS) ---")
        final_distill = DistillationOptimizer(
            base_arch=SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            teachers=experts_shared_base,     # 3 domain-expert teachers
            val_loader=val_loader_merge,
            y_true=y4_adapt,
            device=DEVICE,
            lr=LR_ADAPT,
            epochs=EPOCHS_ADAPT,
            temperature=4.0,
            alpha=0.5,
        ).optimize()


        print("\n--- 3. Differentiable DARE-TIES (EXPERTS) ---")
        final_daries_model_experts = DareTiesOptimizer(
            base_arch=SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            experts=experts_dare_ties, val_loader=val_loader_merge,
            y_true=y4_adapt, device=DEVICE, p_drop=0.5,
            lr=LR_ADAPT, epochs=EPOCHS_ADAPT
        ).optimize()

        print("\n--- 4. Sakana Evolutionary Merging (EXPERTS) ---")
        sakana_ps_model = SakanaPSOptimizer(
            base_arch=SlippageNet(INPUT_DIM_EXPERT, NUM_CLASSES).to(DEVICE),
            experts=experts_dare_ties_sakana, val_loader=val_loader_merge,
            y_true=y4_adapt, device=DEVICE, generations=EPOCHS_ADAPT
        ).run()

        dfs_pool = [sakana_ps_model] + experts_dare_ties_sakana
        expert_sakana_dfs_model = SakanaDFSOptimizer(
            pool_models=dfs_pool, val_loader=val_loader_merge,
            y_true=y4_adapt, device=DEVICE, generations=30
        ).run()
        
        
        print("Adapting Earth Network via FiLM...")
        model_x1_copy = copy.deepcopy(model_x1)
        adapt_loader_x1 = DataLoader(
            RoverDataset(X4_adapt_raw, y4_adapt), batch_size=16, shuffle=True,
            generator=torch.Generator().manual_seed(seeds[4])
        )
        model_x1_copy.set_mode("base")
        opt_adapt = optim.Adam(filter(lambda p: p.requires_grad, model_x1_copy.parameters()), lr=LR_ADAPT)
        train_film_adaptation(model_x1_copy, adapt_loader_x1, adapt_class_weights, opt_adapt, EPOCHS_ADAPT)
    
        
        

        print("\n========================================================")
        print("                  RUN RESULTS COMPARISON")
        print("========================================================")
        def eval_res(name_key, model, X_test, y_test):
            loader = DataLoader(RoverDataset(X_test, y_test), batch_size=64, shuffle=False)
            preds, _ = get_predictions(model, loader)
            report = compute_metrics_all(y_test, preds)

            acc = report['bal_acc']
            macro_f1 = report['macro avg']['f1-score']
            results_global[name_key]['acc'].append(acc)
            results_global[name_key]['macro_f1'].append(macro_f1)

            for c in range(NUM_CLASSES):
                c_str = str(c)
                if c_str in report:
                    results_per_class[name_key][c]['prec'].append(report[c_str]['precision'])
                    results_per_class[name_key][c]['rec'].append(report[c_str]['recall'])
                    results_per_class[name_key][c]['f1'].append(report[c_str]['f1-score'])

            print(f"{name_key:25s} | Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")
            
            
        def plot_adaptation_curve(histories, label, metric='bal_acc'):
            n_epochs = len(histories[0])
            vals = np.array([[h[metric] for h in run] for run in histories])  # (N_REPEATS, epochs)
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            epochs = np.arange(1, n_epochs + 1)

            plt.figure()
            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.3)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title(f"FiLM Adaptation: {metric} on Test Set ({label})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"film_overfit_{label}_{metric}.png", dpi=150)
            plt.close()
            print(f"[Saved] film_overfit_{label}_{metric}.png")
        
        
        eval_res("Cond_Net", model_cond_copy, X4_test_cond, y4_test)
        eval_res("Swarm_Exp", final_swarm_expert, X4_test_raw, y4_test)
        eval_res("ACO_Exp", final_aco_expert, X4_test_raw, y4_test)
        eval_res("Earth_Adapt", model_x1_copy, X4_test_raw, y4_test)
        eval_res("Dare_Ties_Exp", final_daries_model_experts, X4_test_raw, y4_test)
        eval_res("Dare_Ties_Sakana_Exp", expert_sakana_dfs_model, X4_test_raw, y4_test)
        eval_res("model_only_adapt_set", model_only_adapt_set, X4_test_raw, y4_test)
        eval_res("AdaMerge", final_adamerge, X4_test_raw, y4_test)
        eval_res("Distill", final_distill, X4_test_raw, y4_test)
        

        save_dir = "saved_outputs/200_samples"
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        torch.save(final_aco_expert.state_dict(), f"{save_dir}/aco_expert_run_{i}.pth")

        # Save test set
        np.save(f"{save_dir}/X4_test_raw_run_{i}.npy", X4_test_raw)
        np.save(f"{save_dir}/y4_test_run_{i}.npy", y4_test)

        # Save train/adapt set (needed for KNN distances)
        np.save(f"{save_dir}/X4_adapt_raw_run_{i}.npy", X4_adapt_raw)
        np.save(f"{save_dir}/y4_adapt_run_{i}.npy", y4_adapt)

        print(f"[Saved] Model and dataset for run {i} -> {save_dir}/")
        
        print("========================================================")


    save_aco_generation_metrics(aco_exp_gen_metrics, "aco_expert_gen_metrics_experts_"+str(EXPERT_CLOUD)+"_random_"+str(RANDOM_CLOUD)+".csv")

    print_final_report(model_names, results_per_class, results_global, y4_adapt, y4_test)


def save_aco_generation_metrics(all_repeats_metrics, filename):

    n_repeats = len(all_repeats_metrics)
    n_gens = GA_GENERATIONS

    rows = []
    for gen_idx in range(n_gens):
        bal_accs = [all_repeats_metrics[r][gen_idx]['bal_acc'] for r in range(n_repeats)]
        f1_macros = [all_repeats_metrics[r][gen_idx]['f1_macro'] for r in range(n_repeats)]

        rows.append({
            'generation': gen_idx + 1,
            'bal_acc_mean': np.mean(bal_accs),
            'bal_acc_std': np.std(bal_accs),
            'f1_macro_mean': np.mean(f1_macros),
            'f1_macro_std': np.std(f1_macros),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\n[Saved] ACO per-generation test metrics -> {filename}")
    print(df.to_string(index=False))


def print_final_report(model_names, results_per_class, results_global, y4_adapt, y4_test):

    print("\n\n========================================================")
    print(f"       FINAL PER-CLASS RESULTS ({N_REPEATS} Runs)")
    print("========================================================")

    metrics = ['prec', 'rec', 'f1']
    metric_names = ['Precision', 'Recall', 'F1-Score']

    for i, m_key in enumerate(metrics):
        print(f"\n>>> METRIC: {metric_names[i]} (Mean +/- Std Dev) <<<")
        header = f"{'Method':<20} | {'Class 0':<20} | {'Class 1':<20} | {'Class 2':<20}"
        print(header)
        print("-" * len(header))

        for name in model_names:
            row_str = f"{name:<20} | "
            for c in range(NUM_CLASSES):
                values = results_per_class[name][c][m_key]
                if values:
                    row_str += f"{np.mean(values):.4f} +/- {np.std(values):.4f}   | "
                else:
                    row_str += f"{'N/A':<20} | "
            print(row_str)

    print("\n\n>>> BALANCED ACCURACY (Mean +/- Std) <<<")
    for name in model_names:
        vals = results_global[name]['acc']
        print(f"{name:<20}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    print("\n\n>>> F1 MACRO (Mean +/- Std) <<<")
    for name in model_names:
        vals = results_global[name]['macro_f1']
        print(f"{name:<20}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Statistical Tests
    print("\n========================================================")
    print("   STATISTICAL SIGNIFICANCE (Nadeau & Bengio Corrected)")
    print("========================================================")

    n_train_s = len(y4_adapt)
    n_test_s = len(y4_test)

    comparisons = [
        ("ACO_Exp", "Swarm_Exp"),
        ("ACO_Exp", "Cond_Net"),
        ("ACO_Exp", "Earth_Adapt"),
        ("Swarm_Exp", "Cond_Net"),
        ("Cond_Net", "Earth_Adapt"),
        ("Dare_Ties_Exp", "ACO_Exp"),
        ("Dare_Ties_Exp", "Swarm_Exp"),
        ("Dare_Ties_Exp", "Cond_Net"),
        ("Dare_Ties_Exp", "Dare_Ties_NonExp"),
        ("Dare_Ties_Sakana_Exp", "Swarm_Exp"),
        ("Dare_Ties_Sakana_Exp", "Cond_Net"),
        ("Dare_Ties_Sakana_Exp", "ACO_Exp"),
        ("Dare_Ties_Sakana_Exp", "Dare_Ties_Sakana_NonExp"),
        ("Dare_Ties_Sakana_NonExp", "Swarm_Exp"),
        ("Dare_Ties_Sakana_NonExp", "Cond_Net"),
        ("Dare_Ties_Sakana_NonExp", "ACO_Exp"),
        ("Dare_Ties_Sakana_NonExp", "Dare_Ties_Sakana_NonExp"),
        ("model_only_adapt_set", "Dare_Ties_Exp"),
        ("model_only_adapt_set", "Dare_Ties_Sakana_NonExp"),
        ("model_only_adapt_set", "Swarm_Exp"),
        ("model_only_adapt_set", "Cond_Net"),
        ("model_only_adapt_set", "ACO_Exp"),
        ("model_only_adapt_set", "Dare_Ties_Sakana_NonExp"),
    ]

    metrics_to_test = [
        ("GLOBAL ACCURACY", "acc"),
        ("MACRO F1", "macro_f1"),
    ]
    '''
    for metric_title, metric_key in metrics_to_test:
        print(f"\n       {metric_title} COMPARISON")
        print(f"{'Comparison (A vs B)':<35} | {'Diff':<8} | {'p-value':<10} | {'Result'}")
        print("-" * 75)

        for model_a, model_b in comparisons:
            scores_a = results_global[model_a][metric_key]
            scores_b = results_global[model_b][metric_key]
            if not scores_a or not scores_b:
                continue

            diffs = np.array(scores_a) - np.array(scores_b)
            t_val, p_val = corrected_resampled_t_test(diffs, n_train_s, n_test_s)
            mean_diff = np.mean(diffs)

            if p_val < 0.03:
                winner = model_a if mean_diff > 0 else model_b
                result_str = f"** {winner} wins **"
            else:
                result_str = "Tie"

            print(f"{model_a:<15} vs {model_b:<15} | {mean_diff:+.4f}   | {p_val:.5f}    | {result_str}")
    '''
    print("========================================================")


if __name__ == "__main__":
    main()
