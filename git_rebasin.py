import copy
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_slippage_net_layers(model):
    layers = []
    for i in range(1, 2):
        layers.append({
            'linear': getattr(model, f'layer{i}'),
            'norm': getattr(model, f'norm{i}'),
        })
    layers.append({
        'linear': model.output_layer,
        'norm': None,
    })
    return layers


def get_permutation_matrices(model_a, model_b, max_iters=100):
    layers_a = get_slippage_net_layers(model_a)
    layers_b = get_slippage_net_layers(model_b)
    L = len(layers_a)  # 4 (3 hidden + 1 output)

    W_a = [layers_a[l]['linear'].weight.detach().cpu().numpy() for l in range(L)]
    W_b = [layers_b[l]['linear'].weight.detach().cpu().numpy() for l in range(L)]
    b_a = [layers_a[l]['linear'].bias.detach().cpu().numpy()
           if layers_a[l]['linear'].bias is not None else None for l in range(L)]
    b_b = [layers_b[l]['linear'].bias.detach().cpu().numpy()
           if layers_b[l]['linear'].bias is not None else None for l in range(L)]

    ln_w_a, ln_w_b, ln_b_a, ln_b_b = [], [], [], []
    for l in range(L):
        norm = layers_a[l]['norm']
        if norm is not None and hasattr(norm, 'weight') and norm.weight is not None:
            ln_w_a.append(norm.weight.detach().cpu().numpy())
            ln_b_a.append(norm.bias.detach().cpu().numpy())
            ln_w_b.append(layers_b[l]['norm'].weight.detach().cpu().numpy())
            ln_b_b.append(layers_b[l]['norm'].bias.detach().cpu().numpy())
        else:
            ln_w_a.append(None)
            ln_b_a.append(None)
            ln_w_b.append(None)
            ln_b_b.append(None)

    n_hidden = L - 1 
    hidden_dims = [W_a[l].shape[0] for l in range(n_hidden)]

    perms = [np.arange(d) for d in hidden_dims]

    for iteration in range(max_iters):
        progress = False
        layer_order = np.random.permutation(n_hidden)

        for l in layer_order:
            d = hidden_dims[l]
            if l == 0:
                W_b_l_perm = W_b[l]
            else:
                W_b_l_perm = W_b[l][:, perms[l - 1]]

            C = W_a[l] @ W_b_l_perm.T

            if b_a[l] is not None and b_b[l] is not None:
                C += np.outer(b_a[l], b_b[l])

            if ln_w_a[l] is not None:
                C += np.outer(ln_w_a[l], ln_w_b[l])
                C += np.outer(ln_b_a[l], ln_b_b[l])

            if l + 1 < n_hidden:
                W_a_next = W_a[l + 1][perms[l + 1], :]
                W_b_next = W_b[l + 1][perms[l + 1], :]
            else:
                W_a_next = W_a[l + 1]
                W_b_next = W_b[l + 1]

            C += W_a_next.T @ W_b_next

            _, col_ind = linear_sum_assignment(-C)

            if not np.array_equal(col_ind, perms[l]):
                perms[l] = col_ind
                progress = True

        if not progress:
            break

    return perms


def apply_permutation(model_b, perms):
    model_aligned = copy.deepcopy(model_b)
    layers = get_slippage_net_layers(model_aligned)

    for l, perm in enumerate(perms):
        p = torch.LongTensor(perm)

        linear = layers[l]['linear']
        linear.weight.data = linear.weight.data[p]
        if linear.bias is not None:
            linear.bias.data = linear.bias.data[p]

        norm = layers[l]['norm']
        if norm is not None and hasattr(norm, 'weight') and norm.weight is not None:
            norm.weight.data = norm.weight.data[p]
            norm.bias.data = norm.bias.data[p]

        next_linear = layers[l + 1]['linear']
        next_linear.weight.data = next_linear.weight.data[:, p]

    return model_aligned


def weight_matching_align(model_a, model_b, max_iters=100):
    perms = get_permutation_matrices(model_a, model_b, max_iters=max_iters)
    return apply_permutation(model_b, perms)


def align_experts_to_reference(reference_model, expert_models, max_iters=100):
    aligned = []
    for i, expert in enumerate(expert_models):
        print(f"  Aligning expert {i+1}/{len(expert_models)} via Git Re-Basin...")
        aligned_expert = weight_matching_align(reference_model, expert, max_iters=max_iters)
        aligned.append(aligned_expert)
    return aligned


def verify_alignment(model_original, model_aligned, test_input):
    model_original.eval()
    model_aligned.eval()
    with torch.no_grad():
        out_orig = model_original(test_input)
        out_aligned = model_aligned(test_input)
    max_diff = (out_orig - out_aligned).abs().max().item()
    print(f"  Max output difference: {max_diff:.2e} (should be ~0)")
    return max_diff < 1e-5
