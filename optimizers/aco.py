import copy
import numpy as np
import pyezsparkc

from config import GA_GENERATIONS, EXPERT_MULTIPLIER, INPUT_DIM_EXPERT, NUM_CLASSES, DEVICE
from utils import model_to_array, array_to_model, get_predictions, compute_fitness_cross_entropy, compute_test_metrics
from networks import SlippageNet


def run_aco_merging(model_list, val_loader, y_true_val, base_arch, test_loader=None, y_test=None):
    print(f"   [ACO Merging] Initializing for {len(model_list)} models...")
    experts = model_list
    weights = model_to_array(base_arch)
    total_size = weights.shape[0]

    #fcl1 = pyezsparkc.acoFcl(13301, 1, 0)
    fcl1 = pyezsparkc.acoFcl(543, 1, 0)
    activation1 = pyezsparkc.acoActivation(pyezsparkc.PY_RELU, 0, 1, 1, 1)

    number_of_nodes = 2
    matrix = []
    params = []
    for i in range(number_of_nodes - 1):
        l = [0] * number_of_nodes
        l[i + 1] = 1
        matrix.append(l)
        params.append([0] * number_of_nodes)
    params.append([0] * number_of_nodes)
    matrix.append([0] * number_of_nodes)

    number_of_parameters = 3 * (EXPERT_MULTIPLIER + 1)
    initial_pheromone = 15
    number_of_ants = 300
    count_update_x = 0
    time_to_update_with_global_best = 8
    iteration_index = 0
    global_best_before = -1
    number_iterations_for_single_aco_iteration = 12
    number_of_x_iterations = 11
    

    modelpy = pyezsparkc.acoSuperstruct(
        [activation1], [], [], [fcl1], [], [], [], [],
        matrix, params
    )
    modelpy.initialize_aco_eiwoa(
        number_of_parameters, initial_pheromone,
        number_of_ants=number_of_ants,
        number_of_single_aco_iteration=number_iterations_for_single_aco_iteration,
        number_of_total_iterations=GA_GENERATIONS+1,
        number_of_x_iterations=number_of_x_iterations
    )

    for k in range(number_of_parameters):
        modelpy.copy_array_weights_to_specific_struct_weight_position(
            model_to_array(experts[k]), k, total_size
        )

    current_best_weights = model_to_array(experts[0])

    track_test = (test_loader is not None and y_test is not None)
    gen_metrics = []  # Per-generation test metrics

    for gen in range(GA_GENERATIONS):
        modelpy.reset_pheromone_according_to_index(1)
        modelpy.reset_pheromone_according_to_index(2)
        global_best = 0

        for k in range(number_of_ants):
            modelpy.random_select_network()
            combination = modelpy.get_current_weight_combination_to_output_array(total_size)
            array_to_model(experts[1], combination)

            preds, probs = get_predictions(experts[1], val_loader)
            fitness = compute_fitness_cross_entropy(probs, y_true_val)

            if fitness > global_best:
                global_best = fitness
                current_best_weights = model_to_array(experts[1])

            modelpy.set_local_pheromone(fitness)
            modelpy.replace_local_best_with_current()

        iteration_index += 1
        count_update_x += 1
        modelpy.increase_internal_iteration_index()
        modelpy.replace_global_best_with_best_local()
        modelpy.update_taus()
        modelpy.update_pheromone_according_to_local_best()
        modelpy.update_current_p()

        if count_update_x % time_to_update_with_global_best == 0:
            modelpy.update_pheromone_according_to_global_best()
        if global_best > global_best_before:
            modelpy.update_heuristic_according_to_global_best()
            global_best_before = global_best
        if iteration_index % number_iterations_for_single_aco_iteration == 0:
            modelpy.update_eiwoa()
            count_update_x = 0
            modelpy.update_heuristic_according_to_global_best()
        
        #if (gen + 1) % 5 == 0:
        #    print(f"   [ACO] Gen {gen+1}/{GA_GENERATIONS} Best: {global_best:.4f}")

        # Evaluate current best on test set
        if track_test:
            eval_model = copy.deepcopy(base_arch)
            array_to_model(eval_model, current_best_weights)
            bal_acc, macro_f1, loss = compute_test_metrics(eval_model, test_loader, y_test)
            gen_metrics.append({
                'generation': gen + 1,
                'bal_acc': bal_acc,
                'f1_macro': macro_f1,
            })

    final_model = copy.deepcopy(base_arch)
    array_to_model(final_model, current_best_weights)
    return final_model, gen_metrics
