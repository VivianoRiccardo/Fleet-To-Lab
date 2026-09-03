#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from libc.stdio cimport FILE
from libc cimport stdint
ctypedef stdint.uint64_t uint64_t
ctypedef stdint.uint8_t uint8_t


cdef extern from "../src/ezspark.h":
    ctypedef struct aco_fcl:
        pass
    ctypedef struct aco_fcl_kolmogorov_arnold:
        pass
    ctypedef struct aco_cl:
        pass
    ctypedef struct aco_pooling:
        pass
    ctypedef struct aco_lstm:
        pass
    ctypedef struct aco_sum:
        pass
    ctypedef struct aco_concatenation:
        pass
    ctypedef struct aco_activation:
        pass
    ctypedef struct aco_splitter:
        pass
    ctypedef struct aco_node:
        pass
    ctypedef struct BitArray:
        pass
    ctypedef struct aco_superstruct:
        pass
    ctypedef struct multithread_aco_superstruct:
        pass
    cdef enum:
        ALL_ACTIVATIONS
    cdef enum:
        NO_ACTIVATION
    cdef enum:
        SIGMOID
    cdef enum:
        TANH
    cdef enum:
        RELU
    cdef enum:
        ELU
    cdef enum:
        LEAKY_RELU
    cdef enum:
        GELU
    cdef enum:
        SOFTMAX
    cdef enum:
        N_ACTIVATION_FUNCTIONS_1_0
    cdef enum:
        MAX_POOLING
    cdef enum:
        AVERAGE_POOLING
    cdef enum:
        STORING
    cdef enum:
        USED_PARAMS
    cdef enum:
        ALL_PARAMS
    cdef enum:
        PHEROMONE_SCORES
    cdef enum:
        HEURISTICS
    cdef enum:
        PI
    cdef enum:
        EPSILON
    cdef enum:
        SUM_FLAG
    cdef enum:
        MUL_FLAG
    cdef enum:
        DIV_FLAG
    cdef enum:
        DIF_FLAG
    cdef enum:
        EXP_FLAG




     
cdef extern from "../src/aco_activation.h":
    aco_activation* init_aco_activation(int activation_flag, int activations_size, int input_size, int output_size,int identifier, uint64_t  output_array_offset,uint64_t  input_array_offset, uint64_t pheromone_activations_offset, float* output, float* input, float* pheromone_activations, int* activations)
    int* generate_all_activations()

cdef extern from "../src/aco_cl.h":
    aco_cl* init_aco_cl(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, int identifier, int stride_rows, int stride_cols, uint64_t weights_array_offset, uint64_t  biases_array_offset,uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input)

cdef extern from "../src/aco_concatenation.h":
    aco_concatenation* init_aco_concatenation(int input_size, int output_size,float value,int identifier, uint64_t  output_array_offset,uint64_t  input_array_offset, uint64_t  input2_array_offset, float* output, float* input, float* input2)

cdef extern from "../src/aco_fcl.h":
    aco_fcl* init_aco_fcl(int input_size, int output_size, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input)

cdef extern from "../src/aco_fcl_kolmogorov_arnold.h":
    aco_fcl_kolmogorov_arnold* init_aco_fcl_kolmogorov_arnold(int input_size, int output_size,int n_functions, int maximum_degree, int identifier, uint64_t weights_array_offset, uint64_t  ts_array_offset,uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_ts_array_offset, uint64_t  output_array_offset,uint64_t  input_array_offset, float* weights, float* ts, float* pheromone_weights, float* pheromone_ts, float* output, float* input)

cdef extern from "../src/aco_lstm.h":
    aco_lstm* init_aco_lstm(int timestamp, int input_size, int output_size, int window, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,uint64_t  input_array_offset,uint64_t  input_h_array_offset,uint64_t  input_k_array_offset,uint64_t  input_c_array_offset,uint64_t  output_h_array_offset,uint64_t  output_k_array_offset,uint64_t  output_c_array_offset,float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input, float* input_h, float* input_k, float* input_c, float* output_h, float* output_k, float* output_c)

cdef extern from "../src/aco_node.h":
    aco_node* init_aco_node(int n_input, int n_output, int identifier, aco_fcl* fcl, aco_cl* cl, aco_lstm* lstm,aco_sum* sum, aco_pooling* pooling, aco_concatenation* concatenation, aco_activation* activation, aco_fcl_kolmogorov_arnold* fcl_ka,aco_splitter* splitter,aco_node** input, aco_node** output)
    void print_aco_node_offsets(aco_node* f)

cdef extern from "../src/aco_pooling.h":
    aco_pooling* init_aco_pooling(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int identifier, int pooling_type,uint64_t  output_array_offset, uint64_t  input_array_offset, float* input, float* output)

cdef extern from "../src/aco_sum.h":
    aco_sum* init_aco_sum(int input_size, int input2_size, int output_size,int identifier,int flag, uint64_t  output_array_offset,uint64_t  input_array_offset,uint64_t  input2_array_offset, float* output, float* input, float* input2)
cdef extern from "../src/aco_splitter.h":
    aco_splitter* init_aco_splitter(int input_size, int output_size,int identifier, uint64_t index_offset, uint64_t  output_array_offset,uint64_t  input_array_offset, float* output, float* input)

cdef extern from "../src/aco_superstruct.h":
    aco_superstruct* init_aco_superstruct(aco_node** nodes,int** adjacency_matrix, int** same_params, int n_nodes)
    void aco_superstruct_activate_heuristic(aco_superstruct* s, double simulated_annealing_temperature, double alpha_factorization_simulated_annealing, double alpha_aco, double beta_aco, float value)
    void reset_heuristic_all_ones(aco_superstruct* s)
    int output_is_relu(aco_node* n, int depth)
    int input_is_positive(aco_node* n, int depth, int is_always_positive_input)
    int nodes_params_already_checked(aco_superstruct* s, int aco_node_index)
    aco_superstruct* copy_aco_superstruct_with_only_storing_params(aco_superstruct* s)
    void aco_superstruct_restrict_lstm_p_value(aco_superstruct* s)
    void aco_superstruct_update_params_eiwoa(aco_superstruct* s)
    void update_eiwoa_params(aco_superstruct* s)
    void eiwoa_run(aco_superstruct* s, float* values, float* pheromones,int path_best, int size)
    aco_superstruct* init_aco_superstruct_without_storing_arrays(aco_node** nodes,int** adjacency_matrix, int** same_params, int n_nodes)
    void generate_parameters_array_to_superstruct(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_pheromone_exploration(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_best_pheromone(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_bit_array(aco_superstruct* s, int index)
    int get_parameter_index_according_to_aco(float* pheromones, int size)
    int get_parameter_index_according_to_aco_with_heuristic(float* pheromones,float* heuristics,double alpha, double beta, int size)
    int get_parameter_index_according_to_aco_with_heuristic_levy_flight(float* pheromones,float* heuristics,double alpha, double beta,double p_threshold,double a, int size)
    int get_parameter_index_according_to_aco_levy_flight(float* pheromones,double p_threshold,double a, int size)
    int get_parameter_index_according_to_aco_best(float* pheromones, int size)
    void aco_superstruct_set_best_local_network_pheromone(aco_superstruct* s, float pheromone)
    void aco_superstruct_set_best_best_network_pheromone(aco_superstruct* s, float pheromone)
    void aco_superstruct_update_taus(aco_superstruct* s)
    void update_current_p(aco_superstruct* s)
    void init_aco_superstruct_general_stuff(aco_superstruct* s,int number_iterations_for_single_aco_iteration,int number_of_iterations,int max_iterations,int n_ants, float tau_min,float  tau_max,float  p_dec, float p, float rho, int time_to_update_with_global_best, float k_zero)
    void set_pheromone_best_trail(aco_superstruct* s, double pheromone)
    void set_local_pheromone(aco_superstruct* s, double pheromone)
    void check_local_best_pheromone(aco_superstruct* s)
    void check_global_best_pheromone(aco_superstruct* s)
    void update_pheromone_according_to_path(float p, float tau_max, float tau_min,double pheromone_amount, BitArray* path, float* pheromones, uint64_t number_of_parameters, uint64_t size, uint64_t offset)
    void update_alpha_aco_beta_aco(aco_superstruct* s, double A, double B, double current_iteration, double total_iterations)
    void update_heuristics(aco_superstruct* s)
    void update_pheromone_according_to_local_best(aco_superstruct* s)
    void update_pheromone_according_to_global_best(aco_superstruct* s)
    void init_aco_superstruct_eiwoa_stuff(aco_superstruct* s, float percentage_of_elements,float alpha_eiwoa,float  beta_eiwoa, float weight_eiwoa, float softmax_temperature, float v_max, float v_min)
    void save_weights_and_pheromone_aco_superstruct(aco_superstruct* sp, int n)
    void load_weights_and_pheromone_aco_superstruct(aco_superstruct* sp, char* pheromone, char* weights)
    void generate_aco_parameters_array_to_superstruct(aco_superstruct* s, int number_of_parameters)
    void generate_aco_parameters_array_to_superstruct_according_to_chaos_initialization(aco_superstruct* s, int number_of_parameters)
    void generate_aco_pheromone_parameters_array_to_superstruct(aco_superstruct* s, float init_value)
    void generate_aco_superstruct_bit_arrays(aco_superstruct* s)
    void free_aco_superstruct(aco_superstruct* s)
    float* get_storing_arrays(aco_node** roots, int n_roots, uint64_t* size, aco_node** nodes, int n_nodes)
    void assign_random_parameters_to_aco_superstruct(aco_superstruct* s)
    void assign_best_parameters_to_aco_superstruct(aco_superstruct* s)
    int adjacency_matrix_count_outputs(int** adjacency_matrix, int n_nodes, int index)
    int adjacency_matrix_count_inputs(int** adjacency_matrix, int n_nodes, int index)
    aco_node** get_leaves(aco_node** nodes, int n_nodes, int* n_leaves)
    aco_node** connect_nodes(aco_node** nodes, int** adjacency_matrix, int* n_roots, int n_nodes)
    void add_group_nodes(int node_index, int input, int output, aco_node** nodes, int** adjacency_matrix, int n_nodes)
    uint64_t get_superstruct_array_size(aco_node** root, int n_roots)
    void set_aco_superstruct_ff_flag(aco_node** nodes,int value, int n_nodes)
    uint64_t get_storing_arrays_size_from_node(aco_node* node, int depth)
    void assign_array_to_node_for_concatenation(aco_node* node, uint64_t* size, float* array, int depth)
    void assign_array_to_node(aco_node* node, uint64_t* size, float* array, int depth)
    void aco_superstruct_node_visit_feed_forward(aco_node* n)
    void aco_superstruct_single_thread_feed_forward(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight_without_setting_weights(aco_superstruct* s)
    void aco_superstruct_clip(aco_superstruct* s, float* array_to_clip, int n_parameters)
    void copy_aco_superstruct_activation(aco_superstruct* copy, aco_superstruct* original)
    uint64_t aco_superstruct_get_total_input_size(aco_superstruct* s)
    uint64_t aco_superstruct_get_total_output_size(aco_superstruct* s)
    void aco_superstruct_select_params_according_to_index_pso(aco_superstruct* s, int index)
    void aco_update_pso(aco_superstruct* s)
    void aco_update_pso_parameters(aco_superstruct* s)
    void generate_aco_pso_y_alphas_array_to_superstruct(aco_superstruct* s, float init_value)
    void aco_superstruct_copy_inputs(aco_superstruct* s, float** inputs, int n_inputs)
    void reassign_right_storing_arrays(aco_superstruct* s)
    void aco_superstruct_assign_outputs(aco_superstruct* s, float** outputs, int n_outputs)
    int aco_superstruct_get_number_of_roots(aco_superstruct* s)
    int aco_superstruct_getnumber_of_leaves(aco_superstruct* s)
    int aco_superstruct_get_first_size_root_i(aco_superstruct* s, int index)
    int aco_superstruct_get_second_size_root_i(aco_superstruct* s, int index)
    void save_weights_single_model(aco_superstruct* sp, int n)
    void load_weights_single_model(aco_superstruct* s, char* weights)
    void generate_pso_parameters_array_to_superstruct(aco_superstruct* s, int number_of_particles)
    void assign_aco_superstruct_learnable_params(aco_superstruct* s)
    void set_aco_superstruct_array_to_null(aco_superstruct* s, int index)
    void set_losses_according_to_index(aco_superstruct* s, int index, double loss)
    void increase_iteration_index(aco_superstruct* s)
    void reset_pheromone_according_to_index(aco_superstruct* s, int index)
    void aco_superstruct_set_all_params_to_null(aco_superstruct* s)
    void set_aco_superstruct_array_all_params(aco_superstruct* copy_from, aco_superstruct* copy_to, int index)
    void copy_aco_superstruct_array(aco_superstruct* copy_from, aco_superstruct* copy_to, int index)
    void set_pheromone_for_index(aco_superstruct* s, int index, float pheromone)
    void reset_explorative_heuristics_support(aco_superstruct* s)
    void aco_superstruct_activate_explorative_supporting_heuristic_array(aco_superstruct* s)
    void aco_increase_explorative_supporting_heuristic_array(aco_superstruct* s)
    void update_heuristics_according_to_explorative_supporting_heuristic_array(aco_superstruct* s)
    void check_global_best_pheromone_without_updating_heuristic(aco_superstruct* s)
    void set_max_aco_alpha_beta(aco_superstruct* s, float max_alpha_aco, float max_beta_aco,float min_alpha_aco, float min_beta_aco,  float exponential_alpha_decay, float exponential_beta_decay)
    void update_alpha_beta_params(aco_superstruct* s)
    void set_alpha_and_beta_aco(aco_superstruct* s, float alpha_aco,float beta_aco)
    uint8_t* get_bit_array_data_from_index(aco_superstruct* s, int index)
    void* set_bit_array_data_from_index(aco_superstruct* s, uint8_t* array, int index)
    void set_aco_superstruct_only_for_inference(aco_superstruct* s)
    void check_local_best_pheromone_for_gpu(aco_superstruct* s)
    void copy_array_weights_to_specific_struct_weight_position(aco_superstruct* s, float* array, int index, int size)
    void copy_current_weight_combination_to_output_array(aco_superstruct* s, float* array)
    float* get_current_weight_combination_to_output_array(aco_superstruct* s)
    int* get_indices_from_local_best(aco_superstruct* s)
    int get_indices_size(aco_superstruct* s)
    
cdef extern from "../src/utils.h":
    void print_vector(float* vector, int size)
    
cdef extern from "../src/initialization.h":
    float normalized_random();

cdef extern from "../src/multithread_aco_superstruct.h":
    float* get_multi_output_from_multi_input_aco_superstruct_ff(aco_superstruct** s, float* input, int n_superstructs,  int n_inputs)

        
