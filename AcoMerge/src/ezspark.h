/*
MIT License

Copyright (c) 2023 Viviano Riccardo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files ((the "LICENSE")), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef __EZSPARK_H__
#define __EZSPARK_H__

#ifdef _WIN32
typedef unsigned int uint;
#else
//typedef unsigned int uint;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <inttypes.h>

#define ALL_ACTIVATIONS -1
#define NO_ACTIVATION 0
#define SIGMOID 1
#define TANH 2
#define RELU 3
#define ELU 4
#define LEAKY_RELU 5
#define GELU 6
#define SOFTMAX 7
#define INVERSE 8
#define NEGATIVE 9
#define SQUARE 10
#define SQUARE_ROOT 11

#define N_ACTIVATION_FUNCTIONS_1_0 4

#define MAX_POOLING 1
#define AVERAGE_POOLING 2

#define SUM_FLAG 1
#define MUL_FLAG 2
#define DIV_FLAG 3 // not supported yet by edge popup
#define DIF_FLAG 4 // not supported yet by edge popup
#define EXP_FLAG 5 // not supported yet by edge popup


#define STORING 0
#define USED_PARAMS 1
#define ALL_PARAMS 2
#define PHEROMONE_SCORES 3
#define HEURISTICS 4



#define PI 3.14159265358979323846
#define EPSILON 1e-7

#define NO_LOSS 0
#define FOCAL_LOSS 1

#define EUCLIDIAN_DISTANCE 0
#define COSINE_SIMILARITY 1



typedef struct aco_fcl{
    int input_size, output_size, identifier;
    float k_percentage;
    float* weights;
    float* biases;
    float* all_weights;
    float* all_biases;
    float* scores;// edge popup
    float* d_weights;// edge popup
    float* d_biases;
    float* pheromone_weights;
    float* pheromone_biases;
    float* output;
    float* input;
    float* output_error;// edge popup
    float* input_error;// edge popup
    int* indices;// edge popup
    int* output_used;//output_size
    uint64_t weights_array_offset, biases_array_offset, pheromone_weights_array_offset, pheromone_biases_array_offset, output_array_offset, input_array_offset;
}aco_fcl;

// no edge-popup here
typedef struct aco_fcl_kolmogorov_arnold{
    int input_size, output_size,maximum_degree, n_functions, identifier;
    float* weights;
    float* ts;
    float* pheromone_weights;
    float* pheromone_ts;
    float* output;
    float* input;
    float* output_error;
    float* input_error;
    uint64_t weights_array_offset, ts_array_offset, pheromone_weights_array_offset, pheromone_ts_array_offset, output_array_offset, input_array_offset;
}aco_fcl_kolmogorov_arnold;

typedef struct aco_cl{
    int channels, kernel_rows, kernel_cols, n_kernels, stride_rows, stride_cols, input_rows, input_cols, identifier, padding_rows, padding_cols;
    float k_percentage;
    float* weights;
    float* biases;
    float* all_weights;
    float* all_biases;
    float* scores;// edge popup
    float* d_weights;// edge popup
    float* d_biases;
    float* pheromone_weights;
    float* pheromone_biases;
    float* output;
    float* input;
    float* output_error;// edge popup
    float* input_error;// edge popup
    int* indices;
    int* output_used;//output_size
    uint64_t weights_array_offset, biases_array_offset, pheromone_weights_array_offset, pheromone_biases_array_offset, output_array_offset, input_array_offset;
}aco_cl;

typedef struct aco_pooling{
    int input_rows, input_cols, channels, pooling_rows, pooling_cols, stride_rows, stride_cols, pooling_type, identifier, padding_rows, padding_cols;
    float* output;
    float* input;
    float* output_error;// edge popup
    float* input_error;// edge popup
    uint64_t input_array_offset, output_array_offset;
    int* output_used;//output_size
}aco_pooling;


typedef struct aco_lstm{
    int input_size, output_size, window, timestamp, identifier;
    float k_percentage;
    float* weights;
    float* biases;
    float* all_weights;
    float* all_biases;
    float* d_weights;
    float* d_biases;
    float* pheromone_weights;
    float* pheromone_biases;
    float* output;
    float* output_h;
    float* output_k;
    float* output_c;
    float* input;
    float* input_h;
    float* input_k;
    float* input_c;
    float* scores;// edge popup
    float* output_error;// edge popup
    float* output_h_error;// edge popup
    float* output_k_error;// edge popup
    float* output_c_error;// edge popup
    float* input_error;// edge popup
    float* input_h_error;// edge popup
    float* input_k_error;// edge popup
    float* input_c_error;// edge popup
    int* indices;
    int* output_used;//output_size
    int* output_h_used;//output_size
    uint64_t weights_array_offset, biases_array_offset, pheromone_weights_array_offset, pheromone_biases_array_offset, output_array_offset, input_array_offset, output_h_array_offset,output_k_array_offset, input_h_array_offset,input_k_array_offset, output_c_array_offset, input_c_array_offset;
}aco_lstm;

typedef struct aco_sum{
    int input_size, input2_size, output_size, identifier, flag;
    float* output;
    float* input;
    float* input2;
    float* output_error;// edge popup
    float* input_error;// edge popup
    float* input2_error;// edge popup
    int* output_used;//output_size
    uint64_t input_array_offset,input2_array_offset, output_array_offset;
}aco_sum;

typedef struct aco_splitter{
    int input_size, output_size, identifier;
    float* output;
    float* input;
    float* output_error;// edge popup
    float* input_error;// edge popup
    int* output_used;//output_size
    uint64_t input_array_offset, output_array_offset, index_offset;
}aco_splitter;

typedef struct aco_concatenation{
    int input_size,input2_size, output_size, identifier;
    float value;
    float* output;
    float* input;
    float* input2;
    float* output_error;// edge popup
    float* input_error;// edge popup
    float* input2_error;// edge popup
    int* output_used;//output_size
    uint64_t input_array_offset,input2_array_offset, output_array_offset;
}aco_concatenation;

typedef struct aco_activation{
    int activations_size,input_size, output_size,activation_flag,identifier;
    int* activations;
    float* pheromone_activations;
    float* input;
    float* output;
    float* input_error;// edge popup
    float* output_error;// edge popup
    int* output_used;//output_size
    uint64_t input_array_offset, output_array_offset,pheromone_activations_offset;
}aco_activation;

typedef struct aco_node aco_node;

struct aco_node{
    int ff_flag, identifier, n_input, n_output;
    aco_fcl* fcl;
    aco_fcl_kolmogorov_arnold* fcl_ka;
    aco_cl* cl;
    aco_lstm* lstm;
    aco_sum* sum;
    aco_pooling* pooling;
    aco_concatenation* concatenation;
    aco_activation* activation;
    aco_splitter* splitter;
    aco_node** input;
    aco_node** output;
};

typedef struct BitArray{
    //char* data;
    uint8_t* data;
    uint64_t size; // size in bytes
    uint64_t currentBit; // current bit position
    double pheromone;
} BitArray;

typedef struct aco_superstruct{
    int n_roots, n_nodes, n_arrays,n_int_arrays, n_leaves, n_activation_functions, n_bit_arrays, scores_partial_derivative_index;
    aco_node** roots;
    aco_node** aco_nodes;
    aco_node** leaves;
    BitArray** bayesan_bit_arrays;
    uint64_t* array_sizes;
    uint64_t* int_array_sizes;
    float** arrays;
    int* int_arrays;
    int** adjacency_matrix;
    int** same_params;
    BitArray** bit_arrays;//3 best one, best in the current, the current
    // general aco stuff
    int number_of_iterations, max_iterations, n_ants, iteration_index, time_to_update_with_global_best,number_iterations_for_single_aco_iteration, use_heurisitc, free_all_params;
    float tau_min, tau_max, init_tau, p_dec, p, pheromone_best_trail, average, rho, k_zero, current_p;
    double alpha_aco, beta_aco, v_min, v_max, softmax_temperature, max_alpha_aco, max_beta_aco,min_alpha_aco, min_beta_aco, exponential_alpha_decay, exponential_beta_decay;
    
    
    //pso
    double dt,m,gamma,lambda1,lambda2,sigma1,sigma2,alpha;
    
    //pso
    double* losses;//number of particles
    
    // eiwoa stuff
    float alpha_eiwoa, beta_eiwoa, weight_eiwoa, percentage_of_elements;

    
    //levy flight
    float levy_threshold,altering_ratio;
    
    int only_for_inference;
    
    
}aco_superstruct;


typedef struct edge_popup_trainer{
    int batch_size, optimizer_flag, mini_batch_size;
    int* error_flags;// n_leaves (from aco_superstruct)
    float lr, epsilon, m0, n0, mt, nt; 
    aco_superstruct** s;// batch_size
    int* n_arrays; // batch_size
    int* int_n_arrays; // batch_size
    uint64_t** array_sizes;//batch_size X n_arrays[i]
    uint64_t** int_array_sizes;//batch_size X int_n_arrays[i]
    float*** arrays; // batch_size X n_arrays[i] X array_sizes[i][j] 
    int*** int_arrays; // batch_size X int_n_arrays[i] X int_array_sizes[i][j] 
}edge_popup_trainer;



typedef struct multithread_aco_superstruct{
    uint64_t input_offset, output_offset, n_inputs_outputs, index, input_size, output_size;
    float* input, *output, *real_output;
    int* indices;
    double ret_err;
    int n_node_identifiers, n_parameters;
    aco_superstruct* s;
    int* node_identifiers;
    edge_popup_trainer* e;
}multithread_aco_superstruct;

typedef struct multithread_aco_sort{
	aco_superstruct* s;
	int start,end;
}multithread_aco_sort;


#include "aco_activation.h"
#include "aco_cl.h"
#include "aco_concatenation.h"
#include "aco_fcl.h"
#include "aco_fcl_kolmogorov_arnold.h"
#include "aco_lstm.h"
#include "aco_node.h"
#include "aco_pooling.h"
#include "aco_splitter.h"
#include "aco_sum.h"
#include "aco_superstruct.h"
#include "bit_array.h"
#include "distance.h"
#include "edge_popup_trainer.h"
#include "initialization.h"
#include "louvain.h"
#include "math_functions.h"
#include "multithread_aco_superstruct.h"
#include "optimizers.h"
#include "parser.h"
#include "utils.h"

#endif
