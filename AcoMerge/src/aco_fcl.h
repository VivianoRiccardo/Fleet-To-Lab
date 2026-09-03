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
#ifndef __ACO_FCL_H__
#define __ACO_FCL_H__

uint64_t get_aco_fcl_weights_or_pheromone_weights_size(int input, int output);
uint64_t get_aco_fcl_biases_or_pheromone_biases_size(int input, int output);
void init_aco_fcl_weights_xavier_init(int input, int output, float* weights);
void init_aco_fcl_weights_with_value(int input, int output, float* weights, float pheromone_value);
void init_aco_fcl_biases_zeros(int output, float* biases);
void init_aco_fcl_biases_normalized_random(int output, float* biases);
void init_aco_fcl_biases_with_value(int output, float* biases, float pheromone_value);
void init_aco_fcl_weights_pheromones(int input, int output, float* weights_pheromone, float pheromone_value);
void init_aco_fcl_biases_pheromones(int output, float* biases_pheromone, float pheromone_value);
aco_fcl* init_aco_fcl(int input_size, int output_size, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input);
void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size);
void aco_fcl_feed_forward(aco_fcl* f);
void print_aco_fcl_offsets(aco_fcl* f);
aco_fcl* copy_aco_fcl(aco_fcl* a);
void assign_aco_fcl_storing_params(aco_fcl* a, float* storing_params);
void init_aco_fcl_weights_signed_kaiming_constant_init_parameters(int input, int output, float* weights, int number_of_parameters);
void init_aco_fcl_weights_chaos_init_parameters(int input, int output, float* weights, int number_of_parameters);
void init_aco_fcl_biases_signed_kaiming_constant(int input, int output, float* biases);
void init_aco_fcl_weights_kaiming_constant_init(int input, int output, float* weights);
void assign_aco_fcl_learnable_params(aco_fcl* a, float* params);
void assign_aco_fcl_storing_partial_derivatives(aco_fcl* a, float* storing_params);
void assign_aco_fcl_learnable_params_scores(aco_fcl* a, float* params, int* indices, float* scores);
void fully_connected_feed_forward_edge_popup(float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, float k_percentage);
void fully_connected_set_k_percentage(aco_fcl* a, float k_percentage);
void fully_connected_back_propagation_edge_popup(float* input, float* output_error, float* weight,float* bias, int input_size, int output_size, float* input_error, float* weight_error, float k_percentage, int* indices);
void aco_fcl_feed_forward_edge_popup(aco_fcl* f);
void aco_fcl_back_propagation_edge_popup(aco_fcl* f);
void merge_sort_aco_fcl(aco_fcl* c);
float* get_aco_fcl_d_output(aco_fcl* fcl);
void init_fcl_scores(aco_fcl* fcl);
void set_fcl_k_percentage(aco_fcl* c, float k);
void paste_fcl_indices_weights_according_to_scores(aco_fcl* fcl, float* params);
void assign_aco_fcl_storing_params_output_used(aco_fcl* a, int* storing_params);
void set_aco_fcl_used_output(aco_fcl* fcl);
void fully_connected_used_output( int* output_used, int input_size, int output_size, int* indices, float k_percentage);
int* get_aco_fcl_used_outputs(aco_fcl* a);

#endif
