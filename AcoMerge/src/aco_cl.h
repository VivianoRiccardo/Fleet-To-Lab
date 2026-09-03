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
#ifndef __ACO_CL_H__
#define __ACO_CL_H__

uint64_t get_aco_cl_weights_or_pheromone_weights_size(int channels, int kernel_rows, int kernel_cols, int n_kernels);
uint64_t get_aco_cl_biases_or_pheromone_biases_size(int n_kernels);
void init_aco_cl_weights_xavier_init(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights);
void init_aco_cl_weights_with_value(int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, float pheromone_value);
void init_aco_cl_biases_zeros(int n_kernels, float* biases);
void init_aco_cl_biases_normalized_random(int n_kernels, float* biases);
void init_aco_cl_biases_with_value(int n_kernels, float* biases, float pheromone_value);
void init_aco_cl_weights_pheromones(int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights_pheromone, float pheromone_value);
void init_aco_cl_biases_pheromones(int n_kernels, float* biases_pheromone, float pheromone_value);
aco_cl* init_aco_cl(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, int identifier, int stride_rows, int stride_cols, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input);
void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output, int stride1, int stride2, int padding_rows, int padding_cols);
void aco_cl_feed_forward(aco_cl* cl);
int aco_cl_input_size(int channels, int input_rows, int input_cols);
int aco_cl_output_size(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols);
int aco_cl_output_rows(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols);
int aco_cl_output_cols(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols);
void print_aco_cl_offsets(aco_cl* f);
aco_cl* copy_aco_cl(aco_cl* a);
void assign_aco_cl_storing_params(aco_cl* a, float* storing_params);
void init_aco_cl_weights_signed_kaiming_constant_init_parameters(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, int n_parameters);
void init_aco_cl_weights_chaos_init_parameters(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, int n_parameters);
void init_aco_cl_biases_signed_kaiming_constant(int input_rows, int input_cols, int channels,int n_kernels, float* biases);
void init_aco_cl_weights_kaiming_constant_init(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights);
void assign_aco_cl_learnable_params(aco_cl* a, float* params);
void assign_aco_cl_storing_partial_derivatives(aco_cl* a, float* d_storing_params);
void assign_aco_cl_learnable_params_scores(aco_cl* a, float* params, int* indices, float* scores);
void convolutional_feed_forward_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output, int stride1, int stride2, int padding_rows, int padding_cols, int* indices, float k_percentage);
void convolutional_back_propagation_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output_error, int stride1, int stride2, int padding_rows, int padding_cols, float* input_error, float* kernel_error, float k_percentage, int* indices);
void convolutional_set_k_percentage(aco_cl* a, float k_percentage);
void aco_cl_feed_forward_edge_popup(aco_cl* cl);
void aco_cl_back_propagation_edge_popup(aco_cl* cl);
void merge_sort_aco_cl(aco_cl* c);
float* get_aco_cl_d_output(aco_cl* cl);
void init_cl_scores(aco_cl* cl);
void set_cl_k_percentage(aco_cl* c, float k);
void paste_cl_indices_weights_according_to_scores(aco_cl* cl, float* params);
void assign_aco_cl_storing_params_output_used(aco_cl* a, int* storing_params);
void set_aco_cl_used_output(aco_cl* cl);
void convolutional_used_output(int input_i, int input_j, int kernel_i, int kernel_j, int channels, int n_kernels, int* output_used, int stride1, int stride2, int padding_rows, int padding_cols, int* indices, float k_percentage);
int* get_aco_cl_used_outputs(aco_cl* a);

#endif
