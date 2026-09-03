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
#ifndef __ACO_FCL_KOLMOGOROV_ARNOLD_H__
#define __ACO_FCL_KOLMOGOROV_ARNOLD_H__

uint64_t get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(int input, int output, int n_func);
void init_aco_fcl_kolmogorov_arnold_weights_xavier_init(int input, int output,float* weights,int n_func);
void init_aco_fcl_kolmogorov_arnold_weights_kaiming_constant_init(int input, int output, float* weights,int n_func);
void init_aco_fcl_kolmogorov_arnold_weights_signed_kaiming_constant_init_parameters(int input, int output, float* weights, int number_of_parameters,int n_func);
void init_aco_fcl_kolmogorov_arnold_weights_chaos_init_parameters(int input, int output, float* weights, int number_of_parameters,int n_func);
void init_aco_fcl_kolmogorov_arnold_weights_with_value(int input, int output, float* weights, float pheromone_value,int n_func);
void init_aco_fcl_ts_zeros(int input, int output, int n_func, int degree, float* ts);
void init_aco_fclkolmogorov_arnold_ts_normalized_random(int input, int output, int n_func, int degree, float* ts);
void init_aco_fcl_kolmogorov_arnold_ts_signed_kaiming_constant(int input, int output, int n_func, int degree, float* ts);
void init_aco_fcl_kolmogorov_arnold_ts_with_value(int input, int output, int n_func, int degree, float* ts, float pheromone_value);
void init_aco_fcl_kolmogorov_arnold_weights_pheromones(int input, int output, float* weights_pheromone, float pheromone_value, int n_func);
void init_aco_fcl_kolmogorov_arnold_ts_pheromones(int input, int output, int n_func, int degree, float* ts_pheromone, float pheromone_value);
aco_fcl_kolmogorov_arnold* init_aco_fcl_kolmogorov_arnold(int input_size, int output_size,int n_functions, int maximum_degree, int identifier, uint64_t weights_array_offset, uint64_t  ts_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_ts_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* ts, float* pheromone_weights, float* pheromone_ts, float* output, float* input);
void assign_aco_fcl_kolmogorov_arnold_storing_params(aco_fcl_kolmogorov_arnold* a, float* storing_params);
aco_fcl_kolmogorov_arnold* copy_aco_fcl_kolmogorov_arnold(aco_fcl_kolmogorov_arnold* a);
void fully_connected_kolmogorov_arnold_feed_forward(float* input, float* output, float* weight,float* ts, int input_size, int output_size, int n_functions,int degree);
uint64_t get_aco_fcl_kolmogorov_arnold_ts_or_pheromone_biases_size(int input, int output, int n_func, int degree);
void aco_fcl_kolmogorov_arnold_feed_forward(aco_fcl_kolmogorov_arnold* f);
void print_aco_fcl_kolmogorov_arnold_offsets(aco_fcl_kolmogorov_arnold* f);
void init_aco_fcl_kolmogorov_arnold_ts_standard(int input, int output, int n_func, int degree, float* ts);
void init_aco_fcl_kolmogorov_arnold_ts_standard_n_parameters(int input, int output, int n_func, int degree, int n_parameters, float* ts);
void clip_aco_fcl_kolmogorov_arnold_ts(int input, int output, int n_func, int degree, int n_parameters, float* ts);
void aco_fcl_kolmogorov_arnold_clip(aco_fcl_kolmogorov_arnold* f, float* array_to_clip, int n_parameters);


#endif
