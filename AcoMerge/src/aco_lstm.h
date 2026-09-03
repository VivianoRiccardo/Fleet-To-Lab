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
#ifndef __ACO_LSTM_H__
#define __ACO_LSTM_H__

uint64_t get_aco_lstm_weights_or_pheromone_weights_size(int input_size, int output_size);
uint64_t get_aco_lstm_biases_or_pheromone_biases_size(int output_size);
void init_aco_lstm_weights_xavier_init(int input_size, int output_size, float* weights);
void init_aco_lstm_p(int input_size, int output_size, float* weights);
void init_aco_lstm_weights_with_value(int input_size, int output_size, float* weights, float pheromone_value);
void init_aco_lstm_biases_zeros(int output_size, float* biases);
void init_aco_lstm_biases_normalized_random(int output_size, float* biases);
void init_aco_lstm_biases_with_value(int output_size, float* biases, float pheromone_value);
void init_aco_lstm_weights_pheromones(int input_size, int output_size, float* weights_pheromone, float pheromone_value);
void init_aco_lstm_biases_pheromones(int output_size, float* biases_pheromone, float pheromone_value);
aco_lstm* init_aco_lstm(int timestamp, int input_size, int output_size, int window, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset,uint64_t  input_h_array_offset,uint64_t  input_k_array_offset,uint64_t  input_c_array_offset,uint64_t  output_h_array_offset,uint64_t  output_k_array_offset,uint64_t  output_c_array_offset,
                      float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input, float* input_h, float* input_k, float* input_c, float* output_h, float* output_k, float* output_c);
void lstm_feed_forward(float timestamp, float* x, float* h, float* k, float* c, float* cell_state, float* k_state, float* hidden_state, float* z, float* w, float* u, float* b, float p, int input_size, int output_size);
void aco_lstm_feed_forward(aco_lstm* lstm);
int aco_lstm_input_size(int input_size, int output_size);
uint64_t aco_lstm_get_x_input_offset();
uint64_t aco_lstm_get_c_input_offset(int input_size, int output_size);
uint64_t aco_lstm_get_k_input_offset(int input_size, int output_size);
uint64_t aco_lstm_get_h_input_offset(int input_size, int output_size);           
int aco_lstm_get_output_size(int output_size);
int aco_lstm_output_size(int input_size, int output_size);
void print_aco_lstm_offsets(aco_lstm* f);
aco_lstm* copy_aco_lstm(aco_lstm* a);
void assign_aco_lstm_storing_params(aco_lstm* a, float* storing_params);
void init_aco_lstm_weights_signed_kaiming_constant_init_parameters(int input_size, int output_size, float* weights, int number_of_parameters);
void init_aco_lstm_weights_chaos_init_parameters(int input_size, int output_size, float* weights, int number_of_parameters);
void init_aco_lstm_biases_signed_kaiming_constant(int output_size, float* biases);
void init_aco_lstm_weights_kaiming_constant_init(int input_size, int output_size, float* weights);
void assign_aco_lstm_learnable_params(aco_lstm* a, float* params);
void assign_aco_lstm_storing_partial_derivatives(aco_lstm* a, float* storing_params);
void assign_aco_lstm_learnable_params_scores(aco_lstm* a, float* params, int* indices, float* scores);
void lstm_feed_forward_edge_popup(float timestamp, float* x, float* h, float* k, float* c, float* cell_state, float* k_state, float* hidden_state, float* z, float* w, float* u, float* b, float p, int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used);
void lstm_back_propagation_edge_popup(float timestamp, float* x,float* input_error, float* h,float* h_error, float* k,float* k_error, float* c,float* c_error, float* cell_state,float* c_output_error, float* k_state,float* k_output_error, float* hidden_state,float* h_output_error, float* z,float* output_error, float* w, float* d_w, float* u, float* d_u, float* b, float p, int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used);
void aco_lstm_feed_forward_edge_popup(aco_lstm* lstm);
void aco_lstm_back_propagation_edge_popup(aco_lstm* lstm);
void merge_sort_aco_lstm(aco_lstm* c);
float* get_aco_lstm_d_output(aco_lstm* lstm);
void init_lstm_scores(aco_lstm* lstm);
void set_lstm_k_percentage(aco_lstm* c, float k);
void assign_aco_lstm_storing_params_output_used(aco_lstm* a, int* storing_params);
void lstm_used_output(float timestamp,int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used, int* output_h_used);
void set_aco_lstm_used_output(aco_lstm* lstm);
int* get_aco_lstm_used_outputs(aco_lstm* a);

#endif
