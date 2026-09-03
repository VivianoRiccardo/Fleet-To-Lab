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
#ifndef __ACO_NODE_H__
#define __ACO_NODE_H__

aco_node* init_aco_node(int n_input, int n_output, int identifier, aco_fcl* fcl, aco_cl* cl, aco_lstm* lstm,
                        aco_sum* sum, aco_pooling* pooling, aco_concatenation* concatenation, aco_activation* activation, aco_fcl_kolmogorov_arnold* fcl_ka,aco_splitter* splitter,
                        aco_node** input, aco_node** output);
void aco_node_feed_forward(aco_node* n);
void set_ff_flag_to_zero(aco_node* n);
void aco_node_add_input(aco_node* n, aco_node* input);
void aco_node_add_output(aco_node* n, aco_node* output);
void print_aco_node_offsets(aco_node* f);
void free_aco_node(aco_node* n);
void assign_array_aco_node_params(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index);
uint64_t get_aco_node_params_size(int** same_params, aco_node* n, int n_nodes, int index);
aco_node* copy_aco_node_without_input_and_output(aco_node* n);
void assign_aco_node_storing_params(aco_node* a, float* storing_params);
int get_aco_node_output_size(aco_node* n);
int get_aco_node_input_size(aco_node* n);
void assign_aco_node_inputs(aco_node* n, float* input1, float* input2);
void assign_aco_node_outputs(aco_node* n, float* output);
void generate_array_aco_node_params(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params);
uint64_t get_aco_node_input_offset(aco_node* n);
uint64_t get_aco_node_output_offset(aco_node* n);
void generate_array_aco_node_params_chaos_initialization(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params);
void aco_node_back_propagation(aco_node* n, int n_parameters);
float* get_aco_node_output(aco_node* n);
float* get_aco_node_output_error(aco_node* n);
void assign_array_aco_node_params_only_for_kolmogorov_arnold_ts(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index);
void generate_array_aco_node_params_only_for_kolmogorov_arnold_ts(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params);
void clip_aco_node(aco_node* n, float* array_to_clip, int n_parameters);
void copy_aco_node_activation(aco_node* copy, aco_node* original);
int get_aco_node_first_input_size(aco_node* n);
int get_aco_node_second_input_size(aco_node* n);
void assign_aco_node_learnable_params(aco_node* a, float* params);
void merge_sort_aco_node(aco_node* c);
void assign_aco_node_storing_partial_derivatives(aco_node* n, float* params);
void assign_aco_node_learnable_params_scores(aco_node* a, float* params, int* indices, float* scores);
void aco_node_feed_forward_edge_popup(aco_node* n);
void aco_node_back_propagation_edge_popup(aco_node* n);
float* get_aco_node_d_output(aco_node* node);
void init_aco_node_scores(aco_node* node);
void set_aco_node_k_percentage(aco_node* n, float k_percentage);
void assign_aco_node_storing_params_output_used(aco_node* a, int* storing_params);
void set_aco_node_used_output(aco_node* n);
void set_aco_node_used_output_no_parametric(aco_node* n,int* previous_used_outputs1,int* previous_used_outputs2, int is_root, uint64_t previous_used_outputs1_offset, uint64_t previous_used_outputs2_offset);
int* get_aco_node_used_outputs(aco_node* a);

#endif
