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
#ifndef __ACO_ACTIVATION_H__
#define __ACO_ACTIVATION_H__

aco_activation* init_aco_activation(int activation_flag, int activations_size, int input_size, int output_size,int identifier, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, uint64_t pheromone_activations_offset, float* output, float* input, float* pheromone_activations, int* activations);
void aco_activation_ff(float* input,float* output,int activation_flag, int size);
void aco_activation_feed_forward(aco_activation* activation);
void print_aco_activation_offsets(aco_activation* f);
void free_aco_activation(aco_activation* a);
aco_activation* copy_aco_activation(aco_activation* a);
void assign_aco_activation_storing_params(aco_activation* a, float* storing_params);
void assign_aco_activation_storing_params(aco_activation* a, float* storing_params);
int* generate_all_activations();
void aco_activation_bp(float* input,float* input_error, float* output,float* output_error, int* output_used, int activation_flag, int size);
void aco_activation_back_propagation(aco_activation* activation);
void copy_activation_flag(aco_activation* copy, aco_activation* original);
void assign_aco_activation_storing_partial_derivatives(aco_activation* a, float* d_storing_params);
float* get_aco_activations_d_output(aco_activation* activation);
void assign_aco_activation_storing_params_output_used(aco_activation* a, int* storing_params);
void set_aco_activation_used_output(aco_activation* a,int* previous_used_outputs, int is_root);
int* get_aco_activation_used_outputs(aco_activation* a);
void aco_activation_ff_edge_popup(float* input,float* output,int* output_used, int activation_flag, int size);
void aco_activation_feed_forward_edge_popup(aco_activation* activation);

#endif
