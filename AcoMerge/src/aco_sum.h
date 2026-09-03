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
#ifndef __ACO_SUM_H__
#define __ACO_SUM_H__

aco_sum* init_aco_sum(int input_size, int input2_size, int output_size,int identifier,int flag, uint64_t  output_array_offset,
                      uint64_t  input_array_offset,uint64_t  input2_array_offset, float* output, float* input, float* input2);
void aco_sum_ff(float* input1, float* input2,float* output, int input_size1, int input_size2, int output_size, int flag);
void aco_sum_feed_forward(aco_sum* sum);
void print_aco_sum_offsets(aco_sum* f);
aco_sum* copy_aco_sum(aco_sum* a);
void assign_aco_sum_storing_params(aco_sum* a, float* storing_params);
void aco_sum_bp(float* input1,float* input1_error, float* input2,float* input2_error, float* output,float* output_error, int input_size1, int input_size2, int output_size, int flag, int* used_outputs);
void aco_sum_back_propagation(aco_sum* sum);
void assign_aco_sum_storing_partial_derivatives(aco_sum* a, float* storing_params);
float* get_aco_sum_d_output(aco_sum* sum);
void assign_aco_sum_storing_params_output_used(aco_sum* a, int* storing_params);
void set_aco_sum_used_output(aco_sum* a,int* previous_used_outputs1,int* previous_used_outputs2, int is_root);
int* get_aco_sum_used_outputs(aco_sum* a);
void aco_sum_ff_edge_popup(float* input1, float* input2,float* output, int input_size1, int input_size2, int output_size, int flag, int* used_outputs);
void aco_sum_feed_forward_edge_popup(aco_sum* sum);

#endif
