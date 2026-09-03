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
#ifndef __ACO_POOLING_H__
#define __ACO_POOLING_H__

aco_pooling* init_aco_pooling(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int identifier, int pooling_type,
                              uint64_t  output_array_offset, uint64_t  input_array_offset, float* input, float* output);
void max_pooling_feed_forward(float* input, float* output, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols);
void average_pooling_feed_forward(float* input, float* output, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols);
void aco_pooling_feed_forward(aco_pooling* pooling);
int aco_pooling_output_rows(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int strinde_rows, int stride_cols, int padding_rows, int padding_cols);
int aco_pooling_output_cols(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int strinde_rows, int stride_cols, int padding_rows, int padding_cols);
int aco_pooling_output_size(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int strinde_rows, int stride_cols, int padding_rows, int padding_cols);
int aco_pooling_input_size(int channels, int input_rows, int input_cols);
void print_aco_pooling_offsets(aco_pooling* f);
aco_pooling* copy_aco_pooling(aco_pooling* a);
void assign_aco_pooling_storing_params(aco_pooling* a, float* storing_params);  
void average_pooling_back_propagation(float* input,float* input_error, float* output,float* output_error, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols);
void max_pooling_back_propagation(float* input,float* input_error, float* output,float* output_error, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols);
void aco_pooling_back_propagation(aco_pooling* pooling);
void assign_aco_pooling_storing_partial_derivatives(aco_pooling* a, float* storing_params);
float* get_aco_pooling_d_output(aco_pooling* pooling);
void assign_aco_pooling_storing_params_output_used(aco_pooling* a, int* storing_params);
void set_aco_pooling_used_output(aco_pooling* a,int* previous_used_outputs, int is_root);
int* get_aco_pooling_used_outputs(aco_pooling* a);


#endif
