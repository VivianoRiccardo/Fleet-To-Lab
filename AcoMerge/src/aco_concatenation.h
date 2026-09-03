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
#ifndef __ACO_CONCATENATION_H__
#define __ACO_CONCATENATION_H__

aco_concatenation* init_aco_concatenation(int input_size, int output_size,float value,int identifier, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, uint64_t  input2_array_offset, float* output, float* input, float* input2);
void aco_concatenation_ff(float input,float* output, int output_size);
void aco_concatenation_feed_forward(aco_concatenation* concatenation);
void print_aco_concatenation_offsets(aco_concatenation* f);                     
aco_concatenation* copy_aco_concatenation(aco_concatenation* a);
void assign_aco_concatenation_storing_params(aco_concatenation* a, float* storing_params);
void assign_aco_concatenation_storing_partial_derivatives(aco_concatenation* a, float* storing_params);
void aco_concatenation_back_propagation(aco_concatenation* f);
float* get_aco_concatenation_d_output(aco_concatenation* concatenation);
void assign_aco_concatenation_storing_params_output_used(aco_concatenation* a, int* storing_params);
void set_aco_concatenation_used_output(aco_concatenation* a,int* previous_used_outputs1,int* previous_used_outputs2, int is_root, uint64_t offset1, uint64_t offset2);
int* get_aco_concatenation_used_outputs(aco_concatenation* a);

#endif
