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
#ifndef __ACO_SPLITTER_H__
#define __ACO_SPLITTER_H__

aco_splitter* init_aco_splitter(int input_size, int output_size,int identifier, uint64_t index_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* output, float* input);
aco_splitter* copy_aco_splitter(aco_splitter* a);
void assign_aco_splitter_storing_params(aco_splitter* a, float* storing_params);
void assign_aco_splitter_params_output_used(aco_splitter* a, int* storing_params);
void assign_aco_splitter_storing_partial_derivatives(aco_splitter* a, float* storing_params);
float* get_aco_splitter_d_output(aco_splitter* a);
int* get_aco_splitter_used_outputs(aco_splitter* a);
void set_aco_splitter_used_output(aco_splitter* a,int* previous_used_outputs, int is_root);

#endif
