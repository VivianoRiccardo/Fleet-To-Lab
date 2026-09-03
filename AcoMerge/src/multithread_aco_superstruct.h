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
#ifndef __MULTITHREAD_ACO_SUPERSTRUCT_H__
#define __MULTITHREAD_ACO_SUPERSTRUCT_H__

void* aco_superstruct_multi_input_multi_output_ff(void* _args);
float* get_multi_output_from_multi_input_aco_superstruct_ff(aco_superstruct** s, float* input, int n_superstructs,  int n_inputs);
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs);
void* aco_superstruct_multi_input_multi_output_ff_focal_loss(void* _args);
void* aco_superstruct_multi_input_multi_output_ff_focal_loss_with_heuristic(void* _args);
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss_with_heuristic(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs);
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss_with_indices(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs, int* indices);
void* aco_superstruct_multi_input_multi_output_ff_focal_loss_indices(void* _args);
void* aco_superstruct_multi_input_multi_output_ff_loss_bp(void* _args);
void ff_loss_bp_edge_popup(edge_popup_trainer* e, float* input,float* real_output,  int n_inputs, int* indices);
void* sum_partial_derivatives_thread(void* _args);
void multi_thread_sum_partial_derivatives(edge_popup_trainer* e, int n_used_threads, int threads);
void* set_partial_derivatives_to_zero_thread(void* _args);
void multi_thread_set_to_zero_derivatives(edge_popup_trainer* e);
void* sum_partial_derivatives_update_reset_thread(void* _args);
void multi_thread_sum_partial_derivatives_update_reset(edge_popup_trainer* e, int n_used_threads, int threads);
void* aco_superstruct_multi_input_multi_output_ff_edge_popup(void* _args);
float* get_multi_output_from_multi_input_aco_superstruct_ff_edge_popup(aco_superstruct** s, float* input, int n_superstructs,  int n_inputs);
#endif
