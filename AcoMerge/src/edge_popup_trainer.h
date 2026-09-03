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
#ifndef __EDGE_POPUP_TRAINER_H__
#define __EDGE_POPUP_TRAINER_H__

edge_popup_trainer* init_edge_popup_trainer(aco_superstruct** s, int batch_size, int optimizer_flag,int mini_batch_size, int* error_flags, float lr, float epsilon,float m0, float n0);
void free_edge_popup_trainer(edge_popup_trainer* s);
void add_superstruct_storing_error_array_according_to_index(edge_popup_trainer* e, int index);
void add_superstruct_weight_like_array_according_to_index(edge_popup_trainer* e, int index);
void add_superstruct_weight_like_int_array_according_to_index(edge_popup_trainer* e, int index);
void sort_scores(edge_popup_trainer* e);
void assign_bp_storing_params_according_to_index(edge_popup_trainer* e, int index);
void assign_edge_popup_params_according_to_index(edge_popup_trainer* e, int index);
void set_partial_derivative_to_zero_according_to_index(edge_popup_trainer* e, int index);
void update_model(edge_popup_trainer* e);
void update_edge_popup_parameters(edge_popup_trainer* e);
void add_superstruct_storing_used_output(edge_popup_trainer* e);
void assign_superstruct_storing_used_output_according_to_index(edge_popup_trainer* e, int index);
void multi_thread_sort_scores(edge_popup_trainer* e, int threads);

#endif
