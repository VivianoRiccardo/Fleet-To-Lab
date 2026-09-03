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

#include "ezspark.h"

// cannot be a root if index_offset > 0
// cannot have a concatenation after
// cannot have a splitter after

aco_splitter* init_aco_splitter(int input_size, int output_size,int identifier, uint64_t index_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* output, float* input){
    
    
    aco_splitter* f = (aco_splitter*)malloc(sizeof(aco_splitter));
    f->input_size = input_size;
    f->output_size = output_size;
    f->identifier = identifier;
    f->index_offset = index_offset;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->output = output;
    f->input = input;
    f->input_error = NULL;
    f->output_error = NULL;
    f->output_used = NULL;
    return f;
}

aco_splitter* copy_aco_splitter(aco_splitter* a){
    if(a == NULL)    
        return NULL;
    aco_splitter* n = (aco_splitter*)malloc(sizeof(aco_splitter));
    n->input_size = a->input_size;
    n->output_size = a->output_size;
    n->identifier = a->identifier;
    n->index_offset = a->index_offset;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->output = a->output;
    n->input = a->input;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_splitter_storing_params(aco_splitter* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

void assign_aco_splitter_params_output_used(aco_splitter* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_splitter_storing_partial_derivatives(aco_splitter* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
}


float* get_aco_splitter_d_output(aco_splitter* a){
    if(a == NULL)
        return NULL;
    a->output_error;
}

void set_aco_splitter_used_output(aco_splitter* a,int* previous_used_outputs, int is_root){
    if(a == NULL)
        return;
    if(is_root)
        set_int_vector_with_value(1,a->output_used,a->output_size);
    else{
        copy_int_array(previous_used_outputs,a->output_used,a->output_size);
    }
        
}

int* get_aco_splitter_used_outputs(aco_splitter* a){
    if(a == NULL)
        return NULL;
    return a->output_used;
}



