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


aco_concatenation* init_aco_concatenation(int input_size, int output_size,float value,int identifier, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, uint64_t  input2_array_offset, float* output, float* input, float* input2){
    
    
    aco_concatenation* f = (aco_concatenation*)malloc(sizeof(aco_concatenation));
    f->input_size = input_size;
    f->output_size = output_size;
    f->value = value;
    f->identifier = identifier;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->input2_array_offset = input2_array_offset;
    f->output = output;
    f->input = input;
    f->input2 = input2;
    f->input_error = NULL;
    f->input2_error = NULL;
    f->output_error = NULL;
    f->output_used = NULL;
    return f;
}

aco_concatenation* copy_aco_concatenation(aco_concatenation* a){
    if(a == NULL)
        return NULL;
    aco_concatenation* n = (aco_concatenation*)malloc(sizeof(aco_concatenation));
    n->input_size = a->input_size;
    n->output_size = a->output_size;
    n->value = a->value;
    n->identifier = a->identifier;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->input2_array_offset = a->input2_array_offset;
    n->output = a->output;
    n->input = a->input;
    n->input2 = a->input2;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_concatenation_storing_params(aco_concatenation* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
    if(a->input2 != NULL){
        a->input2 = storing_params+a->input2_array_offset;
    }
}

void assign_aco_concatenation_storing_params_output_used(aco_concatenation* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_concatenation_storing_partial_derivatives(aco_concatenation* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
    if(a->input2 != NULL){
        a->input2_error = storing_params+a->input2_array_offset;
    }
}

void print_aco_concatenation_offsets(aco_concatenation* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
    if(f->input2 != NULL)
        printf("input2 array offset: %lud\n",f->input2_array_offset);
}

void aco_concatenation_ff(float input,float* output, int output_size){
    set_vector_with_value(input,output,output_size);
}

void aco_concatenation_feed_forward(aco_concatenation* f){
    if(f == NULL)
        return;
    if(f->input2 == NULL)
        aco_concatenation_ff(f->value,f->output+f->input_size,f->output_size-f->input_size);
}

void aco_concatenation_back_propagation(aco_concatenation* f){
    return;
}

float* get_aco_concatenation_d_output(aco_concatenation* concatenation){
    if(concatenation == NULL)
        return NULL;
    return concatenation->output_error;
}

void set_aco_concatenation_used_output(aco_concatenation* a,int* previous_used_outputs1,int* previous_used_outputs2, int is_root, uint64_t offset1, uint64_t offset2){
    if(a == NULL)
        return;
    if(is_root)
        set_int_vector_with_value(1,a->output_used,a->output_size);
    else{
        
        if(a->input2 == NULL){
            copy_int_array(previous_used_outputs1,a->output_used,a->input_size);
            set_int_vector_with_value(1,a->output_used+a->input_size,a->output_size-a->input_size);
        }
        else{
            if(offset1 == a->input_array_offset){
                copy_int_array(previous_used_outputs1,a->output_used,a->input_size);
                copy_int_array(previous_used_outputs2,a->output_used+a->input_size,a->output_size-a->input_size);
            }
            else{
                copy_int_array(previous_used_outputs2,a->output_used,a->input_size);
                copy_int_array(previous_used_outputs1,a->output_used+a->input_size,a->output_size-a->input_size);
            }
        }
    }
}

int* get_aco_concatenation_used_outputs(aco_concatenation* a){
    if(a == NULL)
        return NULL;
    return a->output_used;
}
