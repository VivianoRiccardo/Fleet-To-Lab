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


aco_sum* init_aco_sum(int input_size, int input2_size, int output_size,int identifier,int flag, uint64_t  output_array_offset,
                      uint64_t  input_array_offset,uint64_t  input2_array_offset, float* output, float* input, float* input2){
    
    
    aco_sum* f = (aco_sum*)malloc(sizeof(aco_sum));
    f->input_size = input_size;
    f->input2_size = input2_size;
    f->output_size = output_size;
    f->identifier = identifier;
    f->flag = flag;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->input2_array_offset = input2_array_offset;
    f->output = output;
    f->input = input;
    f->input2 = input2;
    f->input_error = NULL;
    f->output_error = NULL;
    f->input2_error = NULL;
    f->output_used = NULL;
    return f;
}

aco_sum* copy_aco_sum(aco_sum* a){
    if(a == NULL)    
        return NULL;
    aco_sum* n = (aco_sum*)malloc(sizeof(aco_sum));
    n->input_size = a->input_size;
    n->input2_size = a->input2_size;
    n->output_size = a->output_size;
    n->identifier = a->identifier;
    n->flag = a->flag;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->input2_array_offset = a->input2_array_offset;
    n->output = a->output;
    n->input = a->input;
    n->input2 = a->input2;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_sum_storing_params(aco_sum* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
    a->input2 = storing_params+a->input2_array_offset;
}

void assign_aco_sum_storing_params_output_used(aco_sum* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_sum_storing_partial_derivatives(aco_sum* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
    a->input2_error = storing_params+a->input2_array_offset;
}

void aco_sum_ff(float* input1, float* input2,float* output, int input_size1, int input_size2, int output_size, int flag){
    int min = input_size1;
    if(input_size2 < min)
        min = input_size2;
    if(output_size < min)
        min = output_size;
    if(flag == SUM_FLAG)
        sum_vectors(input1,input2,output,min);
    else if(flag == MUL_FLAG)
        dot1D(input1,input2,output,min);
    /*
    else if(flag == DIV_FLAG)
        div1D(input1,input2,output,min);
    else if(flag == DIF_FLAG)
        dif1D(input1,input2,output,min);
    *///use activation to invert or to multiply by -1 and then do the aco_sum
    else if(flag == EXP_FLAG)
        exp1D(input1,input2,output,min);
}

void aco_sum_ff_edge_popup(float* input1, float* input2,float* output, int input_size1, int input_size2, int output_size, int flag, int* used_outputs){
    int min = input_size1;
    if(input_size2 < min)
        min = input_size2;
    if(output_size < min)
        min = output_size;
    if(flag == SUM_FLAG)
        sum_vectors(input1,input2,output,min);
    else if(flag == MUL_FLAG)
        dot1D(input1,input2,output,min);
    /*
    else if(flag == DIV_FLAG)
        div1D(input1,input2,output,min);
    else if(flag == DIF_FLAG)
        dif1D(input1,input2,output,min);
    
    else if(flag == EXP_FLAG)
        exp1D_with_flags(input1,input2,output,used_outputs, min);
    *///use activation to invert or to multiply by -1 and then do the aco_sum
}

void aco_sum_bp(float* input1,float* input1_error, float* input2,float* input2_error, float* output,float* output_error, int input_size1, int input_size2, int output_size, int flag, int* used_outputs){
    int min = input_size1;
    if(input_size2 < min)
        min = input_size2;
    if(output_size < min)
        min = output_size;
    if(flag == SUM_FLAG){
        sum_vectors(output_error,input1_error,input1_error,min);
        sum_vectors(output_error,input2_error,input2_error,min);
    }
    else if(flag == MUL_FLAG){
        int i;
        for(i = 0; i < min; i++){
            input1_error[i]+=output_error[i]*input2[i]*((float)used_outputs[i]);
            input2_error[i]+=output_error[i]*input1[i]*((float)used_outputs[i]);
        }
    }
}

void aco_sum_feed_forward(aco_sum* sum){
    if(sum == NULL)
        return;
    aco_sum_ff(sum->input, sum->input2,sum->output, sum->input_size, sum->input2_size, sum->output_size, sum->flag);
}

void aco_sum_feed_forward_edge_popup(aco_sum* sum){
    if(sum == NULL)
        return;
    aco_sum_ff_edge_popup(sum->input, sum->input2,sum->output, sum->input_size, sum->input2_size, sum->output_size, sum->flag, sum->output_used);
}

void aco_sum_back_propagation(aco_sum* sum){
    if(sum == NULL)
        return;
    aco_sum_bp(sum->input,sum->input_error, sum->input2,sum->input2_error,sum->output,sum->output_error, sum->input_size, sum->input2_size, sum->output_size, sum->flag, sum->output_used);
}

void print_aco_sum_offsets(aco_sum* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("input2 array offset: %lud\n",f->input2_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
}

float* get_aco_sum_d_output(aco_sum* sum){
    if(sum == NULL)
        return NULL;
    sum->output_error;
}

void set_aco_sum_used_output(aco_sum* a,int* previous_used_outputs1,int* previous_used_outputs2, int is_root){
    if(a == NULL)
        return;
    if(is_root)
        set_int_vector_with_value(1,a->output_used,a->output_size);
    else{
        int min = a->input_size;
        if(a->input2_size < min)
            min = a->input2_size;
        if(a->output_size < min)
            min = a->output_size;
        int i;
        for(i = 0; i < min; i++){
            if(a->flag == SUM_FLAG || a->flag == DIF_FLAG || a->flag == EXP_FLAG ){
                a->output_used[i] = ((previous_used_outputs1[i]||previous_used_outputs2[i]))?1:0;
            }
            else if(a->flag == MUL_FLAG || a->flag == DIV_FLAG){
                a->output_used[i] = ((previous_used_outputs1[i] && previous_used_outputs2[i]))?1:0;
            }
        }
    }
        
}

int* get_aco_sum_used_outputs(aco_sum* a){
    if(a == NULL)
        return NULL;
    return a->output_used;
}



