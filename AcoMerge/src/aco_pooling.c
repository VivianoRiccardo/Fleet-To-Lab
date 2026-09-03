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


aco_pooling* init_aco_pooling(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int identifier, int pooling_type,
                              uint64_t  output_array_offset, uint64_t  input_array_offset, float* input, float* output){
    aco_pooling* p = (aco_pooling*)malloc(sizeof(aco_pooling));
    p->channels = channels;
    p->pooling_rows = pooling_rows;
    p->stride_rows = stride_rows;
    p->pooling_cols = pooling_cols;
    p->stride_cols = stride_cols;
    p->identifier = identifier;
    p->pooling_type = pooling_type;
    p->input_array_offset = input_array_offset;
    p->output_array_offset = output_array_offset;
    p->input_rows = input_rows;
    p->input_cols = input_cols;
    p->input = input;
    p->output = output;
    p->padding_rows = padding_rows;
    p->padding_cols = padding_cols;
    p->input_error = NULL;
    p->output_error = NULL;
    p->output_used = NULL;
    return p;
}

aco_pooling* copy_aco_pooling(aco_pooling* a){
    if(a == NULL)
        return NULL;
    aco_pooling* n =(aco_pooling*)malloc(sizeof(aco_pooling));
    n->channels = a->channels;
    n->pooling_rows = a->pooling_rows;
    n->stride_rows = a->stride_rows;
    n->pooling_cols = a->pooling_cols;
    n->stride_cols = a->stride_cols;
    n->identifier = a->identifier;
    n->pooling_type = a->pooling_type;
    n->input_array_offset = a->input_array_offset;
    n->output_array_offset = a->output_array_offset;
    n->input_rows = a->input_rows;
    n->input_cols = a->input_cols;
    n->input = a->input;
    n->output = a->output;
    n->padding_rows = a->padding_rows;
    n->padding_cols = a->padding_cols;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_pooling_storing_params(aco_pooling* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

void assign_aco_pooling_storing_params_output_used(aco_pooling* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_pooling_storing_partial_derivatives(aco_pooling* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
}

int aco_pooling_input_size(int channels, int input_rows, int input_cols){
    return channels*input_rows*input_cols;
}

int aco_pooling_output_rows(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    return aco_cl_output_rows(channels,input_rows,input_cols,pooling_rows,pooling_cols,stride_rows,stride_cols,padding_rows,padding_cols);
} 

int aco_pooling_output_cols(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    return aco_cl_output_cols(channels,input_rows,input_cols,pooling_rows,pooling_cols,stride_rows,stride_cols,padding_rows,padding_cols);
}

int aco_pooling_output_size(int channels, int input_rows, int input_cols, int pooling_rows, int pooling_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    return aco_cl_output_size(channels,input_rows,input_cols,pooling_rows,pooling_cols,stride_rows,stride_cols,padding_rows,padding_cols);
}

// TO TEST
void max_pooling_feed_forward(float* input, float* output, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols){
    int c,i,j,k1,k2, first, second, third, fourth, fifth,channel_input, channel_output, output_i, output_j;
    if((input_i-sub_pool_i)%stride1 == 0)
        output_i = ((int)((input_i-sub_pool_i)/stride1)) + 1 + 2*padding_rows;
    else
        output_i = ((int)((input_i-sub_pool_i+((input_i-sub_pool_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-sub_pool_j)%stride2 == 0)
        output_j = ((int)((input_j-sub_pool_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-sub_pool_j+((input_j-sub_pool_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int output_i_minus = output_i-padding_rows;
    int output_j_minus = output_j-padding_cols;
    float max = -999999;
    for(c = 0; c < channels; c++){
        channel_input = c*input_i*input_j;
        channel_output = c*output_i*output_j;
        for(i = padding_rows; i < output_i_minus; i++){
            first = i*stride1*input_j+channel_input;
            fourth = (i-padding_rows)*stride1;
            fifth = i*output_j+channel_output;
            for(j = padding_cols; j < output_j_minus; j++){
                third = (j-padding_cols)*stride2;
                for(k1 = 0; k1 < sub_pool_i && k1 + fourth < input_i; k1++){
                    second = input_j*k1+first+third;
                    for(k2 = 0; k2 < sub_pool_j && k2 + third < input_j; k2++){
                        if(input[second + k2] > max)
                            max = input[second + k2];
                    }
                }
                output[fifth+j] = max;
                max = -999999;        
            }
        }
    }
}
void max_pooling_back_propagation(float* input,float* input_error, float* output,float* output_error, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols){
    int c,i,j,k1,k2, first, second, third, fourth, fifth,channel_input, channel_output, output_i, output_j;
    if((input_i-sub_pool_i)%stride1 == 0)
        output_i = ((int)((input_i-sub_pool_i)/stride1)) + 1 + 2*padding_rows;
    else
        output_i = ((int)((input_i-sub_pool_i+((input_i-sub_pool_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-sub_pool_j)%stride2 == 0)
        output_j = ((int)((input_j-sub_pool_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-sub_pool_j+((input_j-sub_pool_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int output_i_minus = output_i-padding_rows;
    int output_j_minus = output_j-padding_cols;
    float max = -999999;
    for(c = 0; c < channels; c++){
        channel_input = c*input_i*input_j;
        channel_output = c*output_i*output_j;
        for(i = padding_rows; i < output_i_minus; i++){
            first = i*stride1*input_j+channel_input;
            fourth = (i-padding_rows)*stride1;
            fifth = i*output_j+channel_output;
            for(j = padding_cols; j < output_j_minus; j++){
                third = (j-padding_cols)*stride2;
                int index = 0;
                for(k1 = 0; k1 < sub_pool_i && k1 + fourth < input_i; k1++){
                    second = input_j*k1+first+third;
                    for(k2 = 0; k2 < sub_pool_j && k2 + third < input_j; k2++){
                        if(input[second + k2] > max){
                            max = input[second + k2];
                            index = second + k2;
                        }
                    }
                }
                input_error[index] += output[fifth+j];
                max = -999999;        
            }
        }
    }
}


// TO TEST
void average_pooling_feed_forward(float* input, float* output, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols){
    int c,i,j,k1,k2, first, second, third, fourth, fifth,channel_input, channel_output, output_i, output_j;
    if((input_i-sub_pool_i)%stride1 == 0)
        output_i = ((int)((input_i-sub_pool_i)/stride1)) + 1+2*padding_rows;
    else
        output_i = ((int)((input_i-sub_pool_i+((input_i-sub_pool_i)%stride1))/stride1)) + 1+2*padding_rows;
    if((input_j-sub_pool_j)%stride2 == 0)
        output_j = ((int)((input_j-sub_pool_j)/stride2)) + 1+2*padding_cols;
    else    
        output_j = ((int)((input_j-sub_pool_j+((input_j-sub_pool_j)%stride2))/stride2)) + 1+2*padding_cols;
    double sum = 0;
    double sum2 = 0;
    int output_i_minus = output_i-padding_rows;
    int output_j_minus = output_j-padding_cols;
    for(c = 0; c < channels; c++){
        channel_input = c*input_i*input_j;
        channel_output = c*output_i*output_j;
        for(i = padding_rows; i < output_i_minus; i++){
            first = i*stride1*input_j+channel_input;
            fourth = (i-padding_rows)*stride1;
            fifth = i*output_j+channel_output;
            for(j = padding_cols; j < output_j_minus; j++){
                third = (j-padding_cols)*stride2;
                for(k1 = 0; k1 < sub_pool_i && k1 + fourth < input_i; k1++){
                    second = input_j*k1+first+third;
                    for(k2 = 0; k2 < sub_pool_j && k2 + third < input_j; k2++){                   
                        sum += input[second + k2];
                        sum2++;
                    }
                }
                output[fifth+j] += sum/sum2;
                sum = 0;
                sum2 = 0;        
            }
        }
    }
}

void average_pooling_back_propagation(float* input,float* input_error, float* output,float* output_error, int channels, int input_i, int input_j, int sub_pool_i, int sub_pool_j, int stride1, int stride2, int padding_rows, int padding_cols){
    int c,i,j,k1,k2, first, second, third, fourth, fifth,channel_input, channel_output, output_i, output_j;
    if((input_i-sub_pool_i)%stride1 == 0)
        output_i = ((int)((input_i-sub_pool_i)/stride1)) + 1+2*padding_rows;
    else
        output_i = ((int)((input_i-sub_pool_i+((input_i-sub_pool_i)%stride1))/stride1)) + 1+2*padding_rows;
    if((input_j-sub_pool_j)%stride2 == 0)
        output_j = ((int)((input_j-sub_pool_j)/stride2)) + 1+2*padding_cols;
    else    
        output_j = ((int)((input_j-sub_pool_j+((input_j-sub_pool_j)%stride2))/stride2)) + 1+2*padding_cols;
    double sum = 0;
    double sum2 = 0;
    int output_i_minus = output_i-padding_rows;
    int output_j_minus = output_j-padding_cols;
    for(c = 0; c < channels; c++){
        channel_input = c*input_i*input_j;
        channel_output = c*output_i*output_j;
        for(i = padding_rows; i < output_i_minus; i++){
            first = i*stride1*input_j+channel_input;
            fourth = (i-padding_rows)*stride1;
            fifth = i*output_j+channel_output;
            for(j = padding_cols; j < output_j_minus; j++){
                third = (j-padding_cols)*stride2;
                for(k1 = 0; k1 < sub_pool_i && k1 + fourth < input_i; k1++){
                    second = input_j*k1+first+third;
                    for(k2 = 0; k2 < sub_pool_j && k2 + third < input_j; k2++){                   
                        sum2++;
                    }
                }
                for(k1 = 0; k1 < sub_pool_i && k1 + fourth < input_i; k1++){
                    second = input_j*k1+first+third;
                    for(k2 = 0; k2 < sub_pool_j && k2 + third < input_j; k2++){ 
                        input_error[second + k2]+=output_error[fifth+j]/sum2;
                    }
                }
                sum2 = 0;        
            }
        }
    }
}

void aco_pooling_feed_forward(aco_pooling* pooling){
    if(pooling == NULL)
        return;
    if(pooling->pooling_type == MAX_POOLING)
        max_pooling_feed_forward(pooling->input, pooling->output, pooling->channels, pooling->input_rows, pooling->input_cols, pooling->pooling_rows, pooling->pooling_cols, pooling->stride_rows, pooling->stride_cols,pooling->padding_rows,pooling->padding_cols);
    else
        average_pooling_feed_forward(pooling->input, pooling->output, pooling->channels, pooling->input_rows, pooling->input_cols, pooling->pooling_rows, pooling->pooling_cols, pooling->stride_rows, pooling->stride_cols,pooling->padding_rows,pooling->padding_cols);

}

void aco_pooling_back_propagation(aco_pooling* pooling){
    if(pooling == NULL)
        return;
    if(pooling->pooling_type == MAX_POOLING)
        max_pooling_back_propagation(pooling->input,pooling->input_error, pooling->output, pooling->output_error,pooling->channels, pooling->input_rows, pooling->input_cols, pooling->pooling_rows, pooling->pooling_cols, pooling->stride_rows, pooling->stride_cols,pooling->padding_rows,pooling->padding_cols);
    else
        average_pooling_back_propagation(pooling->input, pooling->input_error, pooling->output,pooling->output_error, pooling->channels, pooling->input_rows, pooling->input_cols, pooling->pooling_rows, pooling->pooling_cols, pooling->stride_rows, pooling->stride_cols,pooling->padding_rows,pooling->padding_cols);

}

void print_aco_pooling_offsets(aco_pooling* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
}

float* get_aco_pooling_d_output(aco_pooling* pooling){
	if(pooling == NULL)
		return NULL;
	return pooling->output_error;
}

void set_aco_pooling_used_output(aco_pooling* a,int* previous_used_outputs, int is_root){
	if(a == NULL)
		return;
	if(is_root)
		set_int_vector_with_value(1,a->output_used,aco_pooling_output_size(a->channels,a->input_rows,a->input_cols,a->pooling_rows,a->pooling_cols,a->stride_rows,a->stride_cols,a->padding_rows,a->padding_cols));
	else
		copy_int_array(previous_used_outputs,a->output_used,aco_pooling_output_size(a->channels,a->input_rows,a->input_cols,a->pooling_rows,a->pooling_cols,a->stride_rows,a->stride_cols,a->padding_rows,a->padding_cols));
}

int* get_aco_pooling_used_outputs(aco_pooling* a){
	if(a == NULL)
		return NULL;
	return a->output_used;
}

