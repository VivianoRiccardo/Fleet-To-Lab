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


uint64_t get_aco_cl_weights_or_pheromone_weights_size(int channels, int kernel_rows, int kernel_cols, int n_kernels){
    return channels*kernel_rows*kernel_cols*n_kernels;
}

uint64_t get_aco_cl_biases_or_pheromone_biases_size(int n_kernels){
    return n_kernels;
}

int aco_cl_input_size(int channels, int input_rows, int input_cols){
    return channels*input_rows*input_cols;
}

int aco_cl_output_size(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    int output_i,output_j;
    
    if((input_rows-kernel_rows)%stride_rows == 0)
        output_i = ((int)((input_rows-kernel_rows)/stride_rows)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_rows-kernel_rows+((input_rows-kernel_rows)%stride_rows))/stride_rows)) + 1 + 2*padding_rows;
    if((input_cols-kernel_cols)%stride_cols == 0)
        output_j = ((int)((input_cols-kernel_cols)/stride_cols)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_cols-kernel_cols+((input_cols-kernel_cols)%stride_cols))/stride_cols)) + 1 + 2*padding_cols;
    return output_i*output_j*n_kernels;
}

int aco_cl_output_rows(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    int output_i;
    
    if((input_rows-kernel_rows)%stride_rows == 0)
        output_i = ((int)((input_rows-kernel_rows)/stride_rows)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_rows-kernel_rows+((input_rows-kernel_rows)%stride_rows))/stride_rows)) + 1 + 2*padding_rows;
    
    return output_i;
}

int aco_cl_output_cols(int n_kernels, int input_rows, int input_cols, int kernel_rows, int kernel_cols, int stride_rows, int stride_cols, int padding_rows, int padding_cols){
    int output_j;
    
    if((input_cols-kernel_cols)%stride_cols == 0)
        output_j = ((int)((input_cols-kernel_cols)/stride_cols)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_cols-kernel_cols+((input_cols-kernel_cols)%stride_cols))/stride_cols)) + 1 + 2*padding_cols;
    
    return output_j;
}

void init_aco_cl_weights_xavier_init(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights){
    int i, size = n_kernels*channels*kernel_rows*kernel_cols;
    for(i = 0; i < size; i++){
        weights[i] = random_general_gaussian_xavier_init((float)(channels*input_rows*input_cols));
    }
}

void init_aco_cl_weights_kaiming_constant_init(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights){
    int i, size = n_kernels*channels*kernel_rows*kernel_cols;
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(channels*input_rows*input_cols));
    }
}

void init_aco_cl_weights_signed_kaiming_constant_init_parameters(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, int n_parameters){
    int i, size = n_kernels*channels*kernel_rows*kernel_cols*n_parameters;
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(channels*input_rows*input_cols));
    }
}

void init_aco_cl_weights_chaos_init_parameters(int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, int n_parameters){
    int i, size = n_kernels*channels*kernel_rows*kernel_cols,j;
    for(i = 0; i < size; i++){
        float a = normalized_random();
        for(j = 0; j < n_parameters; j++){
            weights[i*n_parameters+j] = a;
            a = chaos_next_sequence(a);
        }
    }
}

void init_aco_cl_weights_with_value(int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights, float pheromone_value){
    set_vector_with_value(pheromone_value,weights,n_kernels*channels*kernel_rows*kernel_cols);
}


void init_aco_cl_biases_zeros(int n_kernels, float* biases){
    set_vector_with_value(0,biases,n_kernels);
}

void init_aco_cl_biases_normalized_random(int n_kernels, float* biases){
    int i;
    for(i = 0; i < n_kernels; i++){
        biases[i] = signed_normalized_random(1.0);
    }
}

void init_aco_cl_biases_signed_kaiming_constant(int input_rows, int input_cols, int channels,int n_kernels, float* biases){
    int i;
    for(i = 0; i < n_kernels; i++){
        biases[i] = signed_normalized_random((float)(channels*input_rows*input_cols));
    }
}


void init_aco_cl_biases_with_value(int n_kernels, float* biases, float pheromone_value){
    set_vector_with_value(pheromone_value,biases,n_kernels);
}

void init_aco_cl_weights_pheromones(int channels, int kernel_rows, int kernel_cols, int n_kernels, float* weights_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,weights_pheromone,n_kernels*channels*kernel_rows*kernel_cols);
}

void init_aco_cl_biases_pheromones(int n_kernels, float* biases_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,biases_pheromone,n_kernels);
}

aco_cl* init_aco_cl(int padding_rows, int padding_cols, int input_rows, int input_cols, int channels, int kernel_rows, int kernel_cols, int n_kernels, int identifier, int stride_rows, int stride_cols, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input){
    
    
    aco_cl* f = (aco_cl*)malloc(sizeof(aco_cl));
    f->k_percentage = 1;
    f->channels = channels;
    f->kernel_rows = kernel_rows;
    f->kernel_cols = kernel_cols;
    f->n_kernels = n_kernels;
    f->stride_rows = stride_rows;
    f->stride_cols = stride_cols;
    f->identifier = identifier;
    f->weights = weights;
    f->biases = biases;
    f->pheromone_weights = pheromone_weights;
    f->pheromone_biases = pheromone_biases;
    f->output = output;
    f->input = input;
    f->weights_array_offset = weights_array_offset;
    f->biases_array_offset = biases_array_offset;
    f->pheromone_weights_array_offset = pheromone_weights_array_offset;
    f->pheromone_biases_array_offset = pheromone_biases_array_offset;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->input_rows = input_rows;
    f->input_cols = input_cols;
    f->padding_rows = padding_rows;
    f->padding_cols = padding_cols;
    f->input_error = NULL;
    f->output_error = NULL;
    f->d_weights = NULL;
    f->d_biases = NULL;
    f->all_weights = NULL;
    f->all_biases = NULL;
    f->indices = NULL;
    f->output_used = NULL;
    return f;
}

aco_cl* copy_aco_cl(aco_cl* a){
    if(a == NULL)
        return NULL;
    aco_cl* n = (aco_cl*)malloc(sizeof(aco_cl));
    n->indices = a->indices;
    n->k_percentage = a->k_percentage;
    n->channels = a->channels;
    n->kernel_rows = a->kernel_rows;
    n->kernel_cols = a->kernel_cols;
    n->n_kernels = a->n_kernels;
    n->stride_rows = a->stride_rows;
    n->stride_cols = a->stride_cols;
    n->identifier = a->identifier;
    n->weights = a->weights;
    n->biases = a->biases;
    n->pheromone_weights = a->pheromone_weights;
    n->pheromone_biases = a->pheromone_biases;
    n->output = a->output;
    n->input = a->input;
    n->weights_array_offset = a->weights_array_offset;
    n->biases_array_offset = a->biases_array_offset;
    n->pheromone_weights_array_offset = a->pheromone_weights_array_offset;
    n->pheromone_biases_array_offset = a->pheromone_biases_array_offset;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->input_rows = a->input_rows;
    n->input_cols = a->input_cols;
    n->padding_rows = a->padding_rows;
    n->padding_cols = a->padding_cols;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_cl_storing_params(aco_cl* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

void assign_aco_cl_storing_params_output_used(aco_cl* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_cl_storing_partial_derivatives(aco_cl* a, float* d_storing_params){
    if(a == NULL)    
        return;
    a->output_error = d_storing_params+a->output_array_offset;
    a->input_error = d_storing_params+a->input_array_offset;
}

void assign_aco_cl_learnable_params(aco_cl* a, float* params){
    if(a == NULL)
        return;
    a->weights = params+a->weights_array_offset;
    a->biases = params+a->biases_array_offset;
}

void assign_aco_cl_learnable_params_scores(aco_cl* a, float* params, int* indices, float* scores){
    if(a == NULL)
        return;
    a->d_weights = params+a->weights_array_offset;
    a->scores = scores+a->weights_array_offset;
    a->indices = indices+a->weights_array_offset;
}

void print_aco_cl_offsets(aco_cl* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
}

void set_cl_k_percentage(aco_cl* c, float k){
	if(c == NULL)
		return;
	if(k < 0 || k > 1)
		return;
	c->k_percentage = k;
}


void convolutional_feed_forward(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output, int stride1, int stride2, int padding_rows, int padding_cols){
    int nk,oi,oj,i = 0,j = 0,c,temp_output1, temp_output2, input2_channel,channel_kernel, channel2_kernel, row_kernel, input_channel, ojstride, oistride , ostride, nkk1, nkk2, oistr, output_i, output_j;
    
    if((input_i-kernel_i)%stride1 == 0)
        output_i = ((int)((input_i-kernel_i)/stride1)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_i-kernel_i+((input_i-kernel_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-kernel_j)%stride2 == 0)
        output_j = ((int)((input_j-kernel_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-kernel_j+((input_j-kernel_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int output_i_minus_stride = output_i-padding_rows;
    int output_j_minus_stride = output_j-padding_cols;
    for(nk = 0; nk < n_kernels; nk++){
        nkk1 = output_i*output_j*nk;
        nkk2 = channels*kernel_i*kernel_j*nk;
        for(oi = padding_rows; oi < output_i_minus_stride; oi++){
            temp_output1 = nkk1+oi*output_j;
            oistr = (oi-padding_rows)*stride1;
            oistride = oistr*input_j;
            for(oj = padding_cols; oj < output_j_minus_stride; oj++){
                temp_output2 = temp_output1+oj;
                ojstride = (oj-padding_cols)*stride2;
                ostride = oistride+ojstride;
                for(c = 0; c < channels; c++){
                    channel_kernel = nkk2+c*kernel_i*kernel_j;
                    input_channel = c*input_i*input_j + ostride;
                    for(i = 0; i < kernel_i && i + oistr < input_i; i++){
                        channel2_kernel = channel_kernel+i*kernel_j;
                        input2_channel = input_channel+i*input_j;
                        for(j = 0; j < kernel_j && j + ojstride < input_j; j++){
                            output[temp_output2] += kernel[channel2_kernel+j]*input[input2_channel + j];
                        }
                    }
                }
                //output[temp_output2] += biases[nk];    
            }
        }
    }
}

void convolutional_set_k_percentage(aco_cl* a, float k_percentage){
    if(k_percentage > 1 || k_percentage < 0){
        fprintf(stderr,"Error, k percentage must be in [0,1]\n");
        exit(1);
    }
    if(a == NULL)
        return;
    a->k_percentage = k_percentage;
}

void convolutional_feed_forward_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output, int stride1, int stride2, int padding_rows, int padding_cols, int* indices, float k_percentage){
    int oi,oj,i,j,c,s,z, output_i, output_j;    
    if((input_i-kernel_i)%stride1 == 0)
        output_i = ((int)((input_i-kernel_i)/stride1)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_i-kernel_i+((input_i-kernel_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-kernel_j)%stride2 == 0)
        output_j = ((int)((input_j-kernel_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-kernel_j+((input_j-kernel_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int last_n = n_kernels*channels*kernel_i*kernel_j*k_percentage;
    int lv_minus1 = n_kernels*channels*kernel_i*kernel_j-last_n;
    int lv0 = n_kernels*channels*kernel_i*kernel_j;
    int lv1 = channels*kernel_i*kernel_j;
    int lv2 = kernel_i*kernel_j;
    int ou_lv1 = output_i*output_j;
    int in_lv1 = input_i*input_j;
    for(oi = padding_rows; oi < output_i-padding_rows; oi++){
        for(oj = padding_cols; oj < output_j-padding_cols; oj++){
            for(s = lv_minus1; s < lv0; s++){
                int kernel_temp = ((int)(indices[s]/(lv1)));
                int rest_temp = ((indices[s]%(lv1)));
                int c_temp = (int)(rest_temp/lv2);
                int i_temp = (int)((rest_temp%lv2)/kernel_j);
                int j_temp = ((rest_temp%lv2)%kernel_j);
                output[kernel_temp*ou_lv1+oi*output_j+oj] += kernel[indices[s]]*input[c_temp*input_i*input_j + i_temp*input_j + j_temp+(oj-padding_cols)*stride2+(oi-padding_rows)*stride1*input_j];
            }
        }
    }
}

void convolutional_back_propagation_edge_popup(float* input, float* kernel, int input_i, int input_j, int kernel_i, int kernel_j, float* biases, int channels, int n_kernels, float* output_error, int stride1, int stride2, int padding_rows, int padding_cols, float* input_error, float* kernel_error, float k_percentage, int* indices){
    int s,nk,oi,oj,i = 0,j = 0,c,temp_output1, temp_output2, input2_channel,channel_kernel, channel2_kernel, row_kernel, input_channel, ojstride, oistride , ostride, nkk1, nkk2, oistr, output_i, output_j;
    
    if((input_i-kernel_i)%stride1 == 0)
        output_i = ((int)((input_i-kernel_i)/stride1)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_i-kernel_i+((input_i-kernel_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-kernel_j)%stride2 == 0)
        output_j = ((int)((input_j-kernel_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-kernel_j+((input_j-kernel_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int output_i_minus_stride = output_i-padding_rows;
    int output_j_minus_stride = output_j-padding_cols;
    for(nk = 0; nk < n_kernels; nk++){
        nkk1 = output_i*output_j*nk;
        nkk2 = channels*kernel_i*kernel_j*nk;
        for(oi = padding_rows; oi < output_i_minus_stride; oi++){
            temp_output1 = nkk1+oi*output_j;
            oistr = (oi-padding_rows)*stride1;
            oistride = oistr*input_j;
            for(oj = padding_cols; oj < output_j_minus_stride; oj++){
                temp_output2 = temp_output1+oj;
                ojstride = (oj-padding_cols)*stride2;
                ostride = oistride+ojstride;
                for(c = 0; c < channels; c++){
                    channel_kernel = nkk2+c*kernel_i*kernel_j;
                    input_channel = c*input_i*input_j + ostride;
                    for(i = 0; i < kernel_i && i + oistr < input_i; i++){
                        channel2_kernel = channel_kernel+i*kernel_j;
                        input2_channel = input_channel+i*input_j;
                        for(j = 0; j < kernel_j && j + ojstride < input_j; j++){
                            kernel_error[channel2_kernel+j] += input[input2_channel + j]*kernel[channel2_kernel+j]*output_error[temp_output2];
                        }
                    }
                }
                //output[temp_output2] += biases[nk];    
            }
        }
    }
    
    
    int last_n = n_kernels*channels*kernel_i*kernel_j*k_percentage;
    int lv_minus1 = n_kernels*channels*kernel_i*kernel_j-last_n;
    int lv0 = n_kernels*channels*kernel_i*kernel_j;
    int lv1 = channels*kernel_i*kernel_j;
    int lv2 = kernel_i*kernel_j;
    int ou_lv1 = output_i*output_j;
    int in_lv1 = input_i*input_j;
    for(oi = padding_rows; oi < output_i-padding_rows; oi++){
        for(oj = padding_cols; oj < output_j-padding_cols; oj++){
            for(s = lv_minus1; s < lv0; s++){
                int kernel_temp = ((int)(indices[s]/(lv1)));
                int rest_temp = ((indices[s]%(lv1)));
                int c_temp = (int)(rest_temp/lv2);
                int i_temp = (int)((rest_temp%lv2)/kernel_j);
                int j_temp = ((rest_temp%lv2)%kernel_j);
                input_error[c_temp*input_i*input_j + i_temp*input_j + j_temp+(oj-padding_cols)*stride2+(oi-padding_rows)*stride1*input_j]+=output_error[kernel_temp*ou_lv1+oi*output_j+oj]*kernel[indices[s]];
               
            }
        }
    }
}



void aco_cl_feed_forward(aco_cl* cl){
    if(cl == NULL)
        return;
    convolutional_feed_forward(cl->input, cl->weights, cl->input_rows, cl->input_cols, cl->kernel_rows, cl->kernel_cols, cl->biases, cl->channels, cl->n_kernels, cl->output, cl->stride_rows, cl->stride_cols,cl->padding_rows,cl->padding_cols);
}


void aco_cl_feed_forward_edge_popup(aco_cl* cl){
    if(cl == NULL)
        return;
    convolutional_feed_forward_edge_popup(cl->input, cl->weights, cl->input_rows, cl->input_cols, cl->kernel_rows, cl->kernel_cols, cl->biases, cl->channels, cl->n_kernels, cl->output, cl->stride_rows, cl->stride_cols,cl->padding_rows,cl->padding_cols, cl->indices, cl->k_percentage);
}

void aco_cl_back_propagation_edge_popup(aco_cl* cl){
    if(cl == NULL)
        return;
    convolutional_back_propagation_edge_popup(cl->input, cl->weights, cl->input_rows, cl->input_cols, cl->kernel_rows, cl->kernel_cols, cl->biases, cl->channels, cl->n_kernels, cl->output_error, cl->stride_rows, cl->stride_cols,cl->padding_rows,cl->padding_cols, cl->input_error,cl->d_weights, cl->k_percentage, cl->indices);
}

void merge_sort_aco_cl(aco_cl* c){
	if(c == NULL)
		return;
	merge_sort(c->scores,c->indices,0,c->channels*c->kernel_rows*c->kernel_cols*c->n_kernels-1);
}

float* get_aco_cl_d_output(aco_cl* cl){
	if(cl == NULL)
		return NULL;
	return cl->output_error;
}

void init_cl_scores(aco_cl* cl){
	if(cl == NULL)
		return;
	int i;
	for(i = 0; i < cl->n_kernels*cl->channels*cl->kernel_rows*cl->kernel_cols; i++){
		cl->indices[i] = i;
	}
}

void paste_cl_indices_weights_according_to_scores(aco_cl* cl, float* params){
	if(cl == NULL || params == NULL)
		return;
	int i,p = cl->n_kernels*cl->channels*cl->kernel_rows*cl->kernel_cols, last_n = p-p*cl->k_percentage;
	for(i = last_n; i < p; i++){
		params[cl->weights_array_offset+cl->indices[i]] = 1;
	}
}


void convolutional_used_output(int input_i, int input_j, int kernel_i, int kernel_j, int channels, int n_kernels, int* output_used, int stride1, int stride2, int padding_rows, int padding_cols, int* indices, float k_percentage){
    int oi,oj,i,j,c,s,z, output_i, output_j;    
    if((input_i-kernel_i)%stride1 == 0)
        output_i = ((int)((input_i-kernel_i)/stride1)) + 1 + 2*padding_rows;
    else    
        output_i = ((int)((input_i-kernel_i+((input_i-kernel_i)%stride1))/stride1)) + 1 + 2*padding_rows;
    if((input_j-kernel_j)%stride2 == 0)
        output_j = ((int)((input_j-kernel_j)/stride2)) + 1 + 2*padding_cols;
    else    
        output_j = ((int)((input_j-kernel_j+((input_j-kernel_j)%stride2))/stride2)) + 1 + 2*padding_cols;
    int last_n = n_kernels*channels*kernel_i*kernel_j*k_percentage;
    int lv_minus1 = n_kernels*channels*kernel_i*kernel_j-last_n;
    int lv0 = n_kernels*channels*kernel_i*kernel_j;
    int lv1 = channels*kernel_i*kernel_j;
    int lv2 = kernel_i*kernel_j;
    int ou_lv1 = output_i*output_j;
    int in_lv1 = input_i*input_j;
    for(oi = padding_rows; oi < output_i-padding_rows; oi++){
        for(oj = padding_cols; oj < output_j-padding_cols; oj++){
            for(s = lv_minus1; s < lv0; s++){
                int kernel_temp = ((int)(indices[s]/(lv1)));
                int rest_temp = ((indices[s]%(lv1)));
                int c_temp = (int)(rest_temp/lv2);
                int i_temp = (int)((rest_temp%lv2)/kernel_j);
                int j_temp = ((rest_temp%lv2)%kernel_j);
                output_used[kernel_temp*ou_lv1+oi*output_j+oj] = 1;
            }
        }
    }
}


void set_aco_cl_used_output(aco_cl* cl){
	if(cl == NULL)
		return;
	convolutional_used_output(cl->input_rows,cl->input_cols,cl->kernel_rows,cl->kernel_cols,cl->channels,cl->n_kernels,cl->output_used,cl->stride_rows,cl->stride_cols,cl->padding_rows,cl->padding_cols, cl->indices, cl->k_percentage);
}

int* get_aco_cl_used_outputs(aco_cl* a){
	if(a == NULL)
		return NULL;
	return a->output_used;
}





