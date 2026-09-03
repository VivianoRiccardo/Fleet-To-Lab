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


uint64_t get_aco_fcl_weights_or_pheromone_weights_size(int input, int output){
    return input*output;
}

uint64_t get_aco_fcl_biases_or_pheromone_biases_size(int input, int output){
    return output;
}

void init_aco_fcl_weights_xavier_init(int input, int output, float* weights){
    int i, size = input*output;
    for(i = 0; i < size; i++){
        weights[i] = random_general_gaussian_xavier_init((float)(input));
    }
}

void init_aco_fcl_weights_kaiming_constant_init(int input, int output, float* weights){
    int i, size = input*output;
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(input));
    }
}

void init_aco_fcl_weights_signed_kaiming_constant_init_parameters(int input, int output, float* weights, int number_of_parameters){
    int i, size = input*output*number_of_parameters;
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(input));
    }
}

void init_aco_fcl_weights_chaos_init_parameters(int input, int output, float* weights, int number_of_parameters){
    int i, size = input*output,j;
    for(i = 0; i < size; i++){
        float a = normalized_random();
        for(j = 0; j < number_of_parameters; j++){
            weights[i*number_of_parameters+j] = a;
            a = chaos_next_sequence(a);
        }
    }
}

void init_aco_fcl_weights_with_value(int input, int output, float* weights, float pheromone_value){
    set_vector_with_value(pheromone_value,weights,input*output);
}


void init_aco_fcl_biases_zeros(int output, float* biases){
    set_vector_with_value(0,biases,output);
}

void init_aco_fcl_biases_normalized_random(int output, float* biases){
    int i;
    for(i = 0; i < output; i++){
        biases[i] = signed_normalized_random(1.0);
    }
}

void init_aco_fcl_biases_signed_kaiming_constant(int input, int output, float* biases){
    int i;
    for(i = 0; i < output; i++){
        biases[i] = signed_kaiming_constant((float)input);
    }
}


void init_aco_fcl_biases_with_value(int output, float* biases, float pheromone_value){
    set_vector_with_value(pheromone_value,biases,output);
}

void init_aco_fcl_weights_pheromones(int input, int output, float* weights_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,weights_pheromone,input*output);
}

void init_aco_fcl_biases_pheromones(int output, float* biases_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,biases_pheromone,output);
}

aco_fcl* init_aco_fcl(int input_size, int output_size, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input){
    
    
    aco_fcl* f = (aco_fcl*)malloc(sizeof(aco_fcl));
    f->k_percentage = 1;
    f->input_size = input_size;
    f->output_size = output_size;
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

void assign_aco_fcl_storing_params(aco_fcl* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

void assign_aco_fcl_storing_params_output_used(aco_fcl* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_fcl_storing_partial_derivatives(aco_fcl* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
}

void assign_aco_fcl_learnable_params(aco_fcl* a, float* params){
    if(a == NULL)
        return;
    a->weights = params+a->weights_array_offset;
    a->biases = params+a->biases_array_offset;
}

void assign_aco_fcl_learnable_params_scores(aco_fcl* a, float* params, int* indices, float* scores){
    if(a == NULL)
        return;
    a->d_weights = params+a->weights_array_offset;
    a->scores = scores+a->weights_array_offset;
    a->indices = indices+a->weights_array_offset;
}

aco_fcl* copy_aco_fcl(aco_fcl* a){
    if(a == NULL)
        return NULL;
    aco_fcl* n = (aco_fcl*)malloc(sizeof(aco_fcl));
    n->k_percentage = a->k_percentage;
    n->indices = a->indices;
    n->input_size = a->input_size;
    n->output_size = a->output_size;
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
    n->output_used = a->output_used;
    return n;
}

void fully_connected_feed_forward(float* input, float* output, float* weight,float* bias, int input_size, int output_size){
    int i,j,val;
    for(j = 0; j < output_size; j++){
        val = j*input_size;
        for(i = 0; i < input_size; i++){
            output[j] += input[i]*weight[val+i];
        }
        //output[j]+=bias[j];
    }
}

void fully_connected_set_k_percentage(aco_fcl* a, float k_percentage){
    if(k_percentage > 1 || k_percentage < 0){
        fprintf(stderr,"Error, k percentage must be in [0,1]\n");
        exit(1);
    }
    if(a == NULL)
        return;
    a->k_percentage = k_percentage;
}


void fully_connected_feed_forward_edge_popup(float* input, float* output, float* weight,float* bias, int input_size, int output_size, int* indices, float k_percentage){
    int i,j;
    int last_n = input_size*output_size*k_percentage;
    int p = output_size*input_size;
    for(j = output_size*input_size-last_n; j < p; j++){
        output[(int)(indices[j]/input_size)] += input[(indices[j]%input_size)]*weight[indices[j]];
    }
}

void fully_connected_back_propagation_edge_popup(float* input, float* output_error, float* weight,float* bias, int input_size, int output_size, float* input_error, float* weight_error, float k_percentage, int* indices){
    int i,j,val;
    int last_n = input_size*output_size*k_percentage;
    int p = output_size*input_size;
    for(j = 0; j < output_size; j++){
        val = j*input_size;
        for(i = 0; i < input_size; i++){
            weight_error[val+i]+=output_error[j]*input[i]*weight[val+i];
        }
    }
    
    for(j = output_size*input_size-last_n; j < p; j++){
		input_error[(indices[j]%input_size)]+=output_error[(int)(indices[j]/input_size)]*weight[indices[j]];
    }
    /*
    printf("%d , %d\n",output_size,input_size);
    for(j = 0; j < input_size*output_size; j++){
        printf("A: %f , ",weight_error[j]); 
    }
    printf("\n");
    for(j = 0; j < input_size; j++){
        printf("B: %f , ",input_error[j]); 
    }
    printf("\n");*/
    
}


void aco_fcl_feed_forward(aco_fcl* f){
    if(f == NULL)
        return;
    fully_connected_feed_forward(f->input,f->output,f->weights,f->biases,f->input_size,f->output_size);
}

void aco_fcl_feed_forward_edge_popup(aco_fcl* f){
    if(f == NULL)
        return;
    fully_connected_feed_forward_edge_popup(f->input,f->output,f->weights,f->biases,f->input_size,f->output_size, f->indices, f->k_percentage);
}

void aco_fcl_back_propagation_edge_popup(aco_fcl* f){
    if(f == NULL)
        return;
    fully_connected_back_propagation_edge_popup(f->input,f->output_error,f->weights,f->biases,f->input_size,f->output_size, f->input_error, f->d_weights, f->k_percentage, f->indices);
}

void print_aco_fcl_offsets(aco_fcl* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
    printf("weights array offset: %lud\n",f->weights_array_offset);
    printf("biases array offset: %lud\n",f->biases_array_offset);
    print_vector(f->weights,f->input_size*f->output_size);
}

void merge_sort_aco_fcl(aco_fcl* c){
    if(c == NULL)
        return;
    merge_sort(c->scores,c->indices,0,c->input_size*c->output_size-1);
}

float* get_aco_fcl_d_output(aco_fcl* fcl){
    if(fcl == NULL)
        return NULL;
    return fcl->output_error;
}

void init_fcl_scores(aco_fcl* fcl){
    if(fcl == NULL)
        return;
    int i;
    
    for(i = 0; i < fcl->input_size*fcl->output_size; i++){
        fcl->indices[i] = i;
    }
}

void set_fcl_k_percentage(aco_fcl* c, float k){
    if(c == NULL)
        return;
    if(k < 0 || k > 1)
        return;
    c->k_percentage = k;
}

void paste_fcl_indices_weights_according_to_scores(aco_fcl* fcl, float* params){
	if(fcl == NULL || params == NULL)
		return;
	int i,p = fcl->output_size*fcl->input_size, last_n = p-p*fcl->k_percentage;
	for(i = last_n; i < p; i++){
		params[fcl->weights_array_offset+fcl->indices[i]] = 1;
	}
}

void fully_connected_used_output( int* output_used, int input_size, int output_size, int* indices, float k_percentage){
    int i,j;
    int last_n = input_size*output_size*k_percentage;
    int p = output_size*input_size;
    for(j = output_size*input_size-last_n; j < p; j++){
        output_used[(int)(indices[j]/input_size)] = 1;
    }
}


void set_aco_fcl_used_output(aco_fcl* fcl){
	if(fcl == NULL)
		return;
	fully_connected_used_output(fcl->output_used,fcl->input_size,fcl->output_size,fcl->indices,fcl->k_percentage);
}

int* get_aco_fcl_used_outputs(aco_fcl* a){
	if(a == NULL)
		return NULL;
	return a->output_used;
}







