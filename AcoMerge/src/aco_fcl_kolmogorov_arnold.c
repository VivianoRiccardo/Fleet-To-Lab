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


uint64_t get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(int input, int output, int n_func){
    return input*output*(n_func+2);
}

uint64_t get_aco_fcl_kolmogorov_arnold_ts_or_pheromone_biases_size(int input, int output, int n_func, int degree){
    return input*output*(n_func+degree+1);
}


void init_aco_fcl_kolmogorov_arnold_weights_xavier_init(int input, int output,float* weights,int n_func){
    int i, size = input*output*(n_func+1);
    for(i = 0; i < size; i++){
        weights[i] = random_general_gaussian_xavier_init((float)(input));
    }
}

void init_aco_fcl_kolmogorov_arnold_weights_kaiming_constant_init(int input, int output, float* weights,int n_func){
    int i, size = input*output*(n_func+2);
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(input));
    }
}

void init_aco_fcl_kolmogorov_arnold_weights_signed_kaiming_constant_init_parameters(int input, int output, float* weights, int number_of_parameters,int n_func){
    int i, size = number_of_parameters*input*output*(n_func+2);
    for(i = 0; i < size; i++){
        weights[i] = signed_kaiming_constant((float)(input));
    }
}

void init_aco_fcl_kolmogorov_arnold_weights_chaos_init_parameters(int input, int output, float* weights, int number_of_parameters,int n_func){
    int i, size = input*output*(n_func+2),j;
    for(i = 0; i < size; i++){
        float a = normalized_random();
        for(j = 0; j < number_of_parameters; j++){
            weights[i*number_of_parameters+j] = a;
            a = chaos_next_sequence(a);
        }
    }
}

void init_aco_fcl_kolmogorov_arnold_weights_with_value(int input, int output, float* weights, float pheromone_value,int n_func){
    set_vector_with_value(pheromone_value,weights,input*output*(n_func+2));
}


void init_aco_fcl_ts_zeros(int input, int output, int n_func, int degree, float* ts){
    set_vector_with_value(0,ts,input*output*(n_func+degree+1));
}

void init_aco_fclkolmogorov_arnold_ts_normalized_random(int input, int output, int n_func, int degree, float* ts){
    int i, size = input*output*(n_func+degree+1);
    for(i = 0; i < size; i++){
        ts[i] = signed_normalized_random(1.0);
    }
}

void init_aco_fcl_kolmogorov_arnold_ts_signed_kaiming_constant(int input, int output, int n_func, int degree, float* ts){
    int i, size = input*output*(n_func+degree+1);
    for(i = 0; i < size; i++){
        ts[i] = signed_kaiming_constant((float)input);
    }
}

void init_aco_fcl_kolmogorov_arnold_ts_standard(int input, int output, int n_func, int degree, float* ts){
    int i, k,size2 = n_func+degree+1;
    float step = 2.0/((float)(n_func+degree)),j;
    for(k = 0; k < input*output; k++){
        for(i = 0,j = -1; i < size2; i++,j+=step){
            ts[k*size2+i] = j;
        }
    }
}

void init_aco_fcl_kolmogorov_arnold_ts_standard_n_parameters(int input, int output, int n_func, int degree, int n_parameters, float* ts){
    int i, k,z,size2 = n_func+degree+1;
    
    float step = 2.0/((float)(n_func+degree)),j, step2 = step/((float)(n_parameters-1));
    for(k = 0; k < input*output; k++){
        for(i = 0,j = -1; i < size2; i++,j+=step){
            for(z = 0; z < n_parameters; z++){
                ts[k*n_parameters*size2+i*n_parameters+z] = j+z*step2;
            }
        }
    }
}


void clip_aco_fcl_kolmogorov_arnold_ts(int input, int output, int n_func, int degree, int n_parameters, float* ts){
    /*int i, k,z,size2 = n_func+degree+1,u;
    float step = 2.0/((float)(n_func+degree+1)),j;
    for(k = 0; k < input*output; k++){
        for(i = 0,j = -1; i < size2; i++,j+=step){
            for(z = 0; z < n_parameters; z++){
                if(ts[k*n_parameters*size2+i*n_parameters+z] > j+step)
                    ts[k*n_parameters*size2+i*n_parameters+z] = j+step;
                if(ts[k*n_parameters*size2+i*n_parameters+z] < j)
                    ts[k*n_parameters*size2+i*n_parameters+z] = j;
                
            }
        }
    }*/
    int i, k,z,size2 = n_func+degree+1;
    
    float step = 2.0/((float)(n_func+degree)),j, step2 = step/((float)(n_parameters-1));
    for(k = 0; k < input*output; k++){
        for(i = 0,j = -1; i < size2; i++,j+=step){
            for(z = 0; z < n_parameters; z++){
                ts[k*n_parameters*size2+i*n_parameters+z] = j+z*step2;
            }
        }
    }
}


void init_aco_fcl_kolmogorov_arnold_ts_with_value(int input, int output, int n_func, int degree, float* ts, float pheromone_value){
    set_vector_with_value(pheromone_value,ts,input*output*(n_func+degree+1));
}

void init_aco_fcl_kolmogorov_arnold_weights_pheromones(int input, int output, float* weights_pheromone, float pheromone_value, int n_func){
    set_vector_with_value(pheromone_value,weights_pheromone,input*output*(n_func+2));
}

void init_aco_fcl_kolmogorov_arnold_ts_pheromones(int input, int output, int n_func, int degree, float* ts_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,ts_pheromone,input*output*(n_func+degree+1));
}

aco_fcl_kolmogorov_arnold* init_aco_fcl_kolmogorov_arnold(int input_size, int output_size,int n_functions, int maximum_degree, int identifier, uint64_t weights_array_offset, uint64_t  ts_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_ts_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, float* weights, float* ts, float* pheromone_weights, float* pheromone_ts, float* output, float* input){
    
    
    aco_fcl_kolmogorov_arnold* f = (aco_fcl_kolmogorov_arnold*)malloc(sizeof(aco_fcl_kolmogorov_arnold));
    f->input_size = input_size;
    f->output_size = output_size;
    f->n_functions = n_functions;
    f->maximum_degree = maximum_degree;
    f->identifier = identifier;
    f->weights = weights;
    f->ts = ts;
    f->pheromone_weights = pheromone_weights;
    f->pheromone_ts = pheromone_ts;
    f->output = output;
    f->input = input;
    f->weights_array_offset = weights_array_offset;
    f->ts_array_offset = ts_array_offset;
    f->pheromone_weights_array_offset = pheromone_weights_array_offset;
    f->pheromone_ts_array_offset = pheromone_ts_array_offset;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->input_error = NULL;
    f->output_error = NULL;
    return f;
}

void assign_aco_fcl_kolmogorov_arnold_storing_params(aco_fcl_kolmogorov_arnold* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

aco_fcl_kolmogorov_arnold* copy_aco_fcl_kolmogorov_arnold(aco_fcl_kolmogorov_arnold* a){
    if(a == NULL)
        return NULL;
    aco_fcl_kolmogorov_arnold* n = (aco_fcl_kolmogorov_arnold*)malloc(sizeof(aco_fcl_kolmogorov_arnold));
    n->input_size = a->input_size;
    n->output_size = a->output_size;
    n->n_functions = a->n_functions;
    n->maximum_degree = a->maximum_degree;
    n->identifier = a->identifier;
    n->weights = a->weights;
    n->ts = a->ts;
    n->pheromone_weights = a->pheromone_weights;
    n->pheromone_ts = a->pheromone_ts;
    n->output = a->output;
    n->input = a->input;
    n->weights_array_offset = a->weights_array_offset;
    n->ts_array_offset = a->ts_array_offset;
    n->pheromone_weights_array_offset = a->pheromone_weights_array_offset;
    n->pheromone_ts_array_offset = a->pheromone_ts_array_offset;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    return n;
}

void fully_connected_kolmogorov_arnold_feed_forward(float* input, float* output, float* weight,float* ts, int input_size, int output_size, int n_functions,int degree){
    int i,j,z,val,val2,size = input_size*(2+n_functions), val_ts, val2_ts, size2 = input_size*(n_functions+degree+1);
    double sum;
    for(j = 0; j < output_size; j++){
        val = j*size;
        val_ts = j*size2;
        for(i = 0,sum = 0; i < input_size; i++,sum=0){
            val2 = i*(2+n_functions);
            val2_ts = i*(n_functions+degree+1);
            for(z = 2; z < n_functions+2; z++){
                float t = b_spline(input[i],ts+val_ts+val2_ts,degree,z-2,1);
                sum+=weight[val+val2+z]*t;
            }
            output[j]+= silu(input[i])*weight[val+val2] + sum*weight[val+val2+1];            
        }
    }
}


void aco_fcl_kolmogorov_arnold_feed_forward(aco_fcl_kolmogorov_arnold* f){
    if(f == NULL)
        return;
    fully_connected_kolmogorov_arnold_feed_forward(f->input,f->output,f->weights,f->ts,f->input_size,f->output_size,f->n_functions,f->maximum_degree);
}

void aco_fcl_kolmogorov_arnold_clip(aco_fcl_kolmogorov_arnold* f, float* array_to_clip, int n_parameters){
    if(f == NULL)
        return;
    clip_aco_fcl_kolmogorov_arnold_ts(f->input_size,f->output_size,f->n_functions,f->maximum_degree,n_parameters,array_to_clip+f->ts_array_offset*n_parameters);
}

void print_aco_fcl_kolmogorov_arnold_offsets(aco_fcl_kolmogorov_arnold* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
    printf("weights array offset: %lud\n",f->weights_array_offset);
    printf("ts array offset: %lud\n",f->ts_array_offset);
}






