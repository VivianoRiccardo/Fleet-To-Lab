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


aco_activation* init_aco_activation(int activation_flag, int activations_size, int input_size, int output_size,int identifier, uint64_t  output_array_offset,
                      uint64_t  input_array_offset, uint64_t pheromone_activations_offset, float* output, float* input, float* pheromone_activations, int* activations){
    
    
    aco_activation* f = (aco_activation*)malloc(sizeof(aco_activation));
    f->input_size = input_size;
    f->output_size = output_size;
    f->activation_flag = activation_flag;
    f->identifier = identifier;
    f->output_array_offset = output_array_offset;
    f->input_array_offset = input_array_offset;
    f->pheromone_activations_offset = pheromone_activations_offset;
    f->pheromone_activations = pheromone_activations;
    f->output = output;
    f->input = input;
    f->activations_size = activations_size;
    f->activations = activations;
    f->input_error = NULL;
    f->output_error = NULL;
    f->output_used = NULL;
    return f;
}

void assign_aco_activation_storing_params(aco_activation* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
}

void assign_aco_activation_storing_params_output_used(aco_activation* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
}

void assign_aco_activation_storing_partial_derivatives(aco_activation* a, float* d_storing_params){
    if(a == NULL)    
        return;
    a->output_error = d_storing_params+a->output_array_offset;
    a->input_error = d_storing_params+a->input_array_offset;
}

void free_aco_activation(aco_activation* a){
    if(a == NULL)
        return;
    free(a->activations);
    free(a);
}

aco_activation* copy_aco_activation(aco_activation* a){
    if(a == NULL)
        return NULL;
    aco_activation* n =(aco_activation*)malloc(sizeof(aco_activation));
    n->input_size = a->input_size;
    n->output_size = a->output_size;
    n->activation_flag = a->activation_flag;
    n->identifier = a->identifier;
    n->output_array_offset = a->output_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->pheromone_activations_offset = a->pheromone_activations_offset; 
    n->pheromone_activations = a->pheromone_activations;
    n->output = a->output;
    n->input = a->input;
    n->activations_size = a->activations_size;
    n->output_used = a->output_used;
    int i;
    if(a->activations != NULL){
        n->activations = (int*)calloc(a->activations_size,sizeof(int));
        copy_int_array(a->activations,n->activations,a->activations_size);
    }
    else{
        n->activations = NULL;
    }
    return n;
}

int* generate_all_activations(){
    int* a = (int*)calloc(N_ACTIVATION_FUNCTIONS_1_0,sizeof(int));
    /*
    a[0] = NO_ACTIVATION;
    a[1] = SIGMOID;
    a[2] = TANH;
    a[3] = RELU;
    a[4] = ELU;
    a[5] = LEAKY_RELU;
    a[6] = GELU;
    */
    a[0] = RELU;
    a[1] = ELU;
    a[2] = LEAKY_RELU;
    a[3] = GELU;
    return a;
}

void print_aco_activation_offsets(aco_activation* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
}

void aco_activation_ff(float* input,float* output,int activation_flag, int size){
    if(activation_flag == NO_ACTIVATION){
        //printf("CPU: NO_ACIVATION\n");
        copy_array(input,output,size);
        return;
    }
    if(activation_flag == SIGMOID){
        //printf("CPU: SIGMOID\n");
        sigmoid_array(input,output,size);
        return;
    }
    if(activation_flag == TANH){
        //printf("CPU: TANH\n");
        tanh_array(input,output,size);
        return;
    }
    if(activation_flag == RELU){
        //printf("CPU: RELU\n");
        relu_array(input,output,size);
        return;
    }
    if(activation_flag == LEAKY_RELU){
        //printf("CPU: LEAKY_RELU\n");
        leaky_relu_array(input,output,size);
        return;
    }
    if(activation_flag == ELU){
        //printf("CPU: ELU\n");
        elu_array(input,output,size);
        return;
    }
    if(activation_flag == GELU){
        //printf("CPU: GELU\n");
        gelu_array(input,output,size);
        return;
    }
    if(activation_flag == SOFTMAX){
        //printf("CPU: SOFTMAX\n");
        softmax(input,output,size);
        return;
    }
    
    if(activation_flag == INVERSE){
		compute_inverse(input,output,size);
	}
	
	if(activation_flag == NEGATIVE){
		mul_value(input,-1.0,output,size);
	}
	
	if(activation_flag == SQUARE){
		compute_square(input,output,size);
	}
	if(activation_flag == SQUARE_ROOT){
		compute_square_root(input,output,size);
	}
    
}

void aco_activation_ff_edge_popup(float* input,float* output,int* output_used, int activation_flag, int size){
    if(activation_flag == NO_ACTIVATION){
        //printf("CPU: NO_ACIVATION\n");
        copy_array(input,output,size);
        return;
    }
    if(activation_flag == SIGMOID){
        //printf("CPU: SIGMOID\n");
        sigmoid_array_with_flags(input,output,output_used, size);
        return;
    }
    if(activation_flag == TANH){
        //printf("CPU: TANH\n");
        tanh_array(input,output,size);
        return;
    }
    if(activation_flag == RELU){
        //printf("CPU: RELU\n");
        relu_array(input,output,size);
        return;
    }
    if(activation_flag == LEAKY_RELU){
        //printf("CPU: LEAKY_RELU\n");
        leaky_relu_array(input,output,size);
        return;
    }
    if(activation_flag == ELU){
        //printf("CPU: ELU\n");
        elu_array(input,output,size);
        return;
    }
    if(activation_flag == GELU){
        //printf("CPU: GELU\n");
        gelu_array(input,output,size);
        return;
    }
    if(activation_flag == SOFTMAX){
        //printf("CPU: SOFTMAX\n");
        softmax_with_flags(input,output,output_used,size);
        return;
    }
    if(activation_flag == INVERSE){
		compute_inverse_with_indices(input,output,output_used,size);
	}
	
	if(activation_flag == NEGATIVE){
		mul_value(input,-1,output,size);
		dot1D_float_with_int(output,output_used,output,size);
	}
	
	if(activation_flag == SQUARE){
		compute_square_with_indices(input,output,output_used,size);
	}
	if(activation_flag == SQUARE_ROOT){
		compute_square_root_with_indices(input,output,output_used,size);
	}
    
}

void aco_activation_bp(float* input,float* input_error, float* output,float* output_error, int* output_used, int activation_flag, int size){
    
    
    
    if(activation_flag == NO_ACTIVATION){// si edge popup
        sum_vectors(input_error,output_error,input_error,size);
        return;
    }
    if(activation_flag == SOFTMAX){// si edge popup
        derivative_softmax_with_flags(input_error,output,output_error,output_used,size);
        return;
    }
    
    float* temp_output =(float*)calloc(size,sizeof(float));
    int i;
    
    if(activation_flag == SIGMOID){// si edge popup
        derivative_sigmoid_array_with_flags(input,temp_output,output_used,size);
    }
    if(activation_flag == TANH){// si edge popup
        derivative_tanh_array(input,temp_output,size);
    }
    if(activation_flag == RELU){// si edge popup
        derivative_relu_array(input,temp_output,size);
    }
    if(activation_flag == LEAKY_RELU){// si edge popup
        derivative_leaky_relu_array(input,temp_output,size);
    }
    if(activation_flag == ELU){// si edge popup
        derivative_elu_array(input,temp_output,size);
    }
    if(activation_flag == GELU){// si edge popup
        derivative_gelu_array(input,temp_output,size);
    }
    
    if(activation_flag == INVERSE){
		compute_inverse_derivative_with_indices(input,temp_output,output_used,size);
	}
	
	if(activation_flag == NEGATIVE){
		mul_value_float_with_int(output_used,-1,temp_output,size);
	}
	
	if(activation_flag == SQUARE){
		compute_square_derivative_with_indices(input,temp_output,output_used,size);
	}
	if(activation_flag == SQUARE_ROOT){
		compute_square_root_derivative_with_indices(input,temp_output,output_used,size);
	}
    
    for(i = 0; i < size; i++){
        input_error[i]+=output_error[i]*temp_output[i]*((float)output_used[i]);
    }
    free(temp_output);
}

void aco_activation_feed_forward(aco_activation* activation){
    if(activation == NULL)
        return;
    aco_activation_ff(activation->input,activation->output,activation->activation_flag,activation->output_size);
}

void aco_activation_feed_forward_edge_popup(aco_activation* activation){
    if(activation == NULL)
        return;
    aco_activation_ff_edge_popup(activation->input,activation->output,activation->output_used, activation->activation_flag,activation->output_size);
}

float* get_aco_activations_d_output(aco_activation* activation){
    if(activation == NULL)
        return NULL;
    return activation->output_error;
}

void aco_activation_back_propagation(aco_activation* activation){
    if(activation == NULL)
        return;
    aco_activation_bp(activation->input,activation->input_error,activation->output,activation->output_error,activation->output_used, activation->activation_flag,activation->output_size);
}

void copy_activation_flag(aco_activation* copy, aco_activation* original){
    if(copy == NULL || original == NULL)
        return;
    copy->activation_flag = original->activation_flag;
}

void set_aco_activation_used_output(aco_activation* a,int* previous_used_outputs, int is_root){
    if(a == NULL)
        return;
    if(is_root)
        set_int_vector_with_value(1,a->output_used,a->output_size);
    else
        copy_int_array(previous_used_outputs,a->output_used,a->output_size);
}

int* get_aco_activation_used_outputs(aco_activation* a){
    if(a == NULL)
        return NULL;
    return a->output_used;
}

