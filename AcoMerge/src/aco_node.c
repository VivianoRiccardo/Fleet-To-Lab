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


aco_node* init_aco_node(int n_input, int n_output, int identifier, aco_fcl* fcl, aco_cl* cl, aco_lstm* lstm,
                        aco_sum* sum, aco_pooling* pooling, aco_concatenation* concatenation, aco_activation* activation, aco_fcl_kolmogorov_arnold* fcl_ka,aco_splitter* splitter,
                        aco_node** input, aco_node** output){
    
    
    aco_node* f = (aco_node*)malloc(sizeof(aco_node));
    f->input = input;
    f->output = output;
    f->fcl = fcl;
    f->fcl_ka = fcl_ka;
    f->cl = cl;
    f->lstm = lstm;
    f->sum = sum;
    f->pooling = pooling;
    f->concatenation = concatenation;
    f->activation = activation;
    f->splitter = splitter;
    f->identifier = identifier;
    f->n_input = n_input;
    f->n_output = n_output;
    f->ff_flag = 0;
    return f;
}

void aco_node_add_input(aco_node* f, aco_node* input){
    f->input = (aco_node**)realloc(f->input,f->n_input+1);
    f->input[f->n_input] = input;
    f->n_input++;
}

void aco_node_add_output(aco_node* f, aco_node* output){
    f->output = (aco_node**)realloc(f->output,f->n_output+1);
    f->output[f->n_output] = output;
    f->n_output++;
}

void aco_node_feed_forward(aco_node* n){
    aco_fcl_feed_forward(n->fcl);
    aco_fcl_kolmogorov_arnold_feed_forward(n->fcl_ka);
    aco_cl_feed_forward(n->cl);
    aco_lstm_feed_forward(n->lstm);
    aco_pooling_feed_forward(n->pooling);
    aco_sum_feed_forward(n->sum);
    aco_concatenation_feed_forward(n->concatenation);
    aco_activation_feed_forward(n->activation);
    n->ff_flag = 1;
}

void aco_node_feed_forward_edge_popup(aco_node* n){
    aco_fcl_feed_forward_edge_popup(n->fcl);
    //aco_fcl_kolmogorov_arnold_feed_forward(n->fcl_ka);
    aco_cl_feed_forward_edge_popup(n->cl);
    aco_lstm_feed_forward_edge_popup(n->lstm);
    aco_pooling_feed_forward(n->pooling);
    aco_sum_feed_forward_edge_popup(n->sum);
    aco_concatenation_feed_forward(n->concatenation);
    aco_activation_feed_forward_edge_popup(n->activation);
    n->ff_flag = 1;
}

void aco_node_back_propagation_edge_popup(aco_node* n){
    aco_fcl_back_propagation_edge_popup(n->fcl);
    //aco_fcl_kolmogorov_arnold_feed_forward(n->fcl_ka);
    aco_cl_back_propagation_edge_popup(n->cl);
    aco_lstm_back_propagation_edge_popup(n->lstm);
    aco_pooling_back_propagation(n->pooling);
    aco_sum_back_propagation(n->sum);
    aco_concatenation_back_propagation(n->concatenation);
    aco_activation_back_propagation(n->activation);
    n->ff_flag = 1;
}


void set_ff_flag_to_zero(aco_node* n){
    n->ff_flag = 0;
}

void print_aco_node_offsets(aco_node* f){
    if(f->fcl != NULL){
        printf("Fcl\n");
        print_aco_fcl_offsets(f->fcl);
    }
    if(f->fcl_ka != NULL){
        printf("Fcl_KA\n");
        print_aco_fcl_kolmogorov_arnold_offsets(f->fcl_ka);
    }
    else if(f->cl != NULL){
        printf("Cl\n");
        print_aco_cl_offsets(f->cl);
    }
    else if(f->pooling != NULL){
        printf("Pooling\n");
        print_aco_pooling_offsets(f->pooling);
    }
    else if(f->sum != NULL){
        printf("Sum\n");
        print_aco_sum_offsets(f->sum);
    }
    else if(f->concatenation != NULL){
        printf("Concatenation\n");
        print_aco_concatenation_offsets(f->concatenation);
    }
    else if(f->lstm != NULL){
        printf("Lstm\n");
        print_aco_lstm_offsets(f->lstm);
    }
    else if(f->activation != NULL){
        printf("Activation\n");
        print_aco_activation_offsets(f->activation);
    }
}

void free_aco_node(aco_node* n){
    free(n->fcl);
    free(n->fcl_ka);
    free(n->cl);
    free(n->pooling);
    free(n->lstm);
    free(n->sum);
    free_aco_activation(n->activation);
    free(n->concatenation);
    free(n->splitter);
    free(n->input);
    free(n->output);
    free(n);
}

uint64_t get_aco_node_params_size(int** same_params, aco_node* n, int n_nodes, int index){
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i]){
            return 0;
        }
    }
    
    if(n->cl != NULL){
        //return get_aco_cl_weights_or_pheromone_weights_size(n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels)+get_aco_cl_biases_or_pheromone_biases_size(n->cl->n_kernels);
        return get_aco_cl_weights_or_pheromone_weights_size(n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels);
    }
    else if(n->fcl != NULL){
        //return get_aco_fcl_weights_or_pheromone_weights_size(n->fcl->input_size, n->fcl->output_size)+get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size);
        return get_aco_fcl_weights_or_pheromone_weights_size(n->fcl->input_size, n->fcl->output_size);
    }
    
    else if(n->lstm != NULL){
        //return get_aco_lstm_weights_or_pheromone_weights_size(n->lstm->input_size, n->lstm->output_size)+get_aco_lstm_biases_or_pheromone_biases_size(n->lstm->output_size); 
        return get_aco_lstm_weights_or_pheromone_weights_size(n->lstm->input_size, n->lstm->output_size); 
    }
    
    else if(n->fcl_ka != NULL){
        return get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions)+get_aco_fcl_kolmogorov_arnold_ts_or_pheromone_biases_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions, n->fcl_ka->maximum_degree);
    }
    
    return 0;
}



void assign_array_aco_node_params(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index){
    
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i]){
            if(nodes[i]->cl != NULL){
                n->cl->weights = nodes[i]->cl->weights;
                n->cl->weights_array_offset = nodes[i]->cl->weights_array_offset;
                //n->cl->biases = nodes[i]->cl->biases;
                //n->cl->biases_array_offset = nodes[i]->cl->biases_array_offset;
            }
            if(nodes[i]->fcl != NULL){
                n->fcl->weights = nodes[i]->fcl->weights;
                n->fcl->weights_array_offset = nodes[i]->fcl->weights_array_offset;
                //n->fcl->biases = nodes[i]->fcl->biases;
                //n->fcl->biases_array_offset = nodes[i]->fcl->biases_array_offset;
            }
            if(nodes[i]->lstm != NULL){
                n->lstm->weights = nodes[i]->lstm->weights;
                n->lstm->weights_array_offset = nodes[i]->lstm->weights_array_offset;
                //n->lstm->biases = nodes[i]->lstm->biases;
                //n->lstm->biases_array_offset = nodes[i]->lstm->biases_array_offset;
            }
            if(nodes[i]->fcl_ka != NULL){
                n->fcl_ka->weights = nodes[i]->fcl_ka->weights;
                n->fcl_ka->weights_array_offset = nodes[i]->fcl_ka->weights_array_offset;
                n->fcl_ka->ts = nodes[i]->fcl_ka->ts;
                n->fcl_ka->ts_array_offset = nodes[i]->fcl_ka->ts_array_offset;
            }
            return;
        }
    }
    
        
    if(n->cl != NULL){
        n->cl->weights = array+(*size);
        n->cl->weights_array_offset = (*size);
        (*size)+=get_aco_cl_weights_or_pheromone_weights_size(n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels);
        //n->cl->biases = array+(*size);
        //n->cl->biases_array_offset = (*size);
        //(*size)+=get_aco_cl_biases_or_pheromone_biases_size(n->cl->n_kernels);
        init_aco_cl_weights_kaiming_constant_init(n->cl->input_rows, n->cl->input_cols, n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels, n->cl->weights);
        //init_aco_cl_biases_signed_kaiming_constant(n->cl->input_rows, n->cl->input_cols, n->cl->channels,n->cl->n_kernels, n->cl->biases);
    }
    else if(n->fcl != NULL){
        n->fcl->weights = array+(*size);
        n->fcl->weights_array_offset = (*size);
        (*size)+=get_aco_fcl_weights_or_pheromone_weights_size(n->fcl->input_size, n->fcl->output_size);
        //n->fcl->biases = array+(*size);
        //n->fcl->biases_array_offset = (*size);
        //(*size)+=get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size);
        init_aco_fcl_weights_kaiming_constant_init(n->fcl->input_size, n->fcl->output_size, n->fcl->weights);
        //init_aco_fcl_biases_signed_kaiming_constant(n->fcl->input_size,n->fcl->output_size, n->fcl->biases);
    }
    else if(n->fcl_ka != NULL){
        n->fcl_ka->weights = array+(*size);
        n->fcl_ka->weights_array_offset = (*size);
        (*size)+=get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions);
        /* we omit the ts arrays because they will be stored at the end of the weights array
        n->fcl->biases = array+(*size);
        n->fcl->biases_array_offset = (*size);
        (*size)+=get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size);
        * */
        init_aco_fcl_kolmogorov_arnold_weights_kaiming_constant_init(n->fcl_ka->input_size, n->fcl_ka->output_size, n->fcl_ka->weights,n->fcl_ka->n_functions);
        // omitting ts initialization init_aco_fcl_biases_signed_kaiming_constant(n->fcl->input_size,n->fcl->output_size, n->fcl->biases);
    }
    
    else if(n->lstm != NULL){
        n->lstm->weights = array+(*size);
        n->lstm->weights_array_offset = (*size);
        (*size)+=get_aco_lstm_weights_or_pheromone_weights_size(n->lstm->input_size, n->lstm->output_size);
        //n->lstm->biases = array+(*size);
        //n->lstm->biases_array_offset = (*size);
        //(*size)+=get_aco_lstm_biases_or_pheromone_biases_size(n->lstm->output_size);
        init_aco_lstm_weights_kaiming_constant_init(n->lstm->input_size, n->lstm->output_size, n->lstm->weights);
        init_aco_lstm_p(n->lstm->input_size, n->lstm->output_size, n->lstm->weights);
        //init_aco_lstm_biases_signed_kaiming_constant(n->lstm->output_size, n->lstm->biases);
    }

}

void assign_array_aco_node_params_only_for_kolmogorov_arnold_ts(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index){
    
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i]){
            if(nodes[i]->fcl_ka != NULL){
                n->fcl_ka->weights = nodes[i]->fcl_ka->weights;
                n->fcl_ka->weights_array_offset = nodes[i]->fcl_ka->weights_array_offset;
                n->fcl_ka->ts = nodes[i]->fcl_ka->ts;
                n->fcl_ka->ts_array_offset = nodes[i]->fcl_ka->ts_array_offset;
            }
            return;
        }
    }
    
        
    if(n->fcl_ka != NULL){
        /*n->fcl_ka->weights = array+(*size);
        n->fcl_ka->weights_array_offset = (*size);
        (*size)+=get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions);*/
        n->fcl_ka->ts = array+(*size);
        n->fcl_ka->ts_array_offset = (*size);
        (*size)+=get_aco_fcl_kolmogorov_arnold_ts_or_pheromone_biases_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions,n->fcl_ka->maximum_degree);
        init_aco_fcl_kolmogorov_arnold_ts_standard(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions,n->fcl_ka->maximum_degree, n->fcl_ka->ts);
    }

}
void generate_array_aco_node_params(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params){
    
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i])
            return;
        
    }
    
        
    if(n->cl != NULL){
        init_aco_cl_weights_signed_kaiming_constant_init_parameters(n->cl->input_rows, n->cl->input_cols, n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels, params+(*size),number_of_parameters);
        params[(*size)] = 0;
        (*size)+=get_aco_cl_weights_or_pheromone_weights_size(n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels)*number_of_parameters;
        //init_aco_cl_biases_signed_kaiming_constant(n->cl->input_rows, n->cl->input_cols, n->cl->channels,n->cl->n_kernels*number_of_parameters, params+(*size));
        //params[(*size)] = 0;
        //(*size)+=get_aco_cl_biases_or_pheromone_biases_size(n->cl->n_kernels)*number_of_parameters;
    }
    else if(n->fcl != NULL){
        init_aco_fcl_weights_signed_kaiming_constant_init_parameters(n->fcl->input_size, n->fcl->output_size, params+(*size),number_of_parameters);
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_weights_or_pheromone_weights_size(n->fcl->input_size, n->fcl->output_size)*number_of_parameters;
        //init_aco_fcl_biases_signed_kaiming_constant(n->fcl->input_size,n->fcl->output_size*number_of_parameters, params+(*size));
        //params[(*size)] = 0;
        //(*size)+=get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size)*number_of_parameters;
    }
    else if(n->fcl_ka != NULL){
        init_aco_fcl_kolmogorov_arnold_weights_signed_kaiming_constant_init_parameters(n->fcl_ka->input_size, n->fcl_ka->output_size, params+(*size),number_of_parameters,n->fcl_ka->n_functions);
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions)*number_of_parameters;
        /* we omit the ts
         * init_aco_fcl_biases_signed_kaiming_constant(n->fcl->input_size,n->fcl->output_size*number_of_parameters, params+(*size));
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size)*number_of_parameters;*/
    }
    
    else if(n->lstm != NULL){
        init_aco_lstm_weights_signed_kaiming_constant_init_parameters(n->lstm->input_size, n->lstm->output_size, params+(*size),number_of_parameters);
        params[(*size)] = 0;
        (*size)+=(get_aco_lstm_weights_or_pheromone_weights_size(n->lstm->input_size, n->lstm->output_size)-1)*(number_of_parameters);
        for(i = 0; i < number_of_parameters; i++){
            params[(*size)+i] = normalized_random();
            //params[(*size)+i] = i/((float)(number_of_parameters));
            //params[(*size)+i] = 0.5;
        }
        params[(*size)] = 0;
        (*size)+=number_of_parameters;
        init_aco_lstm_biases_normalized_random(n->lstm->output_size*number_of_parameters, params+(*size));
        //params[(*size)] = 0;
        //(*size)+=get_aco_lstm_biases_or_pheromone_biases_size(n->lstm->output_size)*number_of_parameters;
    }

}
void generate_array_aco_node_params_only_for_kolmogorov_arnold_ts(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params){
    
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i])
            return;
        
    }
    
        
    if(n->fcl_ka != NULL){
        
        /*init_aco_fcl_kolmogorov_arnold_weights_signed_kaiming_constant_init_parameters(n->fcl_ka->input_size, n->fcl_ka->output_size, params+(*size),number_of_parameters,n->fcl_ka->n_functions);
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_kolmogorov_arnold_weights_or_pheromone_weights_size(n->fcl_ka->input_size, n->fcl_ka->output_size,n->fcl_ka->n_functions)*number_of_parameters;*/
        init_aco_fcl_kolmogorov_arnold_ts_standard_n_parameters(n->fcl_ka->input_size,n->fcl_ka->output_size,n->fcl_ka->n_functions,n->fcl_ka->maximum_degree,number_of_parameters, params+(*size));
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_kolmogorov_arnold_ts_or_pheromone_biases_size(n->fcl_ka->input_size,n->fcl_ka->output_size,n->fcl_ka->n_functions,n->fcl_ka->maximum_degree)*number_of_parameters;
    }
    
}

void generate_array_aco_node_params_chaos_initialization(aco_node** nodes, int** same_params, aco_node* n, uint64_t* size, float* array, int n_nodes, int index, int number_of_parameters, float* params){
    
    int i;
    for(i = 0; i < index; i++){
        if(same_params[index][i])
            return;
        
    }
    
        
    if(n->cl != NULL){
        init_aco_cl_weights_chaos_init_parameters(n->cl->input_rows, n->cl->input_cols, n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels, params+(*size),number_of_parameters);
        (*size)+=get_aco_cl_weights_or_pheromone_weights_size(n->cl->channels, n->cl->kernel_rows, n->cl->kernel_cols, n->cl->n_kernels)*number_of_parameters;
        init_aco_cl_biases_normalized_random(n->cl->n_kernels*number_of_parameters, params+(*size));
        params[(*size)] = 0;
        (*size)+=get_aco_cl_biases_or_pheromone_biases_size(n->cl->n_kernels)*number_of_parameters;
    }
    else if(n->fcl != NULL){
        init_aco_fcl_weights_chaos_init_parameters(n->fcl->input_size, n->fcl->output_size, params+(*size),number_of_parameters);
        (*size)+=get_aco_fcl_weights_or_pheromone_weights_size(n->fcl->input_size, n->fcl->output_size)*number_of_parameters;
        init_aco_fcl_biases_normalized_random(n->fcl->output_size*number_of_parameters, params+(*size));
        params[(*size)] = 0;
        (*size)+=get_aco_fcl_biases_or_pheromone_biases_size(n->fcl->input_size, n->fcl->output_size)*number_of_parameters;
    }
    
    else if(n->lstm != NULL){
        init_aco_lstm_weights_chaos_init_parameters(n->lstm->input_size, n->lstm->output_size, params+(*size),number_of_parameters);
        (*size)+=(get_aco_lstm_weights_or_pheromone_weights_size(n->lstm->input_size, n->lstm->output_size)-1)*(number_of_parameters);
        for(i = 0; i < number_of_parameters; i++){
            params[(*size)+i] = normalized_random();
        }
        (*size)+=number_of_parameters;
        init_aco_lstm_biases_normalized_random(n->lstm->output_size*number_of_parameters, params+(*size));
        params[(*size)] = 0;
        (*size)+=get_aco_lstm_biases_or_pheromone_biases_size(n->lstm->output_size)*number_of_parameters;
    }

}

aco_node* copy_aco_node_without_input_and_output(aco_node* n){
    return init_aco_node(0, 0, n->identifier, copy_aco_fcl(n->fcl), copy_aco_cl(n->cl), copy_aco_lstm(n->lstm), copy_aco_sum(n->sum), copy_aco_pooling(n->pooling), copy_aco_concatenation(n->concatenation), copy_aco_activation(n->activation),copy_aco_fcl_kolmogorov_arnold(n->fcl_ka), copy_aco_splitter(n->splitter), NULL, NULL);
}


void assign_aco_node_storing_params(aco_node* a, float* storing_params){
    assign_aco_activation_storing_params(a->activation,storing_params);
    assign_aco_cl_storing_params(a->cl,storing_params);
    assign_aco_fcl_storing_params(a->fcl,storing_params);
    assign_aco_fcl_kolmogorov_arnold_storing_params(a->fcl_ka,storing_params);
    assign_aco_lstm_storing_params(a->lstm,storing_params);
    assign_aco_concatenation_storing_params(a->concatenation,storing_params);
    assign_aco_sum_storing_params(a->sum,storing_params);
    assign_aco_pooling_storing_params(a->pooling,storing_params);
    assign_aco_splitter_storing_params(a->splitter,storing_params);
}

void assign_aco_node_storing_params_output_used(aco_node* a, int* storing_params){
    assign_aco_activation_storing_params_output_used(a->activation,storing_params);
    assign_aco_cl_storing_params_output_used(a->cl,storing_params);
    assign_aco_fcl_storing_params_output_used(a->fcl,storing_params);
    //assign_aco_fcl_kolmogorov_arnold_storing_params(a->fcl_ka,storing_params);
    assign_aco_lstm_storing_params_output_used(a->lstm,storing_params);
    assign_aco_concatenation_storing_params_output_used(a->concatenation,storing_params);
    assign_aco_sum_storing_params_output_used(a->sum,storing_params);
    assign_aco_pooling_storing_params_output_used(a->pooling,storing_params);
    assign_aco_splitter_params_output_used(a->splitter,storing_params);
}

void assign_aco_node_learnable_params(aco_node* a, float* params){
    assign_aco_fcl_learnable_params(a->fcl,params);
    assign_aco_cl_learnable_params(a->cl,params);
    assign_aco_lstm_learnable_params(a->lstm,params);
}

void assign_aco_node_learnable_params_scores(aco_node* a, float* params, int* indices, float* scores){
    if(a == NULL)
        return;
    assign_aco_fcl_learnable_params_scores(a->fcl,params, indices, scores);
    assign_aco_cl_learnable_params_scores(a->cl,params, indices, scores);
    assign_aco_lstm_learnable_params_scores(a->lstm,params, indices, scores);
}

int get_aco_node_output_size(aco_node* n){
    if(n->activation != NULL){
        return n->activation->output_size;
    }
    else if(n->fcl != NULL){
        return n->fcl->output_size;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->output_size;
    }
    else if(n->cl != NULL){
        return aco_cl_output_size(n->cl->n_kernels, n->cl->input_rows, n->cl->input_cols,n->cl->kernel_rows,n->cl->kernel_cols,n->cl->stride_rows,n->cl->stride_cols,n->cl->padding_rows,n->cl->padding_cols);
    }
    else if(n->lstm != NULL){
        return n->lstm->output_size;
    }
    else if(n->concatenation != NULL){
        return n->concatenation->output_size;
    }
    else if(n->sum != NULL){
        return n->sum->output_size;
    }
    else if(n->pooling != NULL){
        return aco_pooling_output_size(n->pooling->channels, n->pooling->input_rows, n->pooling->input_cols,n->pooling->pooling_rows,n->pooling->pooling_cols,n->pooling->stride_rows,n->pooling->stride_cols,n->pooling->padding_rows,n->pooling->padding_cols);
    }
    else if(n->splitter != NULL){
        return n->splitter->output_size;
    }
    return 0;
}

int get_aco_node_input_size(aco_node* n){
    if(n->activation != NULL){
        return n->activation->input_size;
    }
    else if(n->fcl != NULL){
        return n->fcl->input_size;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->input_size;
    }
    else if(n->cl != NULL){
        return aco_cl_input_size(n->cl->channels, n->cl->input_rows, n->cl->input_cols);
    }
    else if(n->lstm != NULL){
        return n->lstm->input_size;
    }
    else if(n->concatenation != NULL){
            return n->concatenation->output_size;
    }
    else if(n->sum != NULL){
        return n->sum->input_size+n->sum->input2_size;
    }
    else if(n->pooling != NULL){
        return aco_pooling_input_size(n->pooling->channels, n->pooling->input_rows, n->pooling->input_cols);
    }
    else if(n->splitter != NULL){
        return n->splitter->input_size;
    }
    return 0;
}

int get_aco_node_first_input_size(aco_node* n){
    if(n->activation != NULL){
        return n->activation->input_size;
    }
    else if(n->fcl != NULL){
        return n->fcl->input_size;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->input_size;
    }
    else if(n->cl != NULL){
        return aco_cl_input_size(n->cl->channels, n->cl->input_rows, n->cl->input_cols);
    }
    else if(n->lstm != NULL){
        return n->lstm->input_size;
    }
    else if(n->concatenation != NULL){
            return n->concatenation->output_size;
    }
    else if(n->sum != NULL){
        return n->sum->input_size;
    }
    else if(n->pooling != NULL){
        return aco_pooling_input_size(n->pooling->channels, n->pooling->input_rows, n->pooling->input_cols);
    }
    else if(n->splitter != NULL){
        return n->splitter->input_size;
    }
}

int get_aco_node_second_input_size(aco_node* n){
    
    if(n->sum != NULL){
        return n->sum->input2_size;
    }
    return -1;
}

uint64_t get_aco_node_input_offset(aco_node* n){
    if(n->activation != NULL){
        return n->activation->input_array_offset;
    }
    else if(n->fcl != NULL){
        return n->fcl->input_array_offset;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->input_array_offset;
    }
    else if(n->cl != NULL){
        return n->cl->input_array_offset;
    }
    else if(n->lstm != NULL){
        return n->lstm->input_array_offset;
    }
    else if(n->concatenation != NULL){
        return n->concatenation->output_array_offset;
    }
    else if(n->sum != NULL){
        return n->sum->input_array_offset;
    }
    else if(n->pooling != NULL){
        return n->pooling->input_array_offset;
    }
    else if(n->splitter != NULL){
        return n->splitter->input_array_offset;
    }
}

uint64_t get_aco_node_output_offset(aco_node* n){
    if(n->activation != NULL){
        return n->activation->output_array_offset;
    }
    else if(n->fcl != NULL){
        return n->fcl->output_array_offset;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->output_array_offset;
    }
    else if(n->cl != NULL){
        return n->cl->output_array_offset;
    }
    else if(n->lstm != NULL){
        return n->lstm->output_h_array_offset;
    }
    else if(n->concatenation != NULL){
        return n->concatenation->output_array_offset;
    }
    else if(n->sum != NULL){
        return n->sum->output_array_offset;
    }
    else if(n->pooling != NULL){
        return n->pooling->output_array_offset;
    }
    else if(n->splitter != NULL){
        return n->splitter->output_array_offset;
    }
}

float* get_aco_node_output(aco_node* n){
    if(n->activation != NULL){
        return n->activation->output;
    }
    else if(n->fcl != NULL){
        return n->fcl->output;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->output;
    }
    else if(n->cl != NULL){
        return n->cl->output;
    }
    else if(n->lstm != NULL){
        return n->lstm->output_h;
    }
    else if(n->concatenation != NULL){
        return n->concatenation->output;
    }
    else if(n->sum != NULL){
        return n->sum->output;
    }
    else if(n->pooling != NULL){
        return n->pooling->output;
    }
    else if(n->splitter != NULL){
        return n->splitter->output;
    }
}

float* get_aco_node_output_error(aco_node* n){
    if(n->activation != NULL){
        return n->activation->output_error;
    }
    else if(n->fcl != NULL){
        return n->fcl->output_error;
    }
    else if(n->fcl_ka != NULL){
        return n->fcl_ka->output_error;
    }
    else if(n->cl != NULL){
        return n->cl->output_error;
    }
    else if(n->lstm != NULL){
        return n->lstm->output_h_error;
    }
    else if(n->concatenation != NULL){
        return n->concatenation->output_error;
    }
    else if(n->sum != NULL){
        return n->sum->output_error;
    }
    else if(n->pooling != NULL){
        return n->pooling->output_error;
    }
    else if(n->splitter != NULL){
        return n->splitter->output_error;
    }
}


void assign_aco_node_inputs(aco_node* n, float* input1, float* input2){
    if(n->activation != NULL){
        n->activation->input = input1;
    }
    else if(n->fcl != NULL){
        n->fcl->input = input1;
    }
    else if(n->fcl_ka != NULL){
        n->fcl_ka->input = input1;
    }
    else if(n->cl != NULL){
        n->cl->input = input1;
    }
    else if(n->lstm != NULL){
        n->lstm->input = input1;
    }
    else if(n->concatenation != NULL){
        n->concatenation->output = input1;
        int i;
        for(i = 0; i < n->n_output; i++){
            if(n->output[i]->sum == NULL || (n->output[i]->sum != NULL && n->output[i]->sum->input_array_offset == n->concatenation->output_array_offset))
                assign_aco_node_inputs(n->output[i],n->concatenation->output,NULL);
            else
                assign_aco_node_inputs(n->output[i],NULL,n->concatenation->output);
        }
    }
    else if(n->sum != NULL){
        if(input1 != NULL)
            n->sum->input = input1;
        if(input2 != NULL)
            n->sum->input2 = input2;
    }
    else if(n->pooling != NULL){
        n->pooling->input = input1;
    }
    
    else if(n->splitter != NULL){
        n->splitter->input = input1;
    }
}

void assign_aco_node_outputs(aco_node* n, float* output){
    if(n->activation != NULL){
        n->activation->output = output;
    }
    else if(n->fcl != NULL){
        n->fcl->output = output;
    }
    else if(n->fcl_ka != NULL){
        n->fcl_ka->output = output;
    }
    else if(n->cl != NULL){
        n->cl->output = output;
    }
    else if(n->lstm != NULL){
        n->lstm->output_h = output;
    }
    else if(n->sum != NULL){
        n->sum->output = output;
    }
    else if(n->pooling != NULL){
        n->pooling->output = output;
    }
    
    else if(n->splitter != NULL){
        n->splitter->output = output;
    }
}

void clip_aco_node(aco_node* n, float* array_to_clip, int n_parameters){
    aco_fcl_kolmogorov_arnold_clip(n->fcl_ka,array_to_clip,n_parameters);
}

void copy_aco_node_activation(aco_node* copy, aco_node* original){
    if(copy == NULL || original == NULL)
        return;
    copy_activation_flag(copy->activation,original->activation);
}

void merge_sort_aco_node(aco_node* c){
    if(c == NULL)
        return;
    merge_sort_aco_cl(c->cl);
    merge_sort_aco_fcl(c->fcl);
    merge_sort_aco_lstm(c->lstm);
}

void assign_aco_node_storing_partial_derivatives(aco_node* n, float* params){
    if(n == NULL)
        return;
    assign_aco_activation_storing_partial_derivatives(n->activation,params);
    assign_aco_cl_storing_partial_derivatives(n->cl, params);
    assign_aco_concatenation_storing_partial_derivatives(n->concatenation, params);
    assign_aco_lstm_storing_partial_derivatives(n->lstm, params);
    assign_aco_pooling_storing_partial_derivatives(n->pooling, params);
    assign_aco_sum_storing_partial_derivatives(n->sum, params);
    assign_aco_fcl_storing_partial_derivatives(n->fcl,params);
    assign_aco_splitter_storing_partial_derivatives(n->splitter,params);
}

float* get_aco_node_d_output(aco_node* node){
    float* ret = get_aco_activations_d_output(node->activation);
    if(ret != NULL)
        return ret;
    ret = get_aco_cl_d_output(node->cl);
    if(ret != NULL)
        return ret;
    ret = get_aco_concatenation_d_output(node->concatenation);
    if(ret != NULL)
        return ret;
    ret = get_aco_fcl_d_output(node->fcl);
    if(ret != NULL)
        return ret;
    ret = get_aco_lstm_d_output(node->lstm);
    if(ret != NULL)
        return ret;
    ret = get_aco_pooling_d_output(node->pooling);
    if(ret != NULL)
        return ret;
    ret = get_aco_sum_d_output(node->sum);
    
    if(ret != NULL)
        return ret;
    ret = get_aco_splitter_d_output(node->splitter);
    
    return ret;
}

void init_aco_node_scores(aco_node* node){
    if(node == NULL)
        return;
    init_cl_scores(node->cl);
    init_fcl_scores(node->fcl);
    init_lstm_scores(node->lstm);
}

void set_aco_node_k_percentage(aco_node* n, float k_percentage){
    if(n == NULL)
        return;
    set_cl_k_percentage(n->cl,k_percentage);
    set_fcl_k_percentage(n->fcl,k_percentage);
    set_lstm_k_percentage(n->lstm,k_percentage);
}

void set_aco_node_used_output(aco_node* n){
    if(n == NULL)
        return;
    set_aco_cl_used_output(n->cl);
    set_aco_fcl_used_output(n->fcl);
    set_aco_lstm_used_output(n->lstm);
}

void set_aco_node_used_output_no_parametric(aco_node* n,int* previous_used_outputs1,int* previous_used_outputs2, int is_root, uint64_t previous_used_outputs1_offset, uint64_t previous_used_outputs2_offset){
    if(n == NULL)    
        return;
    set_aco_activation_used_output(n->activation,previous_used_outputs1,is_root);
    set_aco_concatenation_used_output(n->concatenation,previous_used_outputs1,previous_used_outputs2,is_root, previous_used_outputs1_offset, previous_used_outputs2_offset);
    set_aco_pooling_used_output(n->pooling,previous_used_outputs1,is_root);
    set_aco_sum_used_output(n->sum,previous_used_outputs1,previous_used_outputs2,is_root);
    set_aco_splitter_used_output(n->splitter,previous_used_outputs1,is_root);
}

int* get_aco_node_used_outputs(aco_node* a){
    if(a == NULL)
        return NULL;
    int* ret = NULL;
    ret = get_aco_activation_used_outputs(a->activation);
    if(ret != NULL)
        return ret;
    ret = get_aco_cl_used_outputs(a->cl);
    if(ret != NULL)
        return ret;
    ret = get_aco_concatenation_used_outputs(a->concatenation);
    if(ret != NULL)
        return ret;
    ret = get_aco_fcl_used_outputs(a->fcl);
    if(ret != NULL)
        return ret;
    ret = get_aco_lstm_used_outputs(a->lstm);
    if(ret != NULL)
        return ret;
    ret = get_aco_pooling_used_outputs(a->pooling);
    if(ret != NULL)
        return ret;
    ret = get_aco_sum_used_outputs(a->sum);
    if(ret != NULL)
        return ret;
    ret = get_aco_splitter_used_outputs(a->splitter);
    return ret;
}


