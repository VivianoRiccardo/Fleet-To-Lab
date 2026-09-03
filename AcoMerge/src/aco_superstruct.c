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


aco_superstruct* init_aco_superstruct(aco_node** nodes,int** adjacency_matrix, int** same_params, int n_nodes){
    
    
    
    // if a node has 2 concatenation as output is a problem
    // if a lstm has more lstm as output is a problem
    // if a concatenation has more than 2 input is a problem
    // if a lstm has more than 1 lstm as input is a problem or if it has more than 1 non lstm is a problem
    // if a sum has more than 2 input is a problem
    // if any other has more than 1 input or more than 1 output is a problem
    
    int n_roots = 0;
    aco_superstruct* s = (aco_superstruct*)malloc(sizeof(aco_superstruct));
    merge_sort_aco_nodes(nodes,0,n_nodes-1);
    
    
    
    if(there_are_missing_nodes(nodes, n_nodes)){
        fprintf(stderr,"Error: there are missing nodes!\n"),
        exit(1);
    }
    

    
    
    int i, n_leaves = 0;
    
    for(i = 0; i < n_nodes; i++){
        if(after_splitter_there_is_splitter_or_concatenation(nodes[i])){
            fprintf(stderr,"Error: you cannot set a concatenation or a splitter after a splitter!\n"),
            exit(1);
        }
    }
    
    aco_node** roots = connect_nodes(nodes, adjacency_matrix,&n_roots,n_nodes);
    
    for(i = 0; i < n_roots; i++){
        if(roots[i]->splitter != NULL && roots[i]->splitter->index_offset > 0){
            fprintf(stderr,"Error: if you start with a splitter, no offset should be done!n"),
            exit(1);
        }
    }
    
    uint64_t storing_arrays_size = 0;
    float* arr = get_storing_arrays(roots, n_roots, &storing_arrays_size,nodes, n_nodes);
    
    uint64_t* array_sizes = (uint64_t*)malloc(sizeof(uint64_t));
    array_sizes[0] = storing_arrays_size;
    
    float** arrays = (float**)malloc(sizeof(float*));
    arrays[0] = arr;
    
    
    
    
    s->leaves = get_leaves(nodes, n_nodes, &n_leaves);
    s->n_leaves = n_leaves;
    s->adjacency_matrix = adjacency_matrix;
    s->array_sizes = array_sizes;
    s->arrays = arrays;
    s->aco_nodes = nodes;
    s->roots = roots;
    
    
    
    
    s->n_nodes = n_nodes;
    s->n_roots = n_roots;
    s->n_arrays = 1;
    s->same_params = same_params;
    s->alpha_aco = 1;
    s->beta_aco = 1;
    s->max_alpha_aco = 1;
    s->max_beta_aco = 1;
    s->min_alpha_aco = 1;
    s->min_beta_aco = 1;
    s->exponential_alpha_decay = 1;
    s->exponential_beta_decay = 1;
    s->use_heurisitc = 0;
    s->n_bit_arrays = 0;
    s->bit_arrays = NULL;
    s->losses = NULL;
    
    // levy flight
    s->levy_threshold = 0;
    s->altering_ratio = 0.4;
    
    //pso
    s->dt = 0.1;
    s->m = 0.2;
    s->gamma = 1-s->m;
    s->lambda1 = 0;
    s->lambda2 = 1;
    s->sigma1 = s->lambda1*sqrtf(0.4);
    s->sigma2 = sqrtf(0.4);
    s->alpha = 50;
    
    s->free_all_params = 1;

    
    merge_sort_aco_nodes(s->leaves,0,n_leaves-1);
    s->n_activation_functions = 0;
    for(i = 0; i < n_nodes; i++){
        if(s->aco_nodes[i]->activation != NULL)
            s->n_activation_functions++;
    }
    
    // edge popup
    s->scores_partial_derivative_index = -1;
    s->int_arrays = NULL;
    s->int_array_sizes = NULL;
    s->n_int_arrays = 0;
    s->only_for_inference = 0;
    
    
    return s;
    
}


void set_aco_superstruct_only_for_inference(aco_superstruct* s){
    s->only_for_inference = 1;
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[s->n_arrays-2];
    s->arrays[s->n_arrays-1] = s->arrays[s->n_arrays-2];
}

void aco_superstruct_set_all_params_to_null(aco_superstruct* s){
    if(s == NULL)
        return;
    s->arrays[2] = NULL;
    return;
}

void set_aco_superstruct_array_all_params(aco_superstruct* copy_from, aco_superstruct* copy_to, int index){
    if(copy_from == NULL || copy_to == NULL)
        return;
    copy_to->n_arrays++;
    copy_to->array_sizes = (uint64_t*)realloc(copy_to->array_sizes,copy_to->n_arrays*sizeof(uint64_t));
    copy_to->arrays = (float**)realloc(copy_to->arrays,copy_to->n_arrays*sizeof(float*));
    copy_to->array_sizes[copy_to->n_arrays-1] = copy_from->array_sizes[index];
    copy_to->arrays[copy_to->n_arrays-1] = copy_from->arrays[index];
    
}

void aco_superstruct_add_scores(aco_superstruct* s){
    float* params = (float*)calloc(s->array_sizes[1],sizeof(float));
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[1];
    s->arrays[s->n_arrays-1] = params;
}

void aco_superstruct_add_scores_partial_derivatives(aco_superstruct* s){
    float* params = (float*)calloc(s->array_sizes[1],sizeof(float));
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[1];
    s->arrays[s->n_arrays-1] = params;
}

void copy_array_weights_to_specific_struct_weight_position(aco_superstruct* s, float* array, int index, int size){
    int i, n_params = s->array_sizes[2]/s->array_sizes[1];
    for(i = 0; i < s->array_sizes[2]; i+=n_params){
        s->arrays[2][i+index] = array[(int)(i/n_params)];
    }
}

void copy_current_weight_combination_to_output_array(aco_superstruct* s, float* array){
    copy_array(s->arrays[1], array, s->array_sizes[1]);
}

float* get_current_weight_combination_to_output_array(aco_superstruct* s){
	float* array = (float*)calloc(s->array_sizes[1], sizeof(float));
    copy_array(s->arrays[1], array, s->array_sizes[1]);
    return array;
}

/*
void aco_superstruct_add_indices_array(aco_superstruct* s){
    int* params = (int*)calloc(s->array_sizes[1],sizeof(int));
    s->n_int_arrays++;
    s->int_array_sizes = (uint64_t*)realloc(s->int_array_sizes,s->n_int_arrays*sizeof(uint64_t));
    s->int_arrays = (int**)realloc(s->int_arrays,s->n_int_arrays*sizeof(int*));
    s->int_array_sizes[s->n_int_arrays-1] = s->array_sizes[1];
    s->int_arrays[s->n_int_arrays-1] = params;
}
*/

void aco_superstruct_add_storing_vector_partial_derivatives(aco_superstruct* s){
    float* params = (float*)calloc(s->array_sizes[0],sizeof(float));
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[0];
    s->arrays[s->n_arrays-1] = params;
}


void set_pheromone_for_index(aco_superstruct* s, int index, float pheromone){
    if(s == NULL || index < 0 || index > 2)
        return;
    s->bit_arrays[index]->pheromone = pheromone;
}

void copy_aco_superstruct_array(aco_superstruct* copy_from, aco_superstruct* copy_to, int index){
    if(copy_from == NULL || copy_to == NULL)
        return;
    copy_array(copy_from->arrays[index],copy_to->arrays[index],copy_from->array_sizes[index]);
}

void free_aco_superstruct(aco_superstruct* s){
    if(s == NULL)
        return;
    if(s->bit_arrays != NULL){
        free_BitArray(s->bit_arrays[0]);
        free_BitArray(s->bit_arrays[1]);
        free_BitArray(s->bit_arrays[2]);
        free_BitArray(s->bit_arrays[3]);
        free(s->bit_arrays);
    }
    
    free_matrix((void**)s->adjacency_matrix,s->n_nodes);
    free_matrix((void**)s->same_params,s->n_nodes);
    free_matrix((void**)s->arrays,s->only_for_inference?s->n_arrays-1:s->n_arrays);
    free_matrix((void**)s->int_arrays,s->n_int_arrays);
    free(s->array_sizes);
    free(s->int_array_sizes);
    int i;
    for(i = 0; i < s->n_nodes; i++){
        free_aco_node(s->aco_nodes[i]);
    }
    free(s->losses);
    free(s->leaves);
    free(s->aco_nodes);
    free(s->roots);
    free(s);
    
}

void aco_superstruct_activate_heuristic(aco_superstruct* s, double simulated_annealing_temperature, double alpha_factorization_simulated_annealing, double alpha_aco, double beta_aco, float value){
    s->use_heurisitc = 1;
    float* params = (float*)calloc(s->array_sizes[2],sizeof(float));
    set_vector_with_value(value,params,s->array_sizes[2]);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[2];
    s->arrays[s->n_arrays-1] = params;
    s->alpha_aco = alpha_aco;
    s->beta_aco = beta_aco;
}

void aco_superstruct_activate_heuristic_edges_type(aco_superstruct* s, double simulated_annealing_temperature, double alpha_factorization_simulated_annealing, double alpha_aco, double beta_aco, float value){
    s->use_heurisitc = 1;
    float* params = (float*)calloc(s->array_sizes[3],sizeof(float));
    set_vector_with_value(value,params,s->array_sizes[3]);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[3];
    s->arrays[s->n_arrays-1] = params;
    s->alpha_aco = alpha_aco;
    s->beta_aco = beta_aco;
}

void set_max_aco_alpha_beta(aco_superstruct* s, float max_alpha_aco, float max_beta_aco,float min_alpha_aco, float min_beta_aco,  float exponential_alpha_decay, float exponential_beta_decay){
    s->max_alpha_aco = max_alpha_aco;
    s->max_beta_aco = max_beta_aco;
    s->exponential_alpha_decay = exponential_alpha_decay;
    s->exponential_beta_decay = exponential_beta_decay;
}

void update_alpha_beta_params(aco_superstruct* s){
    s->alpha_aco*= exp(s->exponential_alpha_decay*((double)(s->iteration_index%s->number_iterations_for_single_aco_iteration)));
    s->beta_aco*= exp(s->exponential_beta_decay*((double)(s->iteration_index%s->number_iterations_for_single_aco_iteration)));
    if(s->alpha_aco > s->max_alpha_aco)
        s->alpha_aco = s->max_alpha_aco;
    if(s->beta_aco > s->max_beta_aco)
        s->beta_aco = s->max_beta_aco;
    if(s->alpha_aco < s->min_alpha_aco)
        s->alpha_aco = s->min_alpha_aco;
    if(s->beta_aco < s->min_beta_aco)
        s->beta_aco = s->min_beta_aco;
}

void set_alpha_and_beta_aco(aco_superstruct* s, float alpha_aco,float beta_aco){
    s->alpha_aco = alpha_aco;
    s->beta_aco = beta_aco;
}

void aco_superstruct_activate_explorative_supporting_heuristic_array(aco_superstruct* s){
    float* params = (float*)calloc(s->array_sizes[2],sizeof(float));
    set_vector_with_value(1.0,params,s->array_sizes[2]);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[2];
    s->arrays[s->n_arrays-1] = params;
}

void aco_increase_explorative_supporting_heuristic_array(aco_superstruct* s){
    int i,size = s->array_sizes[2]/s->array_sizes[1];
    for(i = 0; i < s->array_sizes[1]; i++){
        s->arrays[5][i*size+s->bit_arrays[0]->data[i]]++;
    }
}

void update_heuristics_according_to_explorative_supporting_heuristic_array(aco_superstruct* s){
    int i,j, size = s->array_sizes[2]/s->array_sizes[1];
    for(j = 0; j < s->array_sizes[1]; j++){
        float minimum = s->arrays[5][j*size], maximum = s->arrays[5][j*size]; 
        for(i = 1; i < size; i++){
            if(s->arrays[5][j*size+i] > maximum)
                maximum = s->arrays[5][j*size+i]; 
            if(s->arrays[5][j*size+i]< minimum)
                minimum = s->arrays[5][j*size+i]; 
        }
        if (minimum == maximum){
            set_vector_with_value(1.0,s->arrays[4]+j*size,size);
        }
        else{
            float maxmin = maximum-minimum;
            for(i = 0; i < size; i++){
                s->arrays[4][j*size+i] =  1 - (s->arrays[5][j*size+i]-minimum)/maxmin;
            }
        }
    }
}

void reset_heuristic_all_ones(aco_superstruct* s){
    set_vector_with_value(1.0,s->arrays[4],s->array_sizes[4]);
}

void reset_explorative_heuristics_support(aco_superstruct* s){
    set_vector_with_value(1.0,s->arrays[5],s->array_sizes[5]);
}

int output_is_relu(aco_node* n, int depth){
    int i, ret = 0;
    if(depth){
        if(n->activation != NULL && n->activation->activation_flag == RELU){
            return 1;
        }
        
        else if(n->activation != NULL && n->activation->activation_flag != RELU){
            return 0;
        }
        else if(n->fcl != NULL){
            return 0;
        }
        else if(n->fcl_ka != NULL){
            return 0;
        }
        else if(n->cl != NULL){
            return 0;
        }
        else if(n->lstm != NULL){
            return 0;
        }
        else if(n->sum != NULL){
            return 0;
        }
    }
    
    for(i = 0; i < n->n_output; i++){
        ret = output_is_relu(n->output[i],1);
        if(!ret)
            break;
    }
    return ret;
} 

int input_is_positive(aco_node* n, int depth, int is_always_positive_input){
    int i, ret = 0;
    if(depth){
        if(n->activation != NULL && (n->activation->activation_flag == RELU || n->activation->activation_flag == SIGMOID)){
            return 1;
        }
        
        else if(n->activation != NULL && n->activation->activation_flag != RELU && n->activation->activation_flag == SIGMOID){
            return 0;
        }
        else if(n->fcl != NULL){
            return 0;
        }
        else if(n->fcl_ka != NULL){
            return 0;
        }
        else if(n->cl != NULL){
            return 0;
        }
        else if(n->lstm != NULL){
            return 1;
        }
        else if(n->sum != NULL){
            return 0;
        }
        
        if(!n->n_input){
            if(is_always_positive_input)
                return 1;
            return 0;
        }
    }
    
    for(i = 0; i < n->n_input; i++){
        ret = input_is_positive(n->input[i],1,is_always_positive_input);
        if(!ret)
            break;
    }
    return ret;
}

int nodes_params_already_checked(aco_superstruct* s, int aco_node_index){
    int i;
    for(i = 0; i < aco_node_index; i++){
        if(s->same_params[i][aco_node_index])
            return 1;
    }
    return 0;
}

aco_superstruct* copy_aco_superstruct_with_only_storing_params(aco_superstruct* s){
    
    aco_node** nodes = (aco_node**)malloc(sizeof(aco_node*)*s->n_nodes);
    
    int** adjacency_matrix = (int**)malloc(sizeof(int*)*s->n_nodes);
    int** same_params = (int**)malloc(sizeof(int*)*s->n_nodes);
    
    int i;
    for(i = 0; i < s->n_nodes; i++){
        nodes[i] = copy_aco_node_without_input_and_output(s->aco_nodes[i]);
        adjacency_matrix[i] = (int*)calloc(s->n_nodes,sizeof(int));
        same_params[i] = (int*)calloc(s->n_nodes,sizeof(int));
        copy_int_array(s->adjacency_matrix[i],adjacency_matrix[i],s->n_nodes);
        copy_int_array(s->same_params[i],same_params[i],s->n_nodes);
    }
    
    
    aco_superstruct* copy = init_aco_superstruct_without_storing_arrays(nodes,adjacency_matrix, same_params, s->n_nodes);
    
    float* params = (float*)calloc(s->array_sizes[0],sizeof(float));
    
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_storing_params(nodes[i],params);
    }
    
    copy->n_arrays = 1;
    copy->arrays = (float**)malloc(sizeof(float*));
    copy->arrays[0] = params;
    copy->array_sizes = (uint64_t*)malloc(sizeof(uint64_t));
    copy->array_sizes[0] = s->array_sizes[0];
    copy->n_int_arrays = 0;
    copy->int_arrays = NULL;
    copy->int_array_sizes = NULL;
    copy->only_for_inference = 0;
    return copy;
    
}

void assign_aco_superstruct_storing_params_output_used(aco_superstruct* s, int* params){
    if(s == NULL || params == NULL)
        return;
    int i;
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_storing_params_output_used(s->aco_nodes[i],params);
    }
}

void aco_superstruct_restrict_lstm_p_value(aco_superstruct* s){
    if(s == NULL)
        return;
    uint64_t i, j,size = s->array_sizes[2]/s->array_sizes[1];
    for(i = 0; i < s->n_nodes; i++){
        if(s->aco_nodes[i]->lstm != NULL){
            uint64_t index_p = s->aco_nodes[i]->lstm->weights_array_offset+s->aco_nodes[i]->lstm->output_size*3*(s->aco_nodes[i]->lstm->output_size+s->aco_nodes[i]->lstm->input_size);
            uint64_t index_p2 = size+index_p;
            for(j = index_p; j < index_p2; j++){
                if(s->arrays[2][j] > 1)
                    s->arrays[2][j] = 1;
                if(s->arrays[2][j] < 0)
                    s->arrays[2][j] = 0;
            }
        }
    }
}

void aco_superstruct_clip(aco_superstruct* s, float* array_to_clip, int n_parameters){
    if(s == NULL)
        return;
    uint64_t i;
    for(i = 0; i < s->n_nodes; i++){
        clip_aco_node(s->aco_nodes[i],array_to_clip,n_parameters);
    }
}

void aco_superstruct_update_params_eiwoa(aco_superstruct* s){
    if(s == NULL)
        return;
    uint64_t i, size = s->array_sizes[2]/s->array_sizes[1];
    for(i = 0; i < s->array_sizes[2]; i+=size){
        eiwoa_run(s, s->arrays[2]+i, s->arrays[3]+i,s->bit_arrays[0]->data[((int)(i/size))], size);
    }
    aco_superstruct_restrict_lstm_p_value(s);
    aco_superstruct_clip(s,s->arrays[2],size);
    

}

void aco_superstruct_update_params_eiwoa_edges_type(aco_superstruct* s){
    if(s == NULL)
        return;
    uint64_t i, size = s->array_sizes[2]/s->array_sizes[1], size2 = size*size, val1;
    for(i = 0; i < s->array_sizes[2]; i+=size){
        if(!i)
            eiwoa_run(s, s->arrays[2]+i, s->arrays[3]+i,s->bit_arrays[0]->data[((int)(i/size))], size);
        else
            eiwoa_run_edges_type(s, s->arrays[2]+i, s->arrays[3]+i*size,s->bit_arrays[0]->data[((int)(i/size))], size);
    }
    aco_superstruct_restrict_lstm_p_value(s);
    aco_superstruct_clip(s,s->arrays[2],size);
    

}

void update_eiwoa_params(aco_superstruct* s){
    double temp = (((float)(s->iteration_index/s->number_iterations_for_single_aco_iteration))/((float)(s->max_iterations)));
    s->alpha_eiwoa = 2.0 - 2.0*temp;
    temp=temp*temp*temp;
    s->weight_eiwoa = 1.0 - 2.0*temp;
}

void eiwoa_run(aco_superstruct* s, float* values, float* pheromones,int path_best, int size){
    if(s == NULL)
        return;
    
    int i, j, index_best, index_loc;
    double sum = 0;
    int* sorted_parameters = NULL;
    float* temp_parameters = (float*)calloc(size,sizeof(float));
    copy_array(values,temp_parameters,size);
    // checking the best one
    for(index_best = 0, i = 1; i < size; i++){
        if(pheromones[i] > pheromones[index_best])
            index_best = i;
    }
    path_best = index_best;
    
    // updating the values
    for(i = 0; i < size; i++){
        index_loc = index_best;
        if(i == path_best)
            continue;
        float a = s->alpha_eiwoa*2*normalized_random()-s->alpha_eiwoa;
        float c = 2*normalized_random();
        float d = float_abs(values[index_best]-values[i]);
        if (normalized_random() < 0.5){
            if(float_abs(a) < 1){
                d = float_abs(c*values[index_best]-values[i]);
                values[i] = values[path_best]-s->weight_eiwoa*a*d;
            }
            
            else{
                
                if(sorted_parameters == NULL){
                    // sorting the pheromones
                    sorted_parameters = (int*)calloc(size,sizeof(int));
                    for(j = 0; j < size; j++){
                        sorted_parameters[j] = j;
                    }
                    
                    merge_sort(pheromones,sorted_parameters,0,size-1);

                    
                    
                }
                
                for(sum = 0, j = size-1 - (int)((size-1)*(s->percentage_of_elements)); j < size-1; j++){
                    if(j == i || j == path_best || pheromones[sorted_parameters[j]]<=pheromones[sorted_parameters[i]])
                        continue;
                    sum+=pow(pheromones[sorted_parameters[j]],1.0/s->softmax_temperature);
                }
                float ran = normalized_random();
                for(j = size-1-(int)((size-1)*s->percentage_of_elements); j < size-1; j++){
                    if(j == i || j == path_best || pheromones[sorted_parameters[j]]<=pheromones[sorted_parameters[i]])
                        continue;
                    float v = pow(pheromones[sorted_parameters[j]],1.0/s->softmax_temperature)/sum;
                    if(ran <= v){
                        index_loc = sorted_parameters[j];
                        break;
                    }
                    ran-=v;
                }
                
                
                d = float_abs(c*temp_parameters[index_loc]-values[i]);
                values[i] = temp_parameters[index_loc]-s->weight_eiwoa*a*d;
            }
            
            
        }
                   
        else{
            float value = normalized_random();
            if (normalized_random() < 0.5)
                value*=-1;
            values[i] = s->weight_eiwoa*d*exp(s->beta_eiwoa*value)*cos(2*PI*value) + values[index_best];
        }
        if(values[i] > s->v_max)
            values[i] = s->v_max;
        if(values[i] < s->v_min)
            values[i] = s->v_min;
        
    }
    
    free(sorted_parameters);
    free(temp_parameters);
}

void eiwoa_run_edges_type(aco_superstruct* s, float* values, float* pheromones2,int path_best, int size){
    if(s == NULL)
        return;
    
    int i, j, index_best, index_loc;
    double sum = 0;
    int* sorted_parameters = NULL;
    float* temp_parameters = (float*)calloc(size,sizeof(float));
    float* pheromones = (float*)calloc(size,sizeof(float));
    
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            pheromones[j]+=pheromones2[i*size+j];
        }
    }
    
    copy_array(values,temp_parameters,size);
    // checking the best one
    for(index_best = 0, i = 1; i < size; i++){
        if(pheromones[i] > pheromones[index_best])
            index_best = i;
    }
    path_best = index_best;
    
    // updating the values
    for(i = 0; i < size; i++){
        index_loc = index_best;
        if(i == path_best)
            continue;
        float a = s->alpha_eiwoa*2*normalized_random()-s->alpha_eiwoa;
        float c = 2*normalized_random();
        float d = float_abs(values[index_best]-values[i]);
        if (normalized_random() < 0.5){
            if(float_abs(a) < 1){
                d = float_abs(c*values[index_best]-values[i]);
                values[i] = values[path_best]-s->weight_eiwoa*a*d;
            }
            
            else{
                
                if(sorted_parameters == NULL){
                    // sorting the pheromones
                    sorted_parameters = (int*)calloc(size,sizeof(int));
                    for(j = 0; j < size; j++){
                        sorted_parameters[j] = j;
                    }
                    
                    merge_sort(pheromones,sorted_parameters,0,size-1);

                    
                    
                }
                
                for(sum = 0, j = size-1 - (int)((size-1)*(s->percentage_of_elements)); j < size-1; j++){
                    if(j == i || j == path_best || pheromones[sorted_parameters[j]]<=pheromones[sorted_parameters[i]])
                        continue;
                    sum+=pow(pheromones[sorted_parameters[j]],1.0/s->softmax_temperature);
                }
                float ran = normalized_random();
                for(j = size-1-(int)((size-1)*s->percentage_of_elements); j < size-1; j++){
                    if(j == i || j == path_best || pheromones[sorted_parameters[j]]<=pheromones[sorted_parameters[i]])
                        continue;
                    float v = pow(pheromones[sorted_parameters[j]],1.0/s->softmax_temperature)/sum;
                    if(ran <= v){
                        index_loc = sorted_parameters[j];
                        break;
                    }
                    ran-=v;
                }
                
                
                d = float_abs(c*temp_parameters[index_loc]-values[i]);
                values[i] = temp_parameters[index_loc]-s->weight_eiwoa*a*d;
            }
            
            
        }
                   
        else{
            float value = normalized_random();
            if (normalized_random() < 0.5)
                value*=-1;
            values[i] = s->weight_eiwoa*d*exp(s->beta_eiwoa*value)*cos(2*PI*value) + values[index_best];
        }
        if(values[i] > s->v_max)
            values[i] = s->v_max;
        if(values[i] < s->v_min)
            values[i] = s->v_min;
        
    }
    free(pheromones);
    free(sorted_parameters);
    free(temp_parameters);
}

aco_superstruct* init_aco_superstruct_without_storing_arrays(aco_node** nodes,int** adjacency_matrix, int** same_params, int n_nodes){
    
    
    
    // if a node has 2 concatenation as output is a problem
    // if a lstm has more lstm as output is a problem
    // if a concatenation has more than 2 input is a problem
    // if a lstm has more than 1 lstm as input is a problem or if it has more than 1 non lstm is a problem
    // if a sum has more than 2 input is a problem
    // if any other has more than 1 input or more than 1 output is a problem
    
    int n_roots = 0;
    aco_superstruct* s = (aco_superstruct*)malloc(sizeof(aco_superstruct));
    merge_sort_aco_nodes(nodes,0,n_nodes-1);
    
    
    int i, n_leaves = 0;
    
    aco_node** roots = connect_nodes(nodes, adjacency_matrix,&n_roots,n_nodes);    
    
    s->leaves = get_leaves(nodes, n_nodes, &n_leaves);
    s->n_leaves = n_leaves;
    s->adjacency_matrix = adjacency_matrix;
    s->array_sizes = NULL;
    s->arrays = NULL;
    s->aco_nodes = nodes;
    s->roots = roots;
    s->n_nodes = n_nodes;
    s->n_roots = n_roots;
    s->n_arrays = 0;
    s->same_params = same_params;
    s->bit_arrays = NULL;
    s->losses = NULL;
    merge_sort_aco_nodes(s->leaves,0,n_leaves-1);
    
    return s;
    
}



// the hyperparameters of kolmogorov arnold are set at the end of the array
void generate_pso_parameters_array_to_superstruct(aco_superstruct* s, int number_of_particles){
    if(s == NULL)
        return;
    
    int i,j;
    uint64_t sum = 0;
    for(i = 0; i < s->n_nodes; i++){
        sum += get_aco_node_params_size(s->same_params, s->aco_nodes[i], s->n_nodes,i);
    }
    
    
    float* params = (float*)calloc(sum*number_of_particles,sizeof(float));
    
    for(j = 0; j < number_of_particles; j++){
        uint64_t sum2 = 0;
        
        for(i = 0; i < s->n_nodes; i++){
            assign_array_aco_node_params(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params+j*sum,s->n_nodes,i);
        }
        for(i = 0; i < s->n_nodes; i++){
            assign_array_aco_node_params_only_for_kolmogorov_arnold_ts(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params+j*sum,s->n_nodes,i);
        }
    }
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = sum*number_of_particles;
    s->arrays[s->n_arrays-1] = params;
    s->losses = (double*)calloc(number_of_particles,sizeof(double));
}
void generate_parameters_array_to_superstruct(aco_superstruct* s){
    if(s == NULL)
        return;
    
    int i;
    uint64_t sum = 0;
    for(i = 0; i < s->n_nodes; i++){
        sum += get_aco_node_params_size(s->same_params, s->aco_nodes[i], s->n_nodes,i);
    }
    
    
    float* params = (float*)calloc(sum,sizeof(float));
    
    uint64_t sum2 = 0;
    
    for(i = 0; i < s->n_nodes; i++){
        assign_array_aco_node_params(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params,s->n_nodes,i);
    }
    for(i = 0; i < s->n_nodes; i++){
        assign_array_aco_node_params_only_for_kolmogorov_arnold_ts(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params,s->n_nodes,i);
    }
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = sum;
    s->arrays[s->n_arrays-1] = params;
    
}

void aco_superstruct_select_params_according_to_pheromone_exploration(aco_superstruct* s){
    if(s == NULL)    
        return;
    uint64_t number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    uint64_t i,j,count,count2;
    int index;
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        if(s->iteration_index == 0){
            if(s->use_heurisitc)
                index = get_parameter_index_according_to_aco_with_heuristic(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco, number_of_parameters);
            else
                index = get_parameter_index_according_to_aco(s->arrays[3]+i, number_of_parameters);
        }
        else{
            if(random_normal() <= s->current_p)    
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+i, number_of_parameters);
            else{
                if(s->use_heurisitc)
                    index = get_parameter_index_according_to_aco_with_heuristic(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco, number_of_parameters);
                else
                    index = get_parameter_index_according_to_aco(s->arrays[3]+i, number_of_parameters);
            }
        }
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
    }
    
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                if(s->iteration_index == 0){
                    if(s->use_heurisitc)
                        index = get_parameter_index_according_to_aco_with_heuristic(s->arrays[3]+s->array_sizes[2]+i,s->arrays[4]+s->array_sizes[2]+i,s->alpha_aco,s->beta_aco, s->aco_nodes[j]->activation->activations_size);
                    else
                        index = get_parameter_index_according_to_aco(s->arrays[3]+s->array_sizes[2]+i, s->aco_nodes[j]->activation->activations_size);
                }
                else{
                    if(random_normal() <= s->current_p)    
                        index = get_parameter_index_according_to_aco_best(s->arrays[3]+s->array_sizes[2]+i, s->aco_nodes[j]->activation->activations_size);
                    else{
                        if(s->use_heurisitc)
                            index = get_parameter_index_according_to_aco_with_heuristic(s->arrays[3]+s->array_sizes[2]+i,s->arrays[4]+s->array_sizes[2]+i,s->alpha_aco,s->beta_aco, s->aco_nodes[j]->activation->activations_size);
                        else
                            index = get_parameter_index_according_to_aco(s->arrays[3]+s->array_sizes[2]+i,s->aco_nodes[j]->activation->activations_size);
                    }
                }
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1] + count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
    set_vector_according_to_indices(s->arrays[2], s->arrays[1], s->bit_arrays[2]->data, s->array_sizes[2], s->array_sizes[1]);

}

void aco_superstruct_select_params_according_to_index_pso(aco_superstruct* s, int index){
    s->arrays[1] = s->arrays[2]+(index*s->array_sizes[1]);    
    assign_aco_superstruct_learnable_params(s);
}

void aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight(aco_superstruct* s){
    if(s == NULL)    
        return;
    uint64_t number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    uint64_t i,j,count,count2;
    int index;
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        if(s->iteration_index == 0){
            if(s->use_heurisitc)
                index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
            else
                index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+i,s->levy_threshold,s->altering_ratio, number_of_parameters);
        }
        else{
            if(random_normal() <= s->current_p)    
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+i, number_of_parameters);
            else{
                if(s->use_heurisitc)
                    index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
                else
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+i,s->levy_threshold,s->altering_ratio, number_of_parameters);
            }
        }
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
    }
    
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                if(s->iteration_index == 0){
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[2]+i,s->levy_threshold,s->altering_ratio, s->aco_nodes[j]->activation->activations_size);
                }
                else{
                    if(random_normal() <= s->current_p)    
                        index = get_parameter_index_according_to_aco_best(s->arrays[3]+s->array_sizes[2]+i, s->aco_nodes[j]->activation->activations_size);
                    else{
                        index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[2]+i,s->levy_threshold,s->altering_ratio,s->aco_nodes[j]->activation->activations_size);
                    }
                }
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1] + count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
    set_vector_according_to_indices(s->arrays[2], s->arrays[1], s->bit_arrays[2]->data, s->array_sizes[2], s->array_sizes[1]);

}

void aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight_edges_type(aco_superstruct* s){
    if(s == NULL)    
        return;
    uint64_t number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    uint64_t number_of_parameters_exp = number_of_parameters*number_of_parameters;
    uint64_t i,j,count,count2, val1;
    int index = 0;
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        val1 = ((uint64_t)((i/number_of_parameters)*number_of_parameters_exp+index*number_of_parameters));
        if(s->iteration_index == 0){
            if(s->use_heurisitc){
                index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+val1,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
            }
            else
                index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+val1,s->levy_threshold,s->altering_ratio, number_of_parameters);
        }
        else{
            if(random_normal() <= s->current_p)    
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+val1, number_of_parameters);
            else{
                if(s->use_heurisitc){
                    index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+val1,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
                
                }
                else
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+val1,s->levy_threshold,s->altering_ratio, number_of_parameters);
            }
        }
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
          
    }
    
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                if(s->iteration_index == 0){
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[1]*number_of_parameters_exp+i,s->levy_threshold,s->altering_ratio, s->aco_nodes[j]->activation->activations_size);
                }
                else{
                    if(random_normal() <= s->current_p)    
                        index = get_parameter_index_according_to_aco_best(s->arrays[3]+s->array_sizes[1]*number_of_parameters_exp+i, s->aco_nodes[j]->activation->activations_size);
                    else{
                        index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[1]*number_of_parameters_exp+i,s->levy_threshold,s->altering_ratio,s->aco_nodes[j]->activation->activations_size);
                    }
                }
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1] + count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
    set_vector_according_to_indices(s->arrays[2], s->arrays[1], s->bit_arrays[2]->data, s->array_sizes[2], s->array_sizes[1]);

}

void aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight_without_setting_weights(aco_superstruct* s){
    if(s == NULL)    
        return;
    uint64_t number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    uint64_t i,j,count,count2;
    int index;
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        if(s->iteration_index == 0){
            if(s->use_heurisitc)
                index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
            else
                index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+i,s->levy_threshold,s->altering_ratio, number_of_parameters);
        }
        else{
            if(random_normal() <= s->current_p)    
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+i, number_of_parameters);
            else{
                if(s->use_heurisitc)
                    index = get_parameter_index_according_to_aco_with_heuristic_levy_flight(s->arrays[3]+i,s->arrays[4]+i,s->alpha_aco,s->beta_aco,s->levy_threshold,s->altering_ratio, number_of_parameters);
                else
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+i,s->levy_threshold,s->altering_ratio, number_of_parameters);
            }
        }
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
    }
    
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                if(s->iteration_index == 0){
                    index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[2]+i,s->levy_threshold,s->altering_ratio, s->aco_nodes[j]->activation->activations_size);
                }
                else{
                    if(random_normal() <= s->current_p)    
                        index = get_parameter_index_according_to_aco_best(s->arrays[3]+s->array_sizes[2]+i, s->aco_nodes[j]->activation->activations_size);
                    else{
                        index = get_parameter_index_according_to_aco_levy_flight(s->arrays[3]+s->array_sizes[2]+i,s->levy_threshold,s->altering_ratio,s->aco_nodes[j]->activation->activations_size);
                    }
                }
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1] + count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
}


void aco_superstruct_select_params_according_to_best_pheromone(aco_superstruct* s){
    if(s == NULL)    
        return;
    int number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    int i,j,count,count2;
    int index = 0;
    uint64_t val1;
    reset_BitArray(s->bit_arrays[2]);
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        index = get_parameter_index_according_to_aco_best(s->arrays[3]+i, number_of_parameters);
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
    }
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+(s->array_sizes[2])+i, s->aco_nodes[j]->activation->activations_size);
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1]+count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
    
    set_vector_according_to_indices(s->arrays[2], s->arrays[1], s->bit_arrays[2]->data, s->array_sizes[2], s->array_sizes[1]);
}

void aco_superstruct_select_params_according_to_best_pheromone_edges_type(aco_superstruct* s){
    if(s == NULL)    
        return;
    int number_of_parameters = s->array_sizes[2]/s->array_sizes[1], number_of_parameters_exp = number_of_parameters*number_of_parameters ;
    int i,j,count,count2;
    int index = 0;
    uint64_t val1;
    reset_BitArray(s->bit_arrays[2]);
    for(i = 0; i < s->array_sizes[2]; i+=number_of_parameters){
        val1 = ((uint64_t)((i/number_of_parameters)*number_of_parameters_exp+index*number_of_parameters));
        index = get_parameter_index_according_to_aco_best(s->arrays[3]+val1, number_of_parameters);
        s->bit_arrays[2]->data[((int)(i/number_of_parameters))] = index;
    }
    for(count2 = 0, count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                index = get_parameter_index_according_to_aco_best(s->arrays[3]+(s->array_sizes[1]*number_of_parameters_exp)+i, s->aco_nodes[j]->activation->activations_size);
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index];
                s->bit_arrays[2]->data[s->array_sizes[1]+count2] = index;
                count = j+1;
                count2++;
                break;
            }
        }
    }
    
    set_vector_according_to_indices(s->arrays[2], s->arrays[1], s->bit_arrays[2]->data, s->array_sizes[2], s->array_sizes[1]);
}

void aco_superstruct_select_params_according_to_bit_array(aco_superstruct* s, int index){
    
    uint64_t number_of_parameters = s->array_sizes[2]/s->array_sizes[1];
    uint64_t i,j,count;
    int index2;
    for(i =0; i < s->array_sizes[2]; i+=number_of_parameters){
        index2 = convertToNumbers(s->bit_arrays[index], ((i)/number_of_parameters));
        s->arrays[1][i/number_of_parameters] = s->arrays[2][i + index2];
    }
    
    for(count = 0, i = 0; i < s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0; i+=N_ACTIVATION_FUNCTIONS_1_0){
        for(j = count; j < s->n_nodes; j++){
            if(s->aco_nodes[j]->activation != NULL && s->aco_nodes[j]->activation->activations != NULL){
                index2 = convertToNumbers(s->bit_arrays[index], s->array_sizes[1]+((i)/N_ACTIVATION_FUNCTIONS_1_0));
                s->aco_nodes[j]->activation->activation_flag = s->aco_nodes[j]->activation->activations[index2];
                count = j+1;
                break;
            }
        }
    }
}


int get_parameter_index_according_to_aco(float* pheromones, int size){
    int i, index;
    float val = normalized_random(), val2;
    

    double sum = sum_over_input(pheromones,size);
    val = normalized_random();
    
    for(i = 0; i < size; i++){
        val2 = pheromones[i]/sum;
        if(val <= val2){
            return i;
        }
        val-=val2;
    }
    return size - 1;
    
}

int get_parameter_index_according_to_aco_with_heuristic(float* pheromones,float* heuristics,double alpha, double beta, int size){
    int i, index;
    float val = normalized_random(), val2;
    float* temps = (float*)calloc(size,sizeof(float));
    for(i = 0; i < size; i++){
        if(alpha == 1.0 && beta == 1.0){
            temps[i] = pheromones[i]*heuristics[i];
        }
        else
            temps[i] = pow(pheromones[i],alpha)*pow(heuristics[i],beta);
    }

    double sum = sum_over_input(temps,size);
    val = normalized_random();
    for(i = 0; i < size; i++){
        val2 = temps[i]/sum;
        if(val <= val2){
            free(temps);
            return i;
        }
        val-=val2;
    }
    free(temps);
    return size - 1;
    
}

int get_parameter_index_according_to_aco_with_heuristic_levy_flight(float* pheromones,float* heuristics,double alpha, double beta,double p_threshold,double a, int size){
    int i, index;
    float val = normalized_random(), val2;
    float* temps = (float*)calloc(size,sizeof(float));
    for(i = 0; i < size; i++){
        if(alpha == 1.0 && beta == 1.0){
            temps[i] = pheromones[i]*heuristics[i];
        }
        else
            temps[i] = pow(pheromones[i],alpha)*pow(heuristics[i],beta);
    }

    double sum = sum_over_input(temps,size), p_levy = normalized_random();
    val = normalized_random();
    if(p_levy >= p_threshold){
        for(i = 0; i < size; i++){
            val2 = temps[i]/sum;
            if(val <= val2){
                free(temps);
                return i;
            }
            val-=val2;
        }
    }
    else{
        double t1 = (a*(1.0-p_levy)/(1.0-p_threshold));
        for(i = 0; i < size; i++){
            val2 = 1.0 - (t1*(1.0-temps[i]/sum));
            if(val <= val2){
                free(temps);
                return i;
            }
            val-=val2;
        }
    }
    free(temps);
    return size - 1;
    
}


int get_parameter_index_according_to_aco_levy_flight(float* pheromones,double p_threshold,double a, int size){
    int i, index;
    float val = normalized_random(), val2;
    float* temps = (float*)calloc(size,sizeof(float));
    for(i = 0; i < size; i++){
        temps[i] = pheromones[i];
    }

    double sum = sum_over_input(temps,size), p_levy = normalized_random();
    val = normalized_random();
    if(p_levy >= p_threshold){
        for(i = 0; i < size; i++){
            val2 = temps[i]/sum;
            if(val <= val2){
                free(temps);
                return i;
            }
            val-=val2;
        }
    }
    else{
        double t1 = (a*(1.0-p_levy)/(1.0-p_threshold));
        for(i = 0; i < size; i++){
            val2 = 1.0 - (t1*(1.0-temps[i]/sum));
            if(val <= val2){
                free(temps);
                return i;
            }
            val-=val2;
        }
    }
    free(temps);
    return size - 1;
    
}


int get_parameter_index_according_to_aco_best(float* pheromones, int size){
    int i, index;
    float val = normalized_random(), val2;
    
    val = pheromones[0];
    index = 0;
    for(i = 1; i < size; i++){
        if(pheromones[i] > val){
            val = pheromones[i];
            index = i;
        }
    }
    return index;
}

void aco_superstruct_set_best_local_network_pheromone(aco_superstruct* s, float pheromone){
    s->bit_arrays[1]->pheromone = pheromone;
}

void aco_superstruct_set_best_best_network_pheromone(aco_superstruct* s, float pheromone){
    s->bit_arrays[0]->pheromone = pheromone;
}

void aco_superstruct_update_taus(aco_superstruct* s){
    if((s->bit_arrays[0]->pheromone) > s->tau_max || s->iteration_index == 2){
        s->tau_max = s->bit_arrays[0]->pheromone;
        s->tau_min = s->tau_max*(1-s->p_dec)/((s->average-1)*(-s->current_p+s->p_dec));
        if(s->tau_min > s->tau_max)
            s->tau_min = s->tau_max;
    }
}

void update_current_p(aco_superstruct* s){
    s->current_p = ((((float)(s->iteration_index))-s->k_zero)/((double)(s->number_of_iterations)))*s->rho + s->rho/2.0;
    if(s->current_p > 0.99)
        s->current_p = 0.99;
}

void reset_pheromone_according_to_index(aco_superstruct* s, int index){
    s->bit_arrays[index]->pheromone = 0;
}

void init_aco_superstruct_general_stuff(aco_superstruct* s,int number_iterations_for_single_aco_iteration,int number_of_iterations,int max_iterations,int n_ants, float tau_min,float  tau_max,float  p_dec, float p, float rho, int time_to_update_with_global_best, float k_zero){
    if(s == NULL)
        return;
    s->number_of_iterations = number_of_iterations;
    s->max_iterations = max_iterations;
    s->n_ants = n_ants;
    s->tau_min = tau_min;
    s->tau_max = tau_max;
    s->p_dec = p_dec;
    s->p = p;
    s->rho = rho;
    s->k_zero = k_zero;
    s->current_p = rho;
    s->time_to_update_with_global_best = time_to_update_with_global_best;
    s->number_iterations_for_single_aco_iteration = number_iterations_for_single_aco_iteration;
    s->pheromone_best_trail = 0;
    s->iteration_index = 0;
    s->average = (((double)(s->array_sizes[2])))/((double)(s->array_sizes[1]));
    
}

void set_pheromone_best_trail(aco_superstruct* s, double pheromone){
    s->pheromone_best_trail = pheromone;
}

void set_local_pheromone(aco_superstruct* s, double pheromone){
    if(s == NULL)    
        return;
    s->bit_arrays[2]->pheromone = pheromone;
}

void check_local_best_pheromone(aco_superstruct* s){
    if(s->bit_arrays[2]->pheromone > s->bit_arrays[1]->pheromone){
        BitArray* temp = s->bit_arrays[2];
        s->bit_arrays[2] = s->bit_arrays[1];
        s->bit_arrays[1] = temp;
    }
}

void check_local_best_pheromone_for_gpu(aco_superstruct* s){
    if(s->bit_arrays[2]->pheromone > s->bit_arrays[1]->pheromone){
        copy_uint8_t_array(s->bit_arrays[2]->data, s->bit_arrays[1]->data, s->bit_arrays[2]->size);
        s->bit_arrays[1]->pheromone = s->bit_arrays[2]->pheromone;
    }
}

void check_global_best_pheromone(aco_superstruct* s){
    if(s->bit_arrays[1]->pheromone > s->bit_arrays[0]->pheromone){
        copy_uint8_t_array(s->bit_arrays[1]->data,s->bit_arrays[0]->data,s->bit_arrays[0]->size);
        s->bit_arrays[0]->pheromone = s->bit_arrays[1]->pheromone;
        if(s->use_heurisitc){
            int i,j,size = s->array_sizes[2]/s->array_sizes[1];
            for(i = 0; i < s->array_sizes[1]; i++){
                float max = 0;
                for(j = 0; j < size; j++){
                    float t = s->arrays[2][i*size+s->bit_arrays[0]->data[i]]-s->arrays[2][i*size+j];
                    t*=t;
                    s->arrays[4][i*size+j] = exp(-t);
                    if(s->arrays[4][i*size+j] < 0.1)
                        s->arrays[4][i*size+j] = 0.1;
                }
            }
        }
    }
}

void check_global_best_pheromone_without_updating_heuristic(aco_superstruct* s){
    if(s->bit_arrays[1]->pheromone > s->bit_arrays[0]->pheromone){
        copy_uint8_t_array(s->bit_arrays[1]->data,s->bit_arrays[0]->data,s->bit_arrays[0]->size);
        s->bit_arrays[0]->pheromone = s->bit_arrays[1]->pheromone;
    }
}

int* get_indices_from_local_best(aco_superstruct* s){
	int* a = (int*)calloc(s->bit_arrays[0]->size, sizeof(int));
	int i;
	for(i = 0; i < s->bit_arrays[1]->size; i++){
		a[i] = s->bit_arrays[1]->data[i];
	}
	return a;
}

int get_indices_size(aco_superstruct* s){
	return s->bit_arrays[1]->size;
}


void update_pheromone_according_to_path(float p, float tau_max, float tau_min,double pheromone_amount, BitArray* path, float* pheromones, uint64_t number_of_parameters, uint64_t size, uint64_t offset){
    if(path == NULL)    
        return;
    uint64_t i,j;
    int index;
    for(i =0; i < size; i+=number_of_parameters){
        index = path->data[offset+((int)(i/number_of_parameters))];
        for(j = 0; j < number_of_parameters; j++){
            pheromones[i+j]*=p;
            
        }
        pheromones[i+index] += (1-p)*pheromone_amount;
        
        for(j =0; j < number_of_parameters; j++){
            if(pheromones[i+j] > tau_max)
                pheromones[i+j] = tau_max;
            if(pheromones[i+j] < tau_min)
                pheromones[i+j] = tau_min;
        }
        
        
    }
}

void update_pheromone_according_to_path_edges_type(float p, float tau_max, float tau_min,double pheromone_amount, BitArray* path, float* pheromones, uint64_t number_of_parameters, uint64_t size, uint64_t offset){
    if(path == NULL)    
        return;
    uint64_t i,j, val1, number_of_parameters_exp = number_of_parameters*number_of_parameters;
    int index = 0;
    for(i =0; i < size; i+=number_of_parameters){
        val1 = ((uint64_t)((i/number_of_parameters)*number_of_parameters_exp+index*number_of_parameters));
        index = path->data[offset+((int)(i/number_of_parameters))];
        for(j = 0; j < number_of_parameters_exp; j++){
            pheromones[i+j]*=p;
            
        }
        pheromones[val1+index] += (1-p)*pheromone_amount;
        
        for(j =0; j < number_of_parameters_exp; j++){
            if(pheromones[i+j] > tau_max)
                pheromones[i+j] = tau_max;
            if(pheromones[i+j] < tau_min)
                pheromones[i+j] = tau_min;
        }
        
        
    }
}

void update_alpha_aco_beta_aco(aco_superstruct* s, double A, double B, double current_iteration, double total_iterations){
    s->alpha_aco = A + cos((normalized_random() + current_iteration + PI)/(2.0*total_iterations));
    s->beta_aco = B + sin((normalized_random() + current_iteration + PI)/(2.0*total_iterations));
} 

void update_heuristics(aco_superstruct* s){
    int i,j, size = s->array_sizes[2]/s->array_sizes[1];
    for(j = 0; j < s->array_sizes[1]; j++){
        for(i = 0; i < size; i++){
            if(s->bit_arrays[0]->data[j] == i)
                s->arrays[4][j*size+i] = 1;
            else
                s->arrays[4][j*size+i] = exp(-float_abs(s->arrays[2][j*size+i]-s->arrays[2][j*size+s->bit_arrays[0]->data[j]]));
        }
    }
}

void update_pheromone_according_to_local_best(aco_superstruct* s){
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[1]->pheromone, s->bit_arrays[1], s->arrays[3], s->array_sizes[2]/s->array_sizes[1], s->array_sizes[2],0);
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[1]->pheromone, s->bit_arrays[1], s->arrays[3]+s->array_sizes[2], N_ACTIVATION_FUNCTIONS_1_0, s->array_sizes[3]-s->array_sizes[2],s->array_sizes[1]);
}

void update_pheromone_according_to_local_best_edges_type(aco_superstruct* s){
    update_pheromone_according_to_path_edges_type(s->p, s->tau_max, s->tau_min,s->bit_arrays[1]->pheromone, s->bit_arrays[1], s->arrays[3], s->array_sizes[2]/s->array_sizes[1], s->array_sizes[2],0);
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[1]->pheromone, s->bit_arrays[1], s->arrays[3]+s->array_sizes[2]*s->array_sizes[2]/s->array_sizes[1], N_ACTIVATION_FUNCTIONS_1_0, s->array_sizes[3]-s->array_sizes[2]*s->array_sizes[2]/s->array_sizes[1],s->array_sizes[1]);
}

void update_pheromone_according_to_global_best(aco_superstruct* s){
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[0]->pheromone, s->bit_arrays[0], s->arrays[3], s->array_sizes[2]/s->array_sizes[1], s->array_sizes[2],0);
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[0]->pheromone, s->bit_arrays[0], s->arrays[3]+s->array_sizes[2], N_ACTIVATION_FUNCTIONS_1_0, s->array_sizes[3]-s->array_sizes[2],s->array_sizes[1]);
}


void update_pheromone_according_to_global_best_edges_type(aco_superstruct* s){
    update_pheromone_according_to_path_edges_type(s->p, s->tau_max, s->tau_min,s->bit_arrays[0]->pheromone, s->bit_arrays[0], s->arrays[3], s->array_sizes[2]/s->array_sizes[1], s->array_sizes[2],0);
    update_pheromone_according_to_path(s->p, s->tau_max, s->tau_min,s->bit_arrays[0]->pheromone, s->bit_arrays[0], s->arrays[3]+s->array_sizes[2]*s->array_sizes[2]/s->array_sizes[1], N_ACTIVATION_FUNCTIONS_1_0, s->array_sizes[3]-s->array_sizes[2]*s->array_sizes[2]/s->array_sizes[1],s->array_sizes[1]);
}

void init_aco_superstruct_eiwoa_stuff(aco_superstruct* s, float percentage_of_elements,float alpha_eiwoa,float  beta_eiwoa, float weight_eiwoa, float softmax_temperature, float v_max, float v_min){
    if(s == NULL)
        return;
    s->percentage_of_elements = percentage_of_elements;
    s->alpha_eiwoa = alpha_eiwoa;
    s->beta_eiwoa = beta_eiwoa;
    s->weight_eiwoa = weight_eiwoa;
    s->softmax_temperature = softmax_temperature;
    s->v_max = v_max;
    s->v_min = v_min;
    
}

void save_weights_single_model(aco_superstruct* sp, int n){
    if(n<0)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* s2 = (char*)malloc(sizeof(char)*256);

    s2[0] = '.';
    s2[1] = '/';
    s2[2] = 'w';
    s2[3] = 'e';
    s2[4] = 'i';
    s2[5] = 'g';
    s2[6] = 'h';
    s2[7] = 't';
    s2[8] = 's';
    s2[9] = '_';
    s2[10] = '\0';
    
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    s2 = strcat(s2,s);
    
    free(s);
    
    fw = fopen(s2,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s2);
        exit(1);
    }
    
    convert_data(sp->arrays[1],sizeof(float),sp->array_sizes[1]);
    i = fwrite(sp->arrays[1],sizeof(float)*sp->array_sizes[1],1,fw);
    convert_data(sp->arrays[1],sizeof(float),sp->array_sizes[1]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving weights\n");
        exit(1);
    }
    
    fclose(fw);
    
    free(s2);
}

void load_weights_single_model(aco_superstruct* sp, char* filename){
    
    FILE* fr;
    fr = fopen(filename,"r");
    int i;
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",filename);
        exit(1);
    }
    
    
    i = fread(sp->arrays[1],sizeof(float)*sp->array_sizes[1],1,fr);
    convert_data(sp->arrays[1],sizeof(float),sp->array_sizes[1]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading weights\n");
        exit(1);
    }
    
    fclose(fr);
}


void save_weights_and_pheromone_aco_superstruct(aco_superstruct* sp, int n){
    if(n<0)
        return;
    int i;
    FILE* fw;
    char* s = (char*)malloc(sizeof(char)*256);
    char* s1 = (char*)malloc(sizeof(char)*256);
    char* s2 = (char*)malloc(sizeof(char)*256);
    
    s1[0] = '.';
    s1[1] = '/';
    s1[2] = 'p';
    s1[3] = 'h';
    s1[4] = 'e';
    s1[5] = 'r';
    s1[6] = 'm';
    s1[7] = 'o';
    s1[8] = 'n';
    s1[9] = 'e';
    s1[10] = '_';
    s1[11] = '\0';
    
    s2[0] = '.';
    s2[1] = '/';
    s2[2] = 'w';
    s2[3] = 'e';
    s2[4] = 'i';
    s2[5] = 'g';
    s2[6] = 'h';
    s2[7] = 't';
    s2[8] = 's';
    s2[9] = '_';
    s2[10] = '\0';
    
    char* t = ".bin";
    s = itoa_n(n,s);
    s = strcat(s,t);
    s1 = strcat(s1,s);
    s2 = strcat(s2,s);
    
    free(s);
    
    fw = fopen(s1,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s1);
        exit(1);
    }
    
    convert_data(sp->arrays[2],sizeof(float),sp->array_sizes[2]);
    i = fwrite(sp->arrays[2],sizeof(float)*sp->array_sizes[2],1,fw);
    convert_data(sp->arrays[2],sizeof(float),sp->array_sizes[2]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving weights\n");
        exit(1);
    }
    
    fclose(fw);
    
    fw = fopen(s2,"a+");
    
    if(fw == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",s2);
        exit(1);
    }
    
    convert_data(sp->arrays[3],sizeof(float),sp->array_sizes[3]);
    i = fwrite(sp->arrays[3],sizeof(float)*sp->array_sizes[3],1,fw);
    convert_data(sp->arrays[3],sizeof(float),sp->array_sizes[3]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred saving pheromones\n");
        exit(1);
    }
    
    fclose(fw);
    
    free(s1);
    free(s2);
}


void load_weights_and_pheromone_aco_superstruct(aco_superstruct* sp, char* pheromone, char* weights){
    int i;
    FILE* fr;
    
    
    
    fr = fopen(pheromone,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",pheromone);
        exit(1);
    }
    
    
    i = fread(sp->arrays[2],sizeof(float)*sp->array_sizes[2],1,fr);
    convert_data(sp->arrays[2],sizeof(float),sp->array_sizes[2]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading weights\n");
        exit(1);
    }
    
    fclose(fr);
    
    fr = fopen(weights,"r");
    
    if(fr == NULL){
        fprintf(stderr,"Error: error during the opening of the file %s\n",pheromone);
        exit(1);
    }
    
    
    
    i = fread(sp->arrays[3],sizeof(float)*sp->array_sizes[3],1,fr);
    convert_data(sp->arrays[3],sizeof(float),sp->array_sizes[3]);
    
    if(i != 1){
        fprintf(stderr,"Error: an error occurred loading weights\n");
        exit(1);
    }
    
    fclose(fr);
}

void generate_aco_parameters_array_to_superstruct(aco_superstruct* s, int number_of_parameters){
    if(s == NULL)
        return;
    
    int i;
    uint64_t sum = 0;
    for(i = 0; i < s->n_nodes; i++){
        sum += get_aco_node_params_size(s->same_params, s->aco_nodes[i], s->n_nodes,i);
    }
    
    sum*=number_of_parameters;
    float* params = (float*)calloc(sum,sizeof(float));
    uint64_t sum2 = 0;
    
    for(i = 0; i < s->n_nodes; i++){
        generate_array_aco_node_params(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params,s->n_nodes,i,number_of_parameters,params);
    }
    for(i = 0; i < s->n_nodes; i++){
        generate_array_aco_node_params_only_for_kolmogorov_arnold_ts(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params,s->n_nodes,i,number_of_parameters,params);
    }
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = sum;
    s->arrays[s->n_arrays-1] = params;
    s->losses = (double*)calloc(number_of_parameters,sizeof(double));
}



void generate_aco_parameters_array_to_superstruct_according_to_chaos_initialization(aco_superstruct* s, int number_of_parameters){
    if(s == NULL)
        return;
    
    int i;
    uint64_t sum = 0;
    for(i = 0; i < s->n_nodes; i++){
        sum += get_aco_node_params_size(s->same_params, s->aco_nodes[i], s->n_nodes,i);
    }
    sum*=number_of_parameters;
    float* params = (float*)calloc(sum,sizeof(float));
    uint64_t sum2 = 0;
    
    for(i = 0; i < s->n_nodes; i++){
        generate_array_aco_node_params_chaos_initialization(s->aco_nodes, s->same_params, s->aco_nodes[i],&sum2,params,s->n_nodes,i,number_of_parameters,params);
    }
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = sum;
    s->arrays[s->n_arrays-1] = params;
}

void generate_aco_pheromone_parameters_array_to_superstruct(aco_superstruct* s, float init_value){
    if(s == NULL)
        return;
    float* params = (float*)calloc(s->array_sizes[2]+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0,sizeof(float));
    set_vector_with_value(init_value,params,s->array_sizes[2]+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[2]+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0;
    s->arrays[s->n_arrays-1] = params; 
}

void generate_aco_pheromone_parameters_array_to_superstruct_edges_type(aco_superstruct* s, float init_value){
    if(s == NULL)
        return;
    int val = s->array_sizes[2]/s->array_sizes[1];
    val *= val;
    val = s->array_sizes[1]*val;
    float* params = (float*)calloc(val+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0,sizeof(float));
    set_vector_with_value(init_value,params,val+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = val+s->n_activation_functions*N_ACTIVATION_FUNCTIONS_1_0;
    s->arrays[s->n_arrays-1] = params; 
}

void generate_aco_pso_y_alphas_array_to_superstruct(aco_superstruct* s, float init_value){
    if(s == NULL)
        return;
    float* params = (float*)calloc(s->array_sizes[1],sizeof(float));
    set_vector_with_value(init_value,params,s->array_sizes[1]);
    s->n_arrays++;
    s->array_sizes = (uint64_t*)realloc(s->array_sizes,s->n_arrays*sizeof(uint64_t));
    s->arrays = (float**)realloc(s->arrays,s->n_arrays*sizeof(float*));
    s->array_sizes[s->n_arrays-1] = s->array_sizes[1];
    s->arrays[s->n_arrays-1] = params; 
}


void generate_aco_superstruct_bit_arrays(aco_superstruct* s){
    s->bit_arrays = (BitArray**)malloc(sizeof(BitArray*)*4);
    s->bit_arrays[0] = (BitArray*)malloc(sizeof(BitArray));
    s->bit_arrays[0]->data = (uint8_t*)calloc(s->array_sizes[1]+s->n_activation_functions,sizeof(uint8_t));
    s->bit_arrays[0]->size = s->array_sizes[1]+s->n_activation_functions;
    s->bit_arrays[1] = (BitArray*)malloc(sizeof(BitArray));
    s->bit_arrays[1]->data = (uint8_t*)calloc(s->array_sizes[1]+s->n_activation_functions,sizeof(uint8_t));
    s->bit_arrays[1]->size = s->array_sizes[1]+s->n_activation_functions;
    s->bit_arrays[2] = (BitArray*)malloc(sizeof(BitArray));
    s->bit_arrays[2]->data = (uint8_t*)calloc(s->array_sizes[1]+s->n_activation_functions,sizeof(uint8_t));
    s->bit_arrays[2]->size = s->array_sizes[1]+s->n_activation_functions;
    s->bit_arrays[0]->pheromone = 0;
    s->bit_arrays[1]->pheromone = 0;
    s->bit_arrays[2]->pheromone = 0;
    s->bit_arrays[3] = NULL;
}

void aco_generate_fixed_bit_array(aco_superstruct* s){
    s->bit_arrays[3] = (BitArray*)malloc(sizeof(BitArray));
    s->bit_arrays[3]->data = (uint8_t*)calloc(s->array_sizes[1],sizeof(uint8_t));
    s->bit_arrays[3]->size = s->array_sizes[1];
}





float* get_storing_arrays(aco_node** roots, int n_roots, uint64_t* size, aco_node** nodes, int n_nodes){
    int i;
    for(i = 0; i < n_roots; i++){
        (*size)+=get_storing_arrays_size_from_node(roots[i],0);
    }
    
    
    set_aco_superstruct_ff_flag(nodes,0,n_nodes);

    
    float* array = (float*)calloc((*size),sizeof(float));
    
    uint64_t size2 = 0;
    
    for(i = 0; i < n_roots; i++){
        assign_array_to_node_for_concatenation(roots[i], &size2, array, 0);
    }
    set_aco_superstruct_ff_flag(nodes,0,n_nodes);
    for(i = 0; i < n_roots; i++){
        assign_array_to_node(roots[i], &size2, array, 0);
    }
    set_aco_superstruct_ff_flag(nodes,0,n_nodes);
    
    return array;
}

void assign_random_parameters_to_aco_superstruct(aco_superstruct* s){
    int i,j;
    int n_parameters = s->array_sizes[2]/s->array_sizes[1];
    double param = 1.0/((float)(n_parameters));
    for(i = 0; i < s->array_sizes[2]; i+=n_parameters){
        float val = normalized_random();
        for(j = 0; j < n_parameters; j++,val-=param){
            if(val <= param){
                s->arrays[1][i/n_parameters] = s->arrays[2][i+j];
                break;
            }
        }
    }
}

void assign_aco_superstruct_learnable_params(aco_superstruct* s){
    int i;
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_learnable_params(s->aco_nodes[i],s->arrays[1]);
    }
}

void assign_best_parameters_to_aco_superstruct(aco_superstruct* s){
    int i,j;
    int n_parameters = s->array_sizes[2]/s->array_sizes[1];
    for(i = 0; i < s->array_sizes[2]; i+=n_parameters){
        float best = s->arrays[3][i];
        s->arrays[1][i/n_parameters] = s->arrays[2][i];
        for(j = 1; j < n_parameters; j++){
            if(s->arrays[3][i+j] > best){
                s->arrays[1][i/n_parameters] = s->arrays[2][i+j];
                best = s->arrays[3][i+j];
            }
        }
    }
}



int adjacency_matrix_count_outputs(int** adjacency_matrix, int n_nodes, int index){
    int i,sum;
    for(i = 0,sum = 0; i < n_nodes; i++){
        if(adjacency_matrix[index][i] != 0)
            sum++;
    }
    return sum;
}

int adjacency_matrix_count_inputs(int** adjacency_matrix, int n_nodes, int index){
    int i,sum;
    for(i = 0,sum = 0; i < n_nodes; i++){
        if(adjacency_matrix[i][index] != 0){
            sum++;
        }
    }
    return sum;
}

aco_node** get_leaves(aco_node** nodes, int n_nodes, int* n_leaves){
    int i;
    aco_node** leaves = NULL;
    for(i = 0; i < n_nodes; i++){
        if(!nodes[i]->n_output){
            (*n_leaves)++;
            leaves = (aco_node**)realloc(leaves,(*n_leaves)*sizeof(aco_node*));
            leaves[(*n_leaves)-1] = nodes[i];
        }
    }
    return leaves;
}

aco_node** connect_nodes(aco_node** nodes, int** adjacency_matrix, int* n_roots, int n_nodes){
    int i,j;
    aco_node** roots = NULL;
    // adjacency matrix rows input, cols output
    for(i = 0; i < n_nodes; i++){
        int n_output_nodes = adjacency_matrix_count_outputs(adjacency_matrix, n_nodes,i);
        int n_input_nodes = adjacency_matrix_count_inputs(adjacency_matrix, n_nodes,i);
        if(!n_input_nodes){
            (*n_roots)++;
            roots = (aco_node**)realloc(roots,(*n_roots)*sizeof(aco_node*));
            roots[(*n_roots)-1] = nodes[i];
        }
        add_group_nodes(i, n_input_nodes, n_output_nodes, nodes, adjacency_matrix, n_nodes);
        
    }
    
    merge_sort_aco_nodes(roots,0,(*n_roots)-1);
    return roots;
}

void add_group_nodes(int node_index, int input, int output, aco_node** nodes, int** adjacency_matrix, int n_nodes){
    int i,j,count;
    //int* v = NULL;
    if(output){
        aco_node* node = nodes[node_index];
        //v = (int*)calloc(2*output,sizeof(int));
        node->n_output = output;
        node->output = (aco_node**)malloc(sizeof(aco_node*)*output);
        for(i = 0,count = 0; i < n_nodes; i++){
            if(adjacency_matrix[node_index][i] != 0){
                node->output[count] = nodes[i];
                //v[2*count] = nodes[i]->identifier;
                //v[2*count+1] = adjacency_matrix[node_index][i];
                //nodes[i]->identifier = adjacency_matrix[node_index][i];
                count++;
            }
        }
        
        merge_sort_aco_nodes(node->output,0,output-1);
        
        /*for(i = 0; i < output; i++){
            for(j = 0; j < count; j++){
                if(v[2*j+1] == node->output[i]->identifier){
                    node->output[i]->identifier = v[2*j];
                    break;
                }
            }
        }*/
        
        put_concatenation_first(node->output,output);
        
        //free(v);
        //v = NULL;
        
        
    }
    if(input){
        aco_node* node = nodes[node_index];
        //v = (int*)calloc(2*input,sizeof(int));
        node->n_input = input;
        node->input = (aco_node**)malloc(sizeof(aco_node*)*input);
        for(i = 0,count = 0; i < n_nodes; i++){
            if(adjacency_matrix[i][node_index] != 0){
                node->input[count] = nodes[i];
                //v[2*count] = nodes[i]->identifier;
                //v[2*count+1] = adjacency_matrix[i][node_index];
                //nodes[i]->identifier = adjacency_matrix[i][node_index];
                count++;
            }
        }
        
        merge_sort_aco_nodes(node->input,0,input-1);
        /*for(i = 0; i < input; i++){
            for(j = 0; j < count; j++){
                if(v[2*j+1] == node->input[i]->identifier){
                    node->input[i]->identifier = v[2*j];
                    break;
                }
            }
        }*/
        
        //free(v);
        //v = NULL;
    }
}

uint64_t get_superstruct_array_size(aco_node** root, int n_roots){
    int i;
    uint64_t sum = 0;
    for(i = 0; i < n_roots; i++){
        sum+=get_storing_arrays_size_from_node(root[i],0);
    }
    return sum;
}

void set_aco_superstruct_ff_flag(aco_node** nodes,int value, int n_nodes){
    int i;
    for(i = 0; i < n_nodes; i++){
        nodes[i]->ff_flag = value;
    }
}

uint64_t get_storing_arrays_size_from_node(aco_node* node, int depth){
    if(node->ff_flag)
        return 0;
    int i,previous_is_lstm = 0;
    uint64_t sum = 0;
    
        
    
    if(node->fcl != NULL){
        sum+=node->fcl->output_size;
    }
    if(node->fcl_ka != NULL){
        sum+=node->fcl_ka->output_size;
    }
    else if(node->cl != NULL){
        sum+=aco_cl_output_size(node->cl->n_kernels,node->cl->input_rows,node->cl->input_cols,node->cl->kernel_rows,node->cl->kernel_cols,node->cl->stride_rows,node->cl->stride_cols,node->cl->padding_rows,node->cl->padding_cols);
    }
    else if(node->lstm != NULL){
        for(i = 0; i < node->n_input; i++){
            if(node->input[i]->lstm != NULL){
                previous_is_lstm = 1;
                break;
            }
        }
        if(!previous_is_lstm && node->n_input){
            sum+=aco_lstm_input_size(node->lstm->input_size,node->lstm->output_size)-node->lstm->input_size;
        }
        else if(node->n_input == 1){
            sum+=node->lstm->input_size;
        }
        
        sum+=aco_lstm_output_size(node->lstm->input_size,node->lstm->output_size);
    }
    else if(node->pooling != NULL){
        sum+=aco_pooling_output_size(node->pooling->channels,node->pooling->input_rows,node->pooling->input_cols,node->pooling->pooling_rows,node->pooling->pooling_cols,node->pooling->stride_rows,node->pooling->stride_cols,node->pooling->padding_rows,node->pooling->padding_cols);
    }
    else if(node->activation != NULL){
        
        sum+=node->activation->output_size;
    }
    else if(node->concatenation != NULL){
        if(node->n_input == 1)
            sum+=node->concatenation->output_size-node->concatenation->input_size;
    }
    else if(node->sum != NULL){
        sum+=node->sum->output_size;
    }
    

    for(i = 0; i < node->n_output; i++){
        sum+=get_storing_arrays_size_from_node(node->output[i],depth+1);
    }

    
    if(!depth){
        if(node->fcl != NULL){
            
            sum+=node->fcl->input_size;
        }
        if(node->fcl_ka != NULL){
            
            sum+=node->fcl_ka->input_size;
        }
        else if(node->cl != NULL){
            sum+=node->cl->channels*node->cl->input_rows*node->cl->input_cols;
        }
        else if(node->lstm != NULL){
            sum+=aco_lstm_input_size(node->lstm->input_size,node->lstm->output_size);
        }
        else if(node->pooling != NULL){
            sum+=node->pooling->channels*node->pooling->input_rows*node->pooling->input_cols;
        }
        else if(node->activation != NULL){
            sum+=node->activation->input_size;
        }
        else if(node->concatenation != NULL){
            sum+=node->concatenation->input_size;
            if(node->n_input == 2)
                sum+=node->concatenation->input2_size;
        }
        else if(node->sum != NULL){
            sum+=node->sum->input_size+node->sum->input2_size;
        }
    }
    
    node->ff_flag = 1;
    return sum;
}

void assign_array_to_node_for_concatenation(aco_node* node, uint64_t* size, float* array, int depth){
    if(node->ff_flag)
        return;
    int i, to_do = 1;
    for(i = 0; i < node->n_output; i++){
        if(node->output[i]->concatenation != NULL && node->output[i]->ff_flag && node->concatenation != NULL){
            node->concatenation->input = node->output[i]->concatenation->input2;
            node->concatenation->input_array_offset = node->output[i]->concatenation->input2_array_offset;
            node->concatenation->output_array_offset = node->output[i]->concatenation->input2_array_offset;
            node->concatenation->output = node->output[i]->concatenation->input2;
            if(node->n_input == 2){
                node->concatenation->input2 = array+node->concatenation->input_array_offset+(node->concatenation->output_size-node->concatenation->input_size);
                node->concatenation->input2_array_offset = node->concatenation->input_array_offset+(node->concatenation->output_size-node->concatenation->input_size);
            }
            to_do = 0;
        }
        assign_array_to_node_for_concatenation(node->output[i], size, array, depth+1);
        
        if(to_do && node->output[i]->concatenation != NULL && node->output[i]->ff_flag && node->concatenation != NULL){
            node->concatenation->input = node->output[i]->concatenation->input;
            node->concatenation->input_array_offset = node->output[i]->concatenation->input_array_offset;
            node->concatenation->output_array_offset = node->output[i]->concatenation->input_array_offset;
            node->concatenation->output = node->output[i]->concatenation->input;
            if(node->n_input == 2){
                node->concatenation->input2 = array+node->concatenation->input_array_offset+(node->concatenation->output_size-node->concatenation->input_size);
                node->concatenation->input2_array_offset = node->concatenation->input_array_offset+(node->concatenation->output_size-node->concatenation->input_size);
            }
            to_do = 0;
        }
        
    }
    
    if(to_do && node->concatenation != NULL){
        node->concatenation->input = array+(*size);
        node->concatenation->input_array_offset = (*size);
        (*size)+=node->concatenation->input_size;
        if(node->n_input == 2){
            node->concatenation->input2 = array+(*size);
            node->concatenation->input2_array_offset = (*size);
        }
        node->concatenation->output = node->concatenation->input;
        node->concatenation->output_array_offset = node->concatenation->input_array_offset;
        (*size)+=node->concatenation->output_size-node->concatenation->input_size;
    }
    node->ff_flag = 1;
    
}
void assign_array_to_node(aco_node* node, uint64_t* size, float* array, int depth){
    if(node->ff_flag)
        return;    
    
    int i,previous_is_lstm = 0, to_do = 0;

    if(node->fcl != NULL){
        node->fcl->output_array_offset = (*size);
        node->fcl->output = array+(*size);
        (*size)+=node->fcl->output_size;
        
    }
    if(node->fcl_ka != NULL){
        node->fcl_ka->output_array_offset = (*size);
        node->fcl_ka->output = array+(*size);
        (*size)+=node->fcl_ka->output_size;
        
    }
    else if(node->cl != NULL){
        node->cl->output_array_offset = (*size);
        node->cl->output = array+(*size);
        (*size)+=aco_cl_output_size(node->cl->n_kernels,node->cl->input_rows,node->cl->input_cols,node->cl->kernel_rows,node->cl->kernel_cols,node->cl->stride_rows,node->cl->stride_cols,node->cl->padding_rows,node->cl->padding_cols);
    }
    else if(node->lstm != NULL){
        node->lstm->output_array_offset = (*size);
        node->lstm->output = array+(*size);
        (*size)+=node->lstm->output_size*3;
        node->lstm->output_c_array_offset = (*size);
        node->lstm->output_c = array+(*size);
        (*size)+=node->lstm->output_size;
        node->lstm->output_k_array_offset = (*size);
        node->lstm->output_k = array+(*size);
        (*size)+=node->lstm->output_size;
        node->lstm->output_h_array_offset = (*size);
        node->lstm->output_h = array+(*size);
        (*size)+=node->lstm->output_size;
        for(i = 0; i < node->n_input; i++){
            if(node->input[i]->lstm != NULL){
                previous_is_lstm = 1;
                break;
            }
        }
        if(!previous_is_lstm && node->n_input){
            node->lstm->input_c_array_offset = (*size);
            node->lstm->input_c = array+(*size);
            (*size)+=node->lstm->output_size;
            node->lstm->input_k_array_offset = (*size);
            node->lstm->input_k = array+(*size);
            (*size)+=node->lstm->output_size;
            node->lstm->input_h_array_offset = (*size);
            node->lstm->input_h = array+(*size);
            (*size)+=node->lstm->output_size;
        }
        else if(node->n_input == 1){
            node->lstm->input_array_offset = (*size);
            node->lstm->input = array+(*size);
            (*size)+=node->lstm->input_size;
        }
        
        
    }
    else if(node->pooling != NULL){
        node->pooling->output_array_offset = (*size);
        node->pooling->output = array+(*size);
        (*size)+=aco_pooling_output_size(node->pooling->channels,node->pooling->input_rows,node->pooling->input_cols,node->pooling->pooling_rows,node->pooling->pooling_cols,node->pooling->stride_rows,node->pooling->stride_cols,node->pooling->padding_rows,node->pooling->padding_cols);
    }
    else if(node->activation != NULL){
        node->activation->output_array_offset = (*size);
        node->activation->output = array+(*size);
        (*size)+=node->activation->output_size;
    }
    else if(node->sum != NULL){
        node->sum->output_array_offset = (*size);
        node->sum->output = array+(*size);
        (*size)+=node->sum->output_size;
    }
    else if(node->splitter != NULL){
        node->splitter->output_array_offset = (*size)+node->splitter->index_offset;
        node->splitter->output = array+(*size)+node->splitter->index_offset;
    }
    
    
    for(i = 0; i < node->n_output; i++){
        
        if(node->output[i]->concatenation != NULL){
            if(node->lstm != NULL){
                (*size)-=node->lstm->output_size;
                if(node->output[i]->ff_flag){
                    node->lstm->output_h = node->output[i]->concatenation->input2;
                    node->lstm->output_h_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->lstm->output_h = node->output[i]->concatenation->input;
                    node->lstm->output_h_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->fcl != NULL){
                (*size)-=node->fcl->output_size;
                if(node->output[i]->ff_flag){
                    node->fcl->output = node->output[i]->concatenation->input2;
                    node->fcl->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->fcl->output = node->output[i]->concatenation->input;
                    node->fcl->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->fcl_ka != NULL){
                (*size)-=node->fcl_ka->output_size;
                if(node->output[i]->ff_flag){
                    node->fcl_ka->output = node->output[i]->concatenation->input2;
                    node->fcl_ka->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->fcl_ka->output = node->output[i]->concatenation->input;
                    node->fcl_ka->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->cl != NULL){
                (*size)-=aco_cl_output_size(node->cl->n_kernels,node->cl->input_rows,node->cl->input_cols,node->cl->kernel_rows,node->cl->kernel_cols,node->cl->stride_rows,node->cl->stride_cols,node->cl->padding_rows,node->cl->padding_cols);
                if(node->output[i]->ff_flag){
                    node->cl->output = node->output[i]->concatenation->input2;
                    node->cl->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->cl->output = node->output[i]->concatenation->input;
                    node->cl->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->pooling != NULL){
                (*size)-=aco_pooling_output_size(node->pooling->channels,node->pooling->input_rows,node->pooling->input_cols,node->pooling->pooling_rows,node->pooling->pooling_cols,node->pooling->stride_rows,node->pooling->stride_cols,node->pooling->padding_rows,node->pooling->padding_cols);
                if(node->output[i]->ff_flag){
                    node->pooling->output = node->output[i]->concatenation->input2;
                    node->pooling->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->pooling->output = node->output[i]->concatenation->input;
                    node->pooling->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->splitter != NULL){
                if(node->output[i]->ff_flag){
                    node->splitter->output = node->output[i]->concatenation->input2;
                    node->splitter->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->splitter->output = node->output[i]->concatenation->input;
                    node->splitter->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->sum != NULL){
                (*size)-=node->sum->output_size;
                if(node->output[i]->ff_flag){
                    node->sum->output = node->output[i]->concatenation->input2;
                    node->sum->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->sum->output = node->output[i]->concatenation->input;
                    node->sum->output_array_offset = node->output[i]->concatenation->input_array_offset;
                }
            }
            else if(node->activation != NULL){
                (*size)-=node->activation->output_size;
                if(node->output[i]->ff_flag){
                    node->activation->output = node->output[i]->concatenation->input2;
                    node->activation->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
                else{
                    node->activation->output = node->output[i]->concatenation->input2;
                    node->activation->output_array_offset = node->output[i]->concatenation->input2_array_offset;
                }
            }
        }
        
    }
    
    for(i = 0; i < node->n_output; i++){
        if(node->output[i]->lstm != NULL){
            if(node->lstm != NULL){
                node->output[i]->lstm->input_c = node->lstm->output_c;
                node->output[i]->lstm->input_c_array_offset = node->lstm->output_c_array_offset;
                node->output[i]->lstm->input_k = node->lstm->output_k;
                node->output[i]->lstm->input_k_array_offset = node->lstm->output_k_array_offset;
                node->output[i]->lstm->input_h = node->lstm->output_h;
                node->output[i]->lstm->input_h_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->lstm->input = node->fcl->output;
                node->output[i]->lstm->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->lstm->input = node->fcl_ka->output;
                node->output[i]->lstm->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->lstm->input = node->cl->output;
                node->output[i]->lstm->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->lstm->input = node->pooling->output;
                node->output[i]->lstm->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->lstm->input = node->concatenation->output;
                node->output[i]->lstm->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->lstm->input = node->splitter->output;
                node->output[i]->lstm->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->lstm->input = node->sum->output;
                node->output[i]->lstm->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->lstm->input = node->activation->output;
                node->output[i]->lstm->input_array_offset = node->activation->output_array_offset;
            }
        }
        if(node->output[i]->fcl != NULL){
            if(node->lstm != NULL){
                node->output[i]->fcl->input = node->lstm->output_h;
                node->output[i]->fcl->input_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->fcl->input = node->fcl->output;
                node->output[i]->fcl->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->fcl->input = node->fcl_ka->output;
                node->output[i]->fcl->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->fcl->input = node->cl->output;
                node->output[i]->fcl->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->fcl->input = node->pooling->output;
                node->output[i]->fcl->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->fcl->input = node->concatenation->output;
                node->output[i]->fcl->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->fcl->input = node->splitter->output;
                node->output[i]->fcl->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->fcl->input = node->sum->output;
                node->output[i]->fcl->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->fcl->input = node->activation->output;
                node->output[i]->fcl->input_array_offset = node->activation->output_array_offset;
            }
        }
        if(node->output[i]->fcl_ka != NULL){
            if(node->lstm != NULL){
                node->output[i]->fcl_ka->input = node->lstm->output_h;
                node->output[i]->fcl_ka->input_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->fcl_ka->input = node->fcl->output;
                node->output[i]->fcl_ka->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->fcl_ka->input = node->fcl_ka->output;
                node->output[i]->fcl_ka->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->fcl_ka->input = node->cl->output;
                node->output[i]->fcl_ka->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->fcl_ka->input = node->pooling->output;
                node->output[i]->fcl_ka->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->fcl_ka->input = node->concatenation->output;
                node->output[i]->fcl_ka->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->fcl_ka->input = node->splitter->output;
                node->output[i]->fcl_ka->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->fcl_ka->input = node->sum->output;
                node->output[i]->fcl_ka->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->fcl_ka->input = node->activation->output;
                node->output[i]->fcl_ka->input_array_offset = node->activation->output_array_offset;
            }
        }
        if(node->output[i]->cl != NULL){
            if(node->lstm != NULL){
                node->output[i]->cl->input = node->lstm->output_h;
                node->output[i]->cl->input_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->cl->input = node->fcl->output;
                node->output[i]->cl->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->cl->input = node->fcl_ka->output;
                node->output[i]->cl->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->cl->input = node->cl->output;
                node->output[i]->cl->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->cl->input = node->pooling->output;
                node->output[i]->cl->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->cl->input = node->splitter->output;
                node->output[i]->cl->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->cl->input = node->concatenation->output;
                node->output[i]->cl->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->cl->input = node->sum->output;
                node->output[i]->cl->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->cl->input = node->activation->output;
                node->output[i]->cl->input_array_offset = node->activation->output_array_offset;
            }
        }
        if(node->output[i]->pooling != NULL){
            if(node->lstm != NULL){
                node->output[i]->pooling->input = node->lstm->output_h;
                node->output[i]->pooling->input_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->pooling->input = node->fcl->output;
                node->output[i]->pooling->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->pooling->input = node->fcl_ka->output;
                node->output[i]->pooling->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->pooling->input = node->cl->output;
                node->output[i]->pooling->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->pooling->input = node->pooling->output;
                node->output[i]->pooling->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->pooling->input = node->splitter->output;
                node->output[i]->pooling->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->pooling->input = node->concatenation->output;
                node->output[i]->pooling->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->pooling->input = node->sum->output;
                node->output[i]->pooling->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->pooling->input = node->activation->output;
                node->output[i]->pooling->input_array_offset = node->activation->output_array_offset;
            }
        }
        if(node->output[i]->splitter != NULL){
            if(node->lstm != NULL){
                node->output[i]->splitter->input = node->lstm->output_h+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->lstm->output_h_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->splitter->input = node->fcl->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->fcl->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->splitter->input = node->fcl_ka->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->fcl_ka->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->splitter->input = node->cl->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->cl->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->splitter->input = node->pooling->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->pooling->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->splitter->input = node->splitter->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->splitter->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->splitter->input = node->concatenation->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->concatenation->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->splitter->input = node->sum->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->sum->output_array_offset+node->output[i]->splitter->index_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->splitter->input = node->activation->output+node->output[i]->splitter->index_offset;
                node->output[i]->splitter->input_array_offset = node->activation->output_array_offset+node->output[i]->splitter->index_offset;
            }
        }
        if(node->output[i]->activation != NULL){
            if(node->lstm != NULL){
                node->output[i]->activation->input = node->lstm->output_h;
                node->output[i]->activation->input_array_offset = node->lstm->output_h_array_offset;
            }
            else if(node->fcl != NULL){
                node->output[i]->activation->input = node->fcl->output;
                node->output[i]->activation->input_array_offset = node->fcl->output_array_offset;
            }
            else if(node->fcl_ka != NULL){
                node->output[i]->activation->input = node->fcl_ka->output;
                node->output[i]->activation->input_array_offset = node->fcl_ka->output_array_offset;
            }
            else if(node->cl != NULL){
                node->output[i]->activation->input = node->cl->output;
                node->output[i]->activation->input_array_offset = node->cl->output_array_offset;
            }
            else if(node->pooling != NULL){
                node->output[i]->activation->input = node->pooling->output;
                node->output[i]->activation->input_array_offset = node->pooling->output_array_offset;
            }
            else if(node->concatenation != NULL){
                node->output[i]->activation->input = node->concatenation->output;
                node->output[i]->activation->input_array_offset = node->concatenation->output_array_offset;
            }
            else if(node->splitter != NULL){
                node->output[i]->activation->input = node->splitter->output;
                node->output[i]->activation->input_array_offset = node->splitter->output_array_offset;
            }
            else if(node->sum != NULL){
                node->output[i]->activation->input = node->sum->output;
                node->output[i]->activation->input_array_offset = node->sum->output_array_offset;
            }
            else if(node->activation != NULL){
                node->output[i]->activation->input = node->activation->output;
                node->output[i]->activation->input_array_offset = node->activation->output_array_offset;
            }
        }
        
        if(node->output[i]->sum != NULL){
            if(node->lstm != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->lstm->output_h;
                    node->output[i]->sum->input2_array_offset = node->lstm->output_h_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->lstm->output_h;
                    node->output[i]->sum->input_array_offset = node->lstm->output_h_array_offset;
                }
            }
            else if(node->fcl != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->fcl->output;
                    node->output[i]->sum->input2_array_offset = node->fcl->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->fcl->output;
                    node->output[i]->sum->input_array_offset = node->fcl->output_array_offset;
                }
            }
            else if(node->fcl_ka != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->fcl_ka->output;
                    node->output[i]->sum->input2_array_offset = node->fcl_ka->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->fcl_ka->output;
                    node->output[i]->sum->input_array_offset = node->fcl_ka->output_array_offset;
                }
            }
            else if(node->cl != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->cl->output;
                    node->output[i]->sum->input2_array_offset = node->cl->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->cl->output;
                    node->output[i]->sum->input_array_offset = node->cl->output_array_offset;
                }
            }
            else if(node->pooling != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->pooling->output;
                    node->output[i]->sum->input2_array_offset = node->pooling->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->pooling->output;
                    node->output[i]->sum->input_array_offset = node->pooling->output_array_offset;
                }
            }
            else if(node->splitter != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->splitter->output;
                    node->output[i]->sum->input2_array_offset = node->splitter->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->splitter->output;
                    node->output[i]->sum->input_array_offset = node->splitter->output_array_offset;
                }
            }
            else if(node->concatenation != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->concatenation->output;
                    node->output[i]->sum->input2_array_offset = node->concatenation->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->concatenation->output;
                    node->output[i]->sum->input_array_offset = node->concatenation->output_array_offset;
                }
            }
            else if(node->sum != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->sum->output;
                    node->output[i]->sum->input2_array_offset = node->sum->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->sum->output;
                    node->output[i]->sum->input_array_offset = node->sum->output_array_offset;
                }
            }
            else if(node->activation != NULL){
                if(node->output[i]->ff_flag){
                    node->output[i]->sum->input2 = node->activation->output;
                    node->output[i]->sum->input2_array_offset = node->activation->output_array_offset;
                }
                else{
                    node->output[i]->sum->input = node->activation->output;
                    node->output[i]->sum->input_array_offset = node->activation->output_array_offset;
                }
            }
        }
        
        assign_array_to_node(node->output[i],size,array,depth+1);
    }

    
    if(!depth){
    
        if(node->fcl != NULL){
            node->fcl->input_array_offset = (*size);
            node->fcl->input = array+(*size);
            (*size)+=node->fcl->input_size;
        }
        else if(node->fcl_ka != NULL){
            node->fcl_ka->input_array_offset = (*size);
            node->fcl_ka->input = array+(*size);
            (*size)+=node->fcl_ka->input_size;
        }
        else if(node->cl != NULL){
            node->cl->input_array_offset = (*size);
            node->cl->input = array+(*size);
            (*size)+=node->cl->channels*node->cl->input_rows*node->cl->input_cols;
        }
        else if(node->lstm != NULL){
            node->lstm->input_array_offset = (*size);
            node->lstm->input = array+(*size);
            (*size)+=node->lstm->input_size;
            node->lstm->input_c_array_offset = (*size);
            node->lstm->input_c = array+(*size);
            (*size)+=node->lstm->output_size;
            node->lstm->input_k_array_offset = (*size);
            node->lstm->input_k = array+(*size);
            (*size)+=node->lstm->output_size;
            node->lstm->input_h_array_offset = (*size);
            node->lstm->input_h = array+(*size);
            (*size)+=node->lstm->output_size;
        }
        else if(node->pooling != NULL){
            node->pooling->input_array_offset = (*size);
            node->pooling->input = array+(*size);
            (*size)+=node->pooling->channels*node->pooling->input_rows*node->pooling->input_cols;
        }
        else if(node->splitter != NULL){
            node->splitter->input_array_offset = (*size)+node->splitter->index_offset;
            node->splitter->input = array+(*size)+node->splitter->index_offset;
        }
        else if(node->activation != NULL){
            node->activation->input_array_offset = (*size);
            node->activation->input = array+(*size);
            (*size)+=node->activation->input_size;
        }
        else if(node->sum != NULL){
            node->sum->input_array_offset = (*size);
            node->sum->input = array+(*size);
            (*size)+=node->sum->input_size+node->sum->input2_size;
        }
    }
    
    node->ff_flag = 1;
    return;
}

void set_aco_superstruct_array_to_null(aco_superstruct* s, int index){
    free(s->arrays[index]);
    s->arrays[index] = NULL;
}


void aco_superstruct_node_visit_feed_forward(aco_node* n){
    if(n->ff_flag)
        return;
    int i;
    for(i = 0; i < n->n_input; i++){
        aco_superstruct_node_visit_feed_forward(n->input[i]);
    }
    
    if(n->ff_flag)
        return;
    aco_node_feed_forward(n);
    n->ff_flag = 1;
    
    for(i = 0; i < n->n_output; i++){
        aco_superstruct_node_visit_feed_forward(n->output[i]);
    }
}

void aco_superstruct_node_visit_feed_forward_edge_popup(aco_node* n){
    
    if(n->ff_flag)
        return;
    int i;
    for(i = 0; i < n->n_input; i++){
        aco_superstruct_node_visit_feed_forward_edge_popup(n->input[i]);
    }
    
    if(n->ff_flag)
        return;
    
    aco_node_feed_forward_edge_popup(n);
    n->ff_flag = 1;
    
    /*
    printf("index: %d\n",n->identifier);
    if(n->activation != NULL)
        print_vector(n->activation->input,n->activation->input_size);
    
    if(n->cl != NULL)
        print_vector(n->cl->input,n->cl->channels*n->cl->input_rows*n->cl->input_cols);
    
    if(n->fcl != NULL)
        print_vector(n->fcl->input,n->fcl->input_size);
    
    if(n->pooling != NULL)
        print_vector(n->pooling->input,n->pooling->channels*n->pooling->input_rows*n->pooling->input_cols);
    
    if(n->sum != NULL)
        print_vector(n->sum->input,n->sum->input_size);
    if(n->lstm != NULL)
        print_vector(n->lstm->input,n->lstm->input_size);
    
    if(n->concatenation != NULL)
        print_vector(n->concatenation->output,n->concatenation->output_size);
    */
    
    for(i = 0; i < n->n_output; i++){
        aco_superstruct_node_visit_feed_forward_edge_popup(n->output[i]);
    }
}

void aco_superstruct_node_visit_back_propagation_edge_popup(aco_node* n){
    if(n->ff_flag)
        return;
    int i;
    for(i = 0; i < n->n_output; i++){
        aco_superstruct_node_visit_back_propagation_edge_popup(n->output[i]);
    }
    
    if(n->ff_flag)
        return;
    
    
    
    aco_node_back_propagation_edge_popup(n);
    n->ff_flag = 1;
    
    /*
    printf("index: %d\n",n->identifier);
    if(n->activation != NULL)
        print_vector(n->activation->input,n->activation->input_size);
    
    if(n->cl != NULL)
        print_vector(n->cl->input,n->cl->channels*n->cl->input_rows*n->cl->input_cols);
    
    if(n->fcl != NULL)
        print_vector(n->fcl->input,n->fcl->input_size);
    
    if(n->pooling != NULL)
        print_vector(n->pooling->input,n->pooling->channels*n->pooling->input_rows*n->pooling->input_cols);
    
    if(n->sum != NULL)
        print_vector(n->sum->input,n->sum->input_size);
    if(n->lstm != NULL)
        print_vector(n->lstm->input,n->lstm->input_size);
    
    if(n->concatenation != NULL)
        print_vector(n->concatenation->output,n->concatenation->output_size);
    */
    
    for(i = 0; i < n->n_input; i++){
        aco_superstruct_node_visit_back_propagation_edge_popup(n->input[i]);
    }
}

void aco_superstruct_copy_inputs(aco_superstruct* s, float** inputs, int n_inputs){
    int i,j;
    float* temp1,*temp2;
    for(j = 0, i = 0; i < s->n_roots && j < n_inputs; i++){
        temp1 = inputs[j];
        j++;
        if(s->roots[i]->sum == NULL)
            temp2 = NULL;
        else{
            temp2 = inputs[j];
            j++;
        }
        assign_aco_node_inputs(s->roots[i], temp1,temp2);
    }
}

int aco_superstruct_get_number_of_roots(aco_superstruct* s){
    return s->n_roots;
}

int aco_superstruct_getnumber_of_leaves(aco_superstruct* s){
    return s->n_leaves;
}

int aco_superstruct_get_first_size_root_i(aco_superstruct* s, int index){
    get_aco_node_first_input_size(s->roots[index]);
}

int aco_superstruct_get_second_size_root_i(aco_superstruct* s, int index){
    get_aco_node_second_input_size(s->roots[index]);
}

void aco_superstruct_assign_outputs(aco_superstruct* s, float** outputs, int n_outputs){
    int i,j;
    float* temp1;

    for(j = 0, i = 0; i < s->n_leaves && j < n_outputs; i++, j++){
        temp1 = outputs[j];
        assign_aco_node_outputs(s->leaves[i], temp1);
    }
}

void reassign_right_storing_arrays(aco_superstruct* s){
    int i;
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_storing_params(s->aco_nodes[i],s->arrays[0]);
    }
}


void aco_superstruct_single_thread_feed_forward(aco_superstruct* s){
    int i;
    set_aco_superstruct_ff_flag(s->aco_nodes,0,s->n_nodes);
    
    for(i = 0; i < s->n_roots; i++){
        aco_superstruct_node_visit_feed_forward(s->roots[i]);
    }
}

void aco_superstruct_single_thread_feed_forward_edge_popup(aco_superstruct* s){
    int i;
    set_aco_superstruct_ff_flag(s->aco_nodes,0,s->n_nodes);
    
    for(i = 0; i < s->n_roots; i++){
        aco_superstruct_node_visit_feed_forward_edge_popup(s->roots[i]);
    }
}

void aco_superstruct_single_thread_back_propagation_edge_popup(aco_superstruct* s){
    int i;
    set_aco_superstruct_ff_flag(s->aco_nodes,0,s->n_nodes);
    
    for(i = 0; i < s->n_leaves; i++){
        aco_superstruct_node_visit_back_propagation_edge_popup(s->leaves[i]);
    }
}


void copy_aco_superstruct_activation(aco_superstruct* copy, aco_superstruct* original){
    if(copy == NULL || original == NULL)
        return;
    int i;
    for(i = 0; i < original->n_nodes; i++){
        copy_aco_node_activation(copy->aco_nodes[i],original->aco_nodes[i]);
    }
}

uint64_t aco_superstruct_get_total_input_size(aco_superstruct* s){
    if(s == NULL)
        return 0;
    int i;
    uint64_t input_sum = 0;
    for(i = 0; i < s->n_roots; i++){
        if(s->roots[i]->sum == NULL){
            input_sum += get_aco_node_input_size(s->roots[i]);
        }
        else{
            input_sum += s->roots[i]->sum->input_size+s->roots[i]->sum->input2_size;
        }
    }
    return input_sum;
}

uint64_t aco_superstruct_get_total_output_size(aco_superstruct* s){
    int i;
    uint64_t output_sum = 0;
    for(i = 0; i < s->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s->leaves[i]);
    }
    return output_sum;
}

void increase_iteration_index(aco_superstruct* s){
    s->iteration_index++;
}

void aco_update_pso_parameters(aco_superstruct* s){
    s->sigma1*=log(((double)s->iteration_index))/log(((double)(s->iteration_index+1)));
    s->sigma2*=log(((double)s->iteration_index))/log(((double)(s->iteration_index+1)));
    s->alpha*=2.0;
}

void aco_update_pso(aco_superstruct* s){
    
    int i,j;
    int n_parameters = s->array_sizes[2]/s->array_sizes[1];
    double min_loss = s->losses[0];
    double* exps = (double*)calloc(n_parameters,sizeof(double));
    double max_y_alpha_on_batch_current = 0;
    double max_difference = 0;
    double sum = 0;
    double dt_sqrt = sqrtf(s->dt);
    for(i = 1; i < n_parameters; i++){
        if(s->losses[i] < min_loss)
            min_loss = s->losses[i];
    } 
    for(i = 0; i < n_parameters; i++){
        exps[i] = exp(-s->alpha*(s->losses[i]-min_loss));
        sum+=exps[i];
    }
    for(i = 0; i < s->array_sizes[1]; i++){
        max_y_alpha_on_batch_current = 0;
        for(j = 0; j < n_parameters; j++){
            max_y_alpha_on_batch_current+=s->arrays[2][j*s->array_sizes[1]+i]*exps[j];
        }
        max_y_alpha_on_batch_current/=sum;
        if (float_abs(max_y_alpha_on_batch_current-s->arrays[4][i]) > max_difference){
            max_difference = float_abs(max_y_alpha_on_batch_current-s->arrays[4][i]);
        }
        s->arrays[4][i] = max_y_alpha_on_batch_current;
        for(j = 0; j < n_parameters; j++){
            s->arrays[3][j*s->array_sizes[1]+i]*=(s->m/(s->m+s->gamma*s->dt));
            s->arrays[3][j*s->array_sizes[1]+i]+=(s->lambda2/(s->m+s->gamma*s->dt))*(max_y_alpha_on_batch_current-s->arrays[2][j*s->array_sizes[1]+i])*s->dt;
            s->arrays[3][j*s->array_sizes[1]+i]+=(s->sigma2/(s->m+s->gamma*s->dt))*float_abs(max_y_alpha_on_batch_current-s->arrays[2][j*s->array_sizes[1]+i])*dt_sqrt*random_normal();
            s->arrays[2][j*s->array_sizes[1]+i]+=s->arrays[3][j*s->array_sizes[1]+i]*s->dt;
        }
    }
    if(max_difference<0.0001){
        double max = s->sigma1;
        if(s->sigma2 > max)
            max = s->sigma2;
        for(i = 0; i < s->array_sizes[2]; i++){
            s->arrays[3][i]+=max*dt_sqrt*random_normal();
        }
    }
    
    free(exps);
}


void set_losses_according_to_index(aco_superstruct* s, int index, double loss){
    s->losses[index] = loss;
}


void merge_sort_aco_superstruct(aco_superstruct* s){
    int i;
    for(i = 0; i < s->n_nodes; i++){
        merge_sort_aco_node(s->aco_nodes[i]);
    }
}

void merge_sort_aco_superstruct_multi_thread(aco_superstruct* s, int threads){
    if (threads > s->n_nodes){
        threads = s->n_nodes;
    }
    pthread_t thread[threads];
    multithread_aco_sort* args[threads];
    int i,j,n;
    for(i = 0, j = 0; i < s->n_nodes; i+=s->n_nodes/threads, j++){
        
        int n = s->n_nodes/threads;
        if (n > s->n_nodes-i){
            n = s->n_nodes-i;
        }
        args[j] = (multithread_aco_sort*)malloc(sizeof(multithread_aco_sort));
        args[j]->s = s;
        args[j]->start = i;
        args[j]->end = i+n;
        pthread_create(thread+j, NULL, merge_sort_aco_superstruct_thread, args[j]);
    }
    
    for(i = 0, j = 0; i < s->n_nodes; i+=s->n_nodes/threads, j++){
        pthread_join(thread[j], NULL);
        free(args[j]);
    }
}

void* merge_sort_aco_superstruct_thread(void* _args){
    multithread_aco_sort* args = (multithread_aco_sort*) _args;
    int i;
    for(i = args->start; i < args->end; i++){
        merge_sort_aco_node(args->s->aco_nodes[i]);
    }
}

void assign_aco_superstruct_storing_partial_derivatives(aco_superstruct* s, float* params){
    int i;
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_storing_partial_derivatives(s->aco_nodes[i], params);
    }
}

void assign_aco_superstruct_learnable_params_scores(aco_superstruct* s, float* params, int* indices, float* scores){
    if(s == NULL)
        return;
    int i;
    for(i = 0; i < s->n_nodes; i++){
        assign_aco_node_learnable_params_scores(s->aco_nodes[i], params, indices, scores);
    }
}

void init_aco_superstruct_scores(aco_superstruct* s){
    if(s == NULL)
        return;
    int i;
    for(i = 0; i < s->n_nodes; i++){
        init_aco_node_scores(s->aco_nodes[i]);
    }
}

void set_aco_superstruct_k_percentage(aco_superstruct* s, float k_percentage){
    if(s == NULL)
        return;
    int i;
    for(i = 0; i < s->n_nodes; i++){
        set_aco_node_k_percentage(s->aco_nodes[i], k_percentage);
    }
}

void set_aco_superstruct_used_output(aco_superstruct* s){
    if(s == NULL)
        return;
    int i,j,flag;
    for(i = 0; i < s->n_nodes; i++){
        if(!nodes_params_already_checked(s,i))
            set_aco_node_used_output(s->aco_nodes[i]);
    }
}

int* aco_superstruct_node_visit_used_outputs_no_parametric(aco_node* n, int* used_outputs, int node_id){
    if(n->ff_flag)
        return get_aco_node_used_outputs(n);
    int i;
    int* used_outputs2 = NULL;
    uint64_t used_outputs_offset = 0;
    uint64_t used_outputs_offset2 = 0;
    n->ff_flag = 1;
    for(i = 0; i < n->n_input; i++){
        if(n->input[i]->identifier != node_id){
            if(used_outputs == NULL){
                used_outputs = aco_superstruct_node_visit_used_outputs_no_parametric(n->input[i], NULL, n->identifier);
                used_outputs_offset = get_aco_node_output_offset(n->input[i]);
            }
            else{
                used_outputs2 = aco_superstruct_node_visit_used_outputs_no_parametric(n->input[i], NULL, n->identifier);
                used_outputs_offset2 = get_aco_node_output_offset(n->input[i]);
            }
        }
        else{
            used_outputs_offset = get_aco_node_output_offset(n->input[i]);
        }
    }
    
    if(!n->n_input){
        set_aco_node_used_output_no_parametric(n,NULL,NULL,1, used_outputs_offset, used_outputs_offset2);
    }
    else{
        set_aco_node_used_output_no_parametric(n,used_outputs,used_outputs2,0, used_outputs_offset, used_outputs_offset2);
    }
    for(i = 0; i < n->n_output; i++){
        if(n->output[i]->identifier != node_id){
            aco_superstruct_node_visit_used_outputs_no_parametric(n->output[i], get_aco_node_used_outputs(n),n->identifier);
        }
    }
    return get_aco_node_used_outputs(n);
}

void set_aco_superstruct_used_output_no_parametric(aco_superstruct* s){
    if(s == NULL)
        return;
    int i;
    set_aco_superstruct_ff_flag(s->aco_nodes,0,s->n_nodes);
    
    for(i = 0; i < s->n_roots; i++){
        aco_superstruct_node_visit_used_outputs_no_parametric(s->roots[i], NULL, -1);
    }
}

void set_total_aco_superstruct_used_output(aco_superstruct* s){
    if(s == NULL)
        return;
    set_aco_superstruct_used_output(s);
    set_aco_superstruct_used_output_no_parametric(s);

}

uint8_t* get_bit_array_data_from_index(aco_superstruct* s, int index){
    if(index > 2 || index < 0){
        fprintf(stderr,"Error, index out of range");
        exit(1);
    }
    
    return s->bit_arrays[index]->data;
}

void* set_bit_array_data_from_index(aco_superstruct* s, uint8_t* array, int index){
    if(index > 2 || index < 0){
        fprintf(stderr,"Error, index out of range");
        exit(1);
    }
    
    s->bit_arrays[index]->data = array;
}


