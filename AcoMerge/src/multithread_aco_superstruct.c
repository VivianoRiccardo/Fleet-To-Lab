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


void* aco_superstruct_multi_input_multi_output_ff_focal_loss(void* _args) {
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset = args->input_offset;
    uint64_t output_offset = args->output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    for(i = 0; i < args->n_inputs_outputs; i++){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
                
        for(j = 0; j < args->s->n_leaves; j++){
            temp1 = args->output+output_offset;
            assign_aco_node_outputs(args->s->leaves[j], temp1);
            output_offset+=get_aco_node_output_size(args->s->leaves[j]);
        }
        
        aco_superstruct_single_thread_feed_forward(args->s);
        for(j = 0; j < args->s->n_leaves; j++){
            output_offset-=get_aco_node_output_size(args->s->leaves[j]);
        }
        for(j = 0; j < args->s->n_leaves; j++){
            if(args->s->leaves[j]->concatenation != NULL){
                copy_array(args->s->leaves[j]->concatenation->output,args->output+output_offset,args->s->leaves[j]->concatenation->output_size);
            }
            int size_out = get_aco_node_output_size(args->s->leaves[j]);
            float* temp_out = (float*)calloc(size_out,sizeof(float));
            focal_loss_array(args->output+output_offset,args->real_output+output_offset,temp_out,2.0,size_out);
            args->ret_err += sum_over_input(temp_out,size_out);
            free(temp_out);
            output_offset+=size_out;
        }
        
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);
    }
}

void* sum_partial_derivatives_thread(void* _args){
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    // input_offset where to start to add in the array
    // output offset where to stop to add in the array
    // input_size number of arrays to add
    int i;
    for(i = 1; i < args->input_size; i++){
        sum_vectors(args->e->arrays[0][1]+args->input_offset,args->e->arrays[i][1]+args->input_offset,args->e->arrays[0][1]+args->input_offset,args->output_offset);
    }
}

void* sum_partial_derivatives_update_reset_thread(void* _args){
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    // input_offset where to start to add in the array
    // output offset where to stop to add in the array
    // input_size number of arrays to add
    int i;
    for(i = 1; i < args->input_size; i++){
        sum_vectors(args->e->arrays[0][1]+args->input_offset,args->e->arrays[i][1]+args->input_offset,args->e->arrays[0][1]+args->input_offset,args->output_offset);
    }
    for(i = args->input_offset; i < args->input_offset+args->output_offset; i++){
        nadam(args->e->arrays[0][2]+i, args->e->arrays[0][3]+i, args->e->arrays[0][4]+i, args->e->arrays[0][1][i],  args->e->mt, args->e->nt, args->e->m0, args->e->n0, args->e->mini_batch_size, args->e->epsilon, args->e->lr);
    }
    
    for(i = 0; i < args->input_size; i++){
        set_vector_with_value(0,args->e->arrays[i][1]+args->input_offset,args->output_offset);
    }
    

    
    
}

void* set_partial_derivatives_to_zero_thread(void* _args){
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    set_partial_derivative_to_zero_according_to_index(args->e, args->index);

}

void* aco_superstruct_multi_input_multi_output_ff_loss_bp(void* _args) {
    
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset;
    uint64_t output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    for(input_offset = 0, i = 0; i < args->n_inputs_outputs; i++, input_offset = 0, output_offset = 0){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset+args->indices[args->input_offset+i]*args->input_size;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->indices[args->input_offset+i]*args->input_size+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
        
        
        aco_superstruct_single_thread_feed_forward_edge_popup(args->s);
        computer_derivative_loss(args->real_output+args->indices[args->output_offset+i]*args->output_size, 2.0, args->e, args->index);
        aco_superstruct_single_thread_back_propagation_edge_popup(args->s);
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);// reset outputs
        set_vector_with_value(0,args->e->arrays[args->index][0],args->e->array_sizes[args->index][0]);// reset doutputs
    }
}


void* aco_superstruct_multi_input_multi_output_ff_focal_loss_indices(void* _args) {
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset;
    uint64_t output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    float* output = (float*)calloc(args->n_inputs_outputs*args->output_size,sizeof(float));
    for(input_offset = 0, output_offset = 0, i = 0; i < args->n_inputs_outputs; i++, input_offset = 0, output_offset = 0){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset+args->indices[args->input_offset+i]*args->input_size;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->indices[args->input_offset+i]*args->input_size+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
                
        for(j = 0; j < args->s->n_leaves; j++){
            temp1 = output+output_offset+i*args->output_size;
            assign_aco_node_outputs(args->s->leaves[j], temp1);
            output_offset+=get_aco_node_output_size(args->s->leaves[j]);
        }
        
        aco_superstruct_single_thread_feed_forward(args->s);
        
        output_offset = 0;
        for(j = 0; j < args->s->n_leaves; j++){
            if(args->s->leaves[j]->concatenation != NULL){
                copy_array(args->s->leaves[j]->concatenation->output,output+output_offset+i*args->output_size,args->s->leaves[j]->concatenation->output_size);
            }
            int size_out = get_aco_node_output_size(args->s->leaves[j]);
            float* temp_out = (float*)calloc(size_out,sizeof(float));
            focal_loss_array(output+output_offset+i*args->output_size,args->real_output+output_offset+args->indices[args->output_offset+i]*args->output_size,temp_out,2.0,size_out);
            args->ret_err += sum_over_input(temp_out,size_out);
            free(temp_out);
            output_offset+=size_out;
        }
        
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);
    }
    free(output);
}

void* aco_superstruct_multi_input_multi_output_ff_focal_loss_with_heuristic(void* _args) {
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset = args->input_offset;
    uint64_t output_offset = args->output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    set_vector_with_value(0,args->s->arrays[args->s->n_arrays-1],args->s->array_sizes[0]);
    for(i = 0; i < args->n_inputs_outputs; i++){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
                
        for(j = 0; j < args->s->n_leaves; j++){
            temp1 = args->output+output_offset;
            assign_aco_node_outputs(args->s->leaves[j], temp1);
            output_offset+=get_aco_node_output_size(args->s->leaves[j]);
        }
        
        aco_superstruct_single_thread_feed_forward(args->s);
        for(j = 0; j < args->s->n_leaves; j++){
            output_offset-=get_aco_node_output_size(args->s->leaves[j]);
        }
        for(j = 0; j < args->s->n_leaves; j++){
            if(args->s->leaves[j]->concatenation != NULL){
                copy_array(args->s->leaves[j]->concatenation->output,args->output+output_offset,args->s->leaves[j]->concatenation->output_size);
            }
            int size_out = get_aco_node_output_size(args->s->leaves[j]);
            float* temp_out = (float*)calloc(size_out,sizeof(float));
            focal_loss_array(args->output+output_offset,args->real_output+output_offset,temp_out,2.0,size_out);
            args->ret_err += sum_over_input(temp_out,size_out);
            free(temp_out);
            output_offset+=size_out;
        }
        
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_frequencies_superstruct(args->s->arrays[0],args->s->arrays[args->s->n_arrays-1],args->s->array_sizes[0]);
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);
    }
}

void* aco_superstruct_multi_input_multi_output_ff(void* _args) {
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset = args->input_offset;
    uint64_t output_offset = args->output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    int this_sum = 0;
    for(i = 0; i < args->n_inputs_outputs; i++){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
                
        for(j = 0; j < args->s->n_leaves; j++){
            temp1 = args->output+output_offset;
            assign_aco_node_outputs(args->s->leaves[j], temp1);
            output_offset+=get_aco_node_output_size(args->s->leaves[j]);
        }
        
        aco_superstruct_single_thread_feed_forward(args->s);
        for(j = 0; j < args->s->n_leaves; j++){
            output_offset-=get_aco_node_output_size(args->s->leaves[j]);
        }
        for(j = 0; j < args->s->n_leaves; j++){
            if(args->s->leaves[j]->concatenation != NULL){
                copy_array(args->s->leaves[j]->concatenation->output,args->output+output_offset,args->s->leaves[j]->concatenation->output_size);
            }
            int size_out = get_aco_node_output_size(args->s->leaves[j]);
            output_offset+=size_out;
        }
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);
    }
}
void* aco_superstruct_multi_input_multi_output_ff_edge_popup(void* _args) {
    
    // depacking args
    multithread_aco_superstruct* args = (multithread_aco_superstruct*) _args;
    
    
    int i,j;
    uint64_t input_offset = args->input_offset;
    uint64_t output_offset = args->output_offset;
    float* temp1 = NULL;
    float* temp2 = NULL;
    int this_sum = 0;
    for(i = 0; i < args->n_inputs_outputs; i++){
        for(j = 0; j < args->s->n_roots; j++){
            temp1 = args->input+input_offset;
            if(args->s->roots[j]->sum == NULL)
                temp2 = NULL;
            else
                temp2 = args->input+input_offset+args->s->roots[j]->sum->input_size;
            assign_aco_node_inputs(args->s->roots[j], temp1,temp2);
            input_offset+=get_aco_node_input_size(args->s->roots[j]);
            
        }
        for(j = 0; j < args->s->n_leaves; j++){
            temp1 = args->output+output_offset;
            assign_aco_node_outputs(args->s->leaves[j], temp1);
            output_offset+=get_aco_node_output_size(args->s->leaves[j]);
        }
        
        aco_superstruct_single_thread_feed_forward_edge_popup(args->s);
        for(j = 0; j < args->s->n_leaves; j++){
            output_offset-=get_aco_node_output_size(args->s->leaves[j]);
        }
        for(j = 0; j < args->s->n_leaves; j++){
            if(args->s->leaves[j]->concatenation != NULL){
                copy_array(args->s->leaves[j]->concatenation->output,args->output+output_offset,args->s->leaves[j]->concatenation->output_size);
            }
            int size_out = get_aco_node_output_size(args->s->leaves[j]);
            output_offset+=size_out;
        }
        
        
        for(j = 0; j < args->s->n_nodes; j++){
            assign_aco_node_storing_params(args->s->aco_nodes[j],args->s->arrays[0]);
        }
        set_vector_with_value(0,args->s->arrays[0],args->s->array_sizes[0]);
    }
}

// for inputs to concatenation you should give both input1 and input2
float* get_multi_output_from_multi_input_aco_superstruct_ff(aco_superstruct** s, float* input, int n_superstructs,  int n_inputs){
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }
    
    float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs))*input_sum;
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs))*output_sum;
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = output;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff, args[i]);
            
    }
    
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    
    return output;
}
float* get_multi_output_from_multi_input_aco_superstruct_ff_edge_popup(aco_superstruct** s, float* input, int n_superstructs,  int n_inputs){
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }
    
    float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs))*input_sum;
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs))*output_sum;
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = output;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff_edge_popup, args[i]);
            
    }
    
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    return output;
}

void multi_thread_sum_partial_derivatives(edge_popup_trainer* e, int n_used_threads, int threads){
    if(n_used_threads < 1 || e == NULL)
        return;
    if(n_used_threads >= e->batch_size){
        n_used_threads = e->batch_size;
    }
    pthread_t thread[n_used_threads];
    multithread_aco_superstruct* args[n_used_threads];
    
    uint64_t size_array = e->s[0]->array_sizes[1];// sizes of the weights
    
    int i;
    
    for(i = 0; i < threads; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        args[i]->input_size =  n_used_threads;
        args[i]->input_offset = i*(size_array/threads);
        if(i == threads-1)
            args[i]->output_offset = size_array - i*((int)(size_array/threads));
        else
            args[i]->output_offset = int_min(((int)(size_array/threads)),size_array - i*((int)(size_array/threads)));
        args[i]->e = e;
        pthread_create(thread+i, NULL, sum_partial_derivatives_thread, args[i]);
            
    }
    
    for(i = 0; i < threads; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
}

void multi_thread_sum_partial_derivatives_update_reset(edge_popup_trainer* e, int n_used_threads, int threads){
    if(n_used_threads < 1 || e == NULL)
        return;
    if(n_used_threads >= e->batch_size){
        n_used_threads = e->batch_size;
    }
    pthread_t thread[threads];
    multithread_aco_superstruct* args[threads];
    
    uint64_t size_array = e->s[0]->array_sizes[1];// sizes of the weights
    
    int i;
    
    for(i = 0; i < threads; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        args[i]->input_size =  n_used_threads;
        args[i]->input_offset = i*(size_array/threads);
        if(i == threads-1)
            args[i]->output_offset = size_array - i*((int)(size_array/threads));
        else
            args[i]->output_offset = int_min(((int)(size_array/threads)),size_array - i*((int)(size_array/threads)));
        args[i]->e = e;
        pthread_create(thread+i, NULL, sum_partial_derivatives_update_reset_thread, args[i]);
            
    }
    
    for(i = 0; i < threads; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    
}

void multi_thread_set_to_zero_derivatives(edge_popup_trainer* e){
    if(e == NULL)
        return;
    pthread_t thread[e->batch_size];
    multithread_aco_superstruct* args[e->batch_size];
        
    int i;
    
    for(i = 0; i < e->batch_size; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        args[i]->index =  i;
        args[i]->e = e;
        pthread_create(thread+i, NULL, set_partial_derivatives_to_zero_thread, args[i]);
            
    }
    
    for(i = 0; i < e->batch_size; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
}

// for inputs to concatenation you should give both input1 and input2
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs){
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i,j;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }

    
    float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    double error = 0;
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs))*input_sum;
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs))*output_sum;
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = output;
        args[i]->ret_err = 0;
        args[i]->real_output = real_output;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff_focal_loss, args[i]);
            
    }
                
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        error+=args[i]->ret_err;
        free(args[i]);
    }
    free(output);
    return error;
}
// for inputs to concatenation you should give both input1 and input2
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss_with_indices(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs, int* indices){
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }
    
    //float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    double error = 0;
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->indices = indices;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs));
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs));
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = NULL;
        args[i]->ret_err = 0;
        args[i]->real_output = real_output;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff_focal_loss_indices, args[i]);
            
    }
                
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        error+=args[i]->ret_err;
        free(args[i]);
    }
    //free(output);
    return error;
}
// for inputs to concatenation you should give both input1 and input2
void ff_loss_bp_edge_popup(edge_popup_trainer* e, float* input,float* real_output,  int n_inputs, int* indices){
    if(e == NULL)
        return;
    aco_superstruct** s = e->s;
    int n_superstructs = e->batch_size;
    
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }
    
    //float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    double error = 0;
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->indices = indices;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs));
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs));
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = NULL;
        args[i]->ret_err = 0;
        args[i]->real_output = real_output;
        args[i]->e = e;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff_loss_bp, args[i]);
        
            
    }
                
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        free(args[i]);
    }
    //free(output);
    return;
}

// for inputs to concatenation you should give both input1 and input2
double get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss_with_heuristic(aco_superstruct** s, float* input,float* real_output, int n_superstructs,  int n_inputs){
    if(n_superstructs > n_inputs)
        n_superstructs = n_inputs;
    pthread_t thread[n_superstructs];
    multithread_aco_superstruct* args[n_superstructs];
    
    uint64_t input_sum = 0;
    uint64_t output_sum = 0;
    
    int i;
    for(i = 0; i < s[0]->n_leaves; i++){
        output_sum+=get_aco_node_output_size(s[0]->leaves[i]);
    }
    for(i = 0; i < s[0]->n_roots; i++){
        input_sum+=get_aco_node_input_size(s[0]->roots[i]);
    }
    
    float* output = (float*)calloc(output_sum*n_inputs,sizeof(float));
    double error = 0;
    for(i = 0; i < n_superstructs; i++){
        args[i] = (multithread_aco_superstruct*)malloc(sizeof(multithread_aco_superstruct));
        if(i == n_superstructs-1)
            args[i]->n_inputs_outputs = n_inputs - i*((int)(n_inputs/n_superstructs));
        else
            args[i]->n_inputs_outputs = int_min(((int)(n_inputs/n_superstructs)),n_inputs - i*((int)(n_inputs/n_superstructs)));
        args[i]->s = s[i];
        args[i]->index = i;
        args[i]->input_offset = i*((int)(n_inputs/n_superstructs))*input_sum;
        args[i]->output_offset = i*((int)(n_inputs/n_superstructs))*output_sum;
        args[i]->input_size = input_sum;
        args[i]->output_size = output_sum;
        args[i]->input = input;
        args[i]->output = output;
        args[i]->ret_err = 0;
        args[i]->real_output = real_output;
        pthread_create(thread+i, NULL, aco_superstruct_multi_input_multi_output_ff_focal_loss_with_heuristic, args[i]);
            
    }
                
    for(i = 0; i < n_superstructs; i++){
        pthread_join(thread[i], NULL);
        error+=args[i]->ret_err;
        free(args[i]);
    }
    free(output);
    return error;
}


