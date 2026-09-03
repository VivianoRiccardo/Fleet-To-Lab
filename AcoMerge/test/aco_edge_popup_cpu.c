#include "../src/ezspark.h"

double value_the_network(float* network_output, float* outputs, int instances, int size){
    int i;
    // getting the pheromone
    double error = 0;
    for(i = 0; i < size; i++){
        error+=focal_loss(network_output[i],outputs[i],2.0);
    }
    return error;
}

int main(){
    
    // randomization
    srand(time(NULL));
    
    // parameters for the network
    int fcl_input1_size = 784;
    int fcl_output1_size = 100;
    int fcl_input2_size = 100;
    int fcl_output2_size = 100;
    int fcl_input3_size = 100;
    int fcl_output3_size = 10;
    int n_nodes = 6;
    int count_update_x = 0;
    int divisor_of_instances = 1;
    
    // number of threads to use
    int threads = 16;
    
    // instances of the training set
    int instances = 50000;
    
    // for the mini batch considered
    int mini_batch_size = 80;
    
    // parameters for reading the file data
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "./data/train.bin";
    char temp[2];
    temp[1] = '\0';
    int file_size = 0;
    
    
    int i,j,z;
    
    // the network
    aco_node** nodes = (aco_node**)malloc(sizeof(aco_node*)*n_nodes);
    
    aco_fcl* node0 = init_aco_fcl(fcl_input1_size,fcl_output1_size,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node1 = init_aco_activation(GELU,N_ACTIVATION_FUNCTIONS_1_0,fcl_output1_size,fcl_output1_size,1,0,0,0,NULL,NULL,NULL,generate_all_activations());
    aco_fcl* node2 = init_aco_fcl(fcl_input2_size,fcl_output2_size,2,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node3 = init_aco_activation(GELU,N_ACTIVATION_FUNCTIONS_1_0,fcl_output2_size,fcl_output2_size,3,0,0,0,NULL,NULL,NULL,generate_all_activations());
    aco_fcl* node4 = init_aco_fcl(fcl_input3_size,fcl_output3_size,4,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node5 = init_aco_activation(SOFTMAX,0,fcl_output3_size,fcl_output3_size,5,0,0,0,NULL,NULL,NULL,NULL);
    
    nodes[0] = init_aco_node(0,0,0,node0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    nodes[1] = init_aco_node(0,0,1,NULL,NULL,NULL,NULL,NULL,NULL,node1,NULL,NULL,NULL,NULL);
    nodes[2] = init_aco_node(0,0,2,node2,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    nodes[3] = init_aco_node(0,0,3,NULL,NULL,NULL,NULL,NULL,NULL,node3,NULL,NULL,NULL,NULL);
    nodes[4] = init_aco_node(0,0,4,node4,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    nodes[5] = init_aco_node(0,0,5,NULL,NULL,NULL,NULL,NULL,NULL,node5,NULL,NULL,NULL,NULL);
    
    
    
    int** adjacency_matrix = (int**)malloc(sizeof(int*)*n_nodes);
    int** same_params = (int**)malloc(sizeof(int*)*n_nodes);
    for(i = 0; i < n_nodes; i++){
        adjacency_matrix[i] = (int*)calloc(n_nodes,sizeof(int));
        same_params[i] = (int*)calloc(n_nodes,sizeof(int));
    }
    
    
    adjacency_matrix[0][1] = 1;
    adjacency_matrix[1][2] = 1;
    adjacency_matrix[2][3] = 1;
    adjacency_matrix[3][4] = 1;
    adjacency_matrix[4][5] = 1;
    
    
    aco_superstruct* s = init_aco_superstruct(nodes,adjacency_matrix,same_params,n_nodes);// auto generate where to store the output
    generate_parameters_array_to_superstruct(s);// generate where the arrays to store weights and biases
    generate_aco_parameters_array_to_superstruct(s, 30);// generate the weights and biases particles
    generate_aco_pheromone_parameters_array_to_superstruct(s, 10.0);// generate the array for the pheromone for each particle
    aco_superstruct_activate_heuristic(s,100,0.99,1.0,1.0,1.0);
    generate_aco_superstruct_bit_arrays(s);// generate the structs to store 3 subnetworks path
    init_aco_superstruct_general_stuff(s,26,250,10,128,0,10,0.95,0.5,0.5,8,0.7);// init general stuff for aco algorithm
    init_aco_superstruct_eiwoa_stuff(s,1.0,2.0,1.0,1.0,2.0, 5,-5);//init eiwoa params
    
    set_aco_superstruct_k_percentage(s,0.5);
    node4->k_percentage = 1;
    
    // copying the network configuration threads times
    aco_superstruct** ss = (aco_superstruct**)malloc(sizeof(aco_superstruct*)*threads);
    ss[0] = s;
    
    for(i = 1; i < threads; i++){
        ss[i] = copy_aco_superstruct_with_only_storing_params(s);
    }
    
    // Putting the data in float* vectors
    read_file_in_char_vector(ksource,filename,&file_size);
    float* inputs = (float*)calloc(instances*fcl_input1_size,sizeof(float));
    float* outputs = (float*)calloc(instances*fcl_output3_size,sizeof(float));
    
    for(i = 0; i < instances; i++){
        for(j = 0; j < fcl_input1_size+1; j++){
            temp[0] = ksource[0][i*(fcl_input1_size+1)+j];
            if(j == fcl_input1_size)
                outputs[i*fcl_output3_size+atoi(temp)] = 1;
            else
                inputs[i*fcl_input1_size+j] = atof(temp);
        }
    }
    
    // instantiate the indices for the input/output shuffling
    int* indices = (int*)calloc(instances/divisor_of_instances,sizeof(int));
    
    for(i = 0; i < instances/divisor_of_instances; i++){
        indices[i] = i;
    }
    
    // for the heuristic check always the best global ever got and the best before
    double global_best = -1, global_best_before = -2;
    
    
    // set the error for the edge popup algorithm
    int* error_flags = (int*)malloc(sizeof(int));
    error_flags[0] = FOCAL_LOSS;
    edge_popup_trainer* e = init_edge_popup_trainer(ss,threads,0,mini_batch_size,error_flags,0.002,0.00000001,0.975,0.999);
    
    
    for(i = 0; i < threads; i++){
        add_superstruct_storing_error_array_according_to_index(e,i);// d storing for each superstruct
        add_superstruct_weight_like_array_according_to_index(e,i);// d scores for each superstruct
    }
    
    add_superstruct_weight_like_array_according_to_index(e,0);// scores for main superstruct
    add_superstruct_weight_like_array_according_to_index(e,0);// m (nadam) for main superstruct
    add_superstruct_weight_like_array_according_to_index(e,0);// n (nadam) for main superstruct
    add_superstruct_weight_like_int_array_according_to_index(e,0);// indices for main superstruct
    add_superstruct_storing_used_output(e);// used outputs creation

    
    
    
    for(i = 0; i < threads; i++){
        assign_bp_storing_params_according_to_index(e, i);// assigning d storing
        assign_edge_popup_params_according_to_index(e, i);// assigning scores, d weights, indices
        assign_superstruct_storing_used_output_according_to_index(e,i);// assigning used_outputs
    }
    
    init_aco_superstruct_scores(e->s[0]);// init the indices of the scores
    set_total_aco_superstruct_used_output(e->s[0]);// init the used outputs
    
    int epochs = 10;
    for(i = 0; i < epochs; i++){
        printf("Epoch: %d/%d\n",i,epochs);
        shuffle_int_array(indices,instances/divisor_of_instances);
        
        float* network_output = get_multi_output_from_multi_input_aco_superstruct_ff_edge_popup(ss,inputs,threads,instances/divisor_of_instances);
        double error = value_the_network(network_output,outputs,instances/divisor_of_instances,10*instances/divisor_of_instances);
        printf("%lf\n",error);
        free(network_output);
        
        for(j = 0; j < (instances/divisor_of_instances); j+=mini_batch_size){
            ff_loss_bp_edge_popup(e, inputs,outputs,  mini_batch_size, indices+j);
            multi_thread_sum_partial_derivatives_update_reset(e, threads, threads);
            multi_thread_sort_scores(e,16);
            update_edge_popup_parameters(e);
        }
        
        
    }
    

    
    free_edge_popup_trainer(e);
    
    /*
    // training
    for(i = 0; i < s->number_of_iterations; i++){
        // reset the current pheromone best
        double ph = -1;
        s->bit_arrays[1]->pheromone = 0;
        s->bit_arrays[2]->pheromone = 0;
        printf("iteration number: %d/%d\n",i,s->number_of_iterations);
        for(j = 0; j < s->n_ants; j++){
            // generating a subnetwork
            aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight(s);
            for(z = 0; z < threads; z++){
                copy_aco_superstruct_activation(ss[z],s);
            }
            double pheromone = ((double)(instances/divisor_of_instances*10))/get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss(ss,inputs,outputs,threads,instances/divisor_of_instances);
            if(pheromone>ph){
                ph = pheromone;
            }
            if(pheromone > global_best)
                global_best = pheromone;
            // setting the pheromone for the current network
            set_local_pheromone(s, pheromone);
            // replace the best local network if the pheromone is better
            check_local_best_pheromone(s);
        }
        
        printf("best pheromone for this iteration: %f\n",ph);
        // increasing the index iteration
        s->iteration_index++;
        count_update_x++;
        // replace the best global if another best has been found
        check_global_best_pheromone(s);
        // update taus
        aco_superstruct_update_taus(s);
        // updating the pheromone
        update_pheromone_according_to_local_best(s);
        // update the current p
        update_current_p(s);
        // in case we update the global best
        if((count_update_x)%s->time_to_update_with_global_best == 0){
            printf("updating best global\n");
            update_pheromone_according_to_global_best(s);
        }

        if(global_best > global_best_before){
            global_best_before = global_best;
            update_heuristics(s);
        }
        // x update
        if((s->iteration_index)%s->number_iterations_for_single_aco_iteration == 0){
            printf("X update\n");
            aco_superstruct_update_params_eiwoa(s);
            update_eiwoa_params(s);
            count_update_x = 0;
            update_heuristics(s);
        }
        
        // value the actual network according to pheromone amount
        aco_superstruct_select_params_according_to_best_pheromone(s);
        float* network_output = get_multi_output_from_multi_input_aco_superstruct_ff(ss,inputs,threads,instances/divisor_of_instances);
        ph = value_the_network(network_output, outputs, instances/divisor_of_instances, instances/divisor_of_instances*fcl_output3_size);
        free(network_output);
        printf("best global pheromone: %lf\n",ph);
        printf("Saving pheromone and weights\n");
        save_weights_and_pheromone_aco_superstruct(s, i);

        
    }
    */

    
    
    // freeing all the resources
    free(ksource[0]);
    free(ksource);
    free(inputs);
    free(outputs);
    free(indices);
    for(i = 0; i < threads; i++){
        free_aco_superstruct(ss[i]);
    }
    free(ss);
    
}
