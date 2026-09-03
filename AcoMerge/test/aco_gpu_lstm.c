#include "../src/ezspark_gpu.h"

double value_the_network(float* network_output, float* outputs, int instances, int size){
    int i;
    // getting the pheromone
    double error = 0;
    for(i = 0; i < size; i++){
        error+=focal_loss(network_output[i],outputs[i],2.0);
    }
    error = ((double)(10*instances))/error;
    if(error != error)
        error = -1;
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
    int n_nodes = 88;
    int count_update_x = 0;
    int divisor_of_instances = 1;
    
    // number of threads to use
    int threads = 1;
    
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
    
    
    int i,j,z, w, ww;
    
    // the network
    aco_node** nodes = (aco_node**)malloc(sizeof(aco_node*)*n_nodes);
    
    aco_fcl** fcls = (aco_fcl**)malloc(sizeof(aco_fcl*)*28);
    aco_activation** activations =(aco_activation**)malloc(sizeof(aco_activation*)*28);
    aco_lstm** lstms = (aco_lstm**)malloc(sizeof(aco_lstm*)*28);
    
    
    for(i = 0; i < 28; i++){
        fcls[i] = init_aco_fcl(28,fcl_output1_size,i*3,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
        lstms[i] = init_aco_lstm(i+1,fcl_output1_size,fcl_output1_size,28,i*3+1,0,0,0,0,0,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
        activations[i] = init_aco_activation(RELU,N_ACTIVATION_FUNCTIONS_1_0,fcl_output1_size,fcl_output1_size,i*3+2,0,0,0,NULL,NULL,NULL,generate_all_activations());
    }
    
    
    
    aco_fcl* node2 = init_aco_fcl(fcl_input2_size,fcl_output2_size,84,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node3 = init_aco_activation(RELU,N_ACTIVATION_FUNCTIONS_1_0,fcl_output2_size,fcl_output2_size,85,0,0,0,NULL,NULL,NULL,generate_all_activations());
    aco_fcl* node4 = init_aco_fcl(fcl_input3_size,fcl_output3_size,86,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node5 = init_aco_activation(RELU,0,fcl_output3_size,fcl_output3_size,87,0,0,0,NULL,NULL,NULL,NULL);
    
    for(i = 0; i < 28; i++){
        nodes[i*3] = init_aco_node(0,0,i*3,fcls[i],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
        nodes[i*3+1] = init_aco_node(0,0,i*3+1,NULL,NULL,lstms[i],NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
        nodes[i*3+2] = init_aco_node(0,0,i*3+2,NULL,NULL,NULL,NULL,NULL,NULL,activations[i],NULL,NULL,NULL,NULL);
    }
    free(fcls);
    free(lstms);
    free(activations);
    
    nodes[84] = init_aco_node(0,0,84,node2,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    nodes[85] = init_aco_node(0,0,85,NULL,NULL,NULL,NULL,NULL,NULL,node3,NULL,NULL,NULL,NULL);
    nodes[86] = init_aco_node(0,0,86,node4,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    nodes[87] = init_aco_node(0,0,87,NULL,NULL,NULL,NULL,NULL,NULL,node5,NULL,NULL,NULL,NULL);
    
    
    
    int** adjacency_matrix = (int**)malloc(sizeof(int*)*n_nodes);
    int** same_params = (int**)malloc(sizeof(int*)*n_nodes);
    for(i = 0; i < n_nodes; i++){
        adjacency_matrix[i] = (int*)calloc(n_nodes,sizeof(int));
        same_params[i] = (int*)calloc(n_nodes,sizeof(int));
    }
    
    for(i = 0; i < 28; i++){
        adjacency_matrix[3*i][3*i+2] = 1;
        
        if(i < 27)
            adjacency_matrix[3*i+1][3*i+4] = 1;
        else
            adjacency_matrix[3*i+1][84] = 1;
        adjacency_matrix[3*i+2][3*i+1] = 1;
        if(i < 27){
            same_params[3*(27-i)][0] = 1;// l'ultimo ha gli stessi parametri del primo, il penultimo ha gli stessi parametri del primo e così via
            same_params[3*(27-i)+1][1] = 1;// l'ultimo ha gli stessi parametri del primo, il penultimo ha gli stessi parametri del primo e così via
        }
    }
    
    adjacency_matrix[84][85] = 1;
    adjacency_matrix[85][86] = 1;
    adjacency_matrix[86][87] = 1;
    
    
    
    aco_superstruct* s = init_aco_superstruct(nodes,adjacency_matrix,same_params,n_nodes);// auto generate where to store the output
    generate_parameters_array_to_superstruct(s);// generate where the arrays to store weights and biases
    generate_aco_parameters_array_to_superstruct(s, 30);// generate the weights and biases particles
    generate_aco_pheromone_parameters_array_to_superstruct(s, 10.0);// generate the array for the pheromone for each particle
    aco_superstruct_activate_heuristic(s,100,0.99,1.0,1.0,1.0);
    generate_aco_superstruct_bit_arrays(s);// generate the structs to store 3 subnetworks path
    init_aco_superstruct_general_stuff(s,26,250,10,128,0,10,0.95,0.5,0.5,8,0.7);// init general stuff for aco algorithm
    init_aco_superstruct_eiwoa_stuff(s,1.0,2.0,1.0,1.0,2.0, 5,-5);//init eiwoa params
    
    
    
    // copying the network configuration threads times
    aco_superstruct** ss = (aco_superstruct**)malloc(sizeof(aco_superstruct*)*threads);//alloc
    ss[0] = s;
    
    
    for(i = 1; i < threads; i++){
        ss[i] = copy_aco_superstruct_with_only_storing_params(s);
    }
    
    
    
    // Putting the data in float* vectors
    read_file_in_char_vector(ksource,filename,&file_size);
    float* inputs = (float*)calloc(instances*fcl_input1_size,sizeof(float));//alloc
    float* outputs = (float*)calloc(instances*fcl_output3_size,sizeof(float));//alloc
    
    for(i = 0; i < instances; i++){
        for(j = 0; j < fcl_input1_size+1; j++){
            temp[0] = ksource[0][i*(fcl_input1_size+1)+j];
            if(j == fcl_input1_size)
                outputs[i*fcl_output3_size+atoi(temp)] = 1;
            else
                inputs[i*fcl_input1_size+j] = atof(temp);
        }
    }
    
    
    
    instances = 64;
    
    int n_paths = 16;
    
    
    for(i = 1; i < n_paths; i++){
        copy_array(inputs,inputs+i*instances*fcl_input1_size,instances*fcl_input1_size);
    }
    
    uint8_t* paths_ok = (uint8_t*)calloc(instances*n_paths,sizeof(uint8_t));//alloc
    
    for(i = 0; i < instances*n_paths; i++){
        paths_ok[i] = 1;
    }
    
    
    
    int gpu_batch_size = get_gpu_first_dimension_size();
    aco_superstruct_gpu* super = init_aco_superstruct_gpu(s,gpu_batch_size,n_paths,"../src/kernels_gpu.cl");

    
    // for the heuristic check always the best global ever got and the best before
    double global_best = -1, global_best_before = -2;
    
    uint8_t* temp0, *temp1, *temp2;
    temp0 = s->bit_arrays[0]->data;
    temp1 = s->bit_arrays[1]->data;
    temp2 = s->bit_arrays[2]->data;
	float** cpus = (float**)malloc(sizeof(float*)*n_paths);//alloc
    
    // training
    for(i = 0; i < s->number_of_iterations; i++){
        // reset the current pheromone best
        double ph = -1;
        s->bit_arrays[1]->pheromone = 0;
        s->bit_arrays[2]->pheromone = 0;
        printf("iteration number: %d/%d\n",i,s->number_of_iterations);
        
        for(z = 0, j = 0; j < s->n_ants; j++, z++, z = z%n_paths){
            s->bit_arrays[2]->data = super->paths+z*super->s->bit_arrays[2]->size;
            // generating a subnetwork
            
            
            
            aco_superstruct_select_params_according_to_pheromone_exploration_levy_flight(s);
            

            
            for(w = 0; w < threads; w++){
                copy_aco_superstruct_activation(ss[w],s);
            }
            
            cpus[z] = get_multi_output_from_multi_input_aco_superstruct_ff(ss,inputs,threads,instances/divisor_of_instances);

            
            
            if(z == n_paths-1){
                aco_superstruct_complete_multiple_instances_feed_forward_gpu(super,paths_ok,inputs);
                float* gpus = aco_superstruct_get_outputs_gpu(super,paths_ok);
                //exit(0);
                for(w = 0; w < n_paths; w++){
					for(ww = 0; ww < fcl_output3_size*instances; ww++){
						printf("{%f , %f} - ",cpus[w][ww],gpus[w*fcl_output3_size*instances+ww]);
					}
					printf("\n");
				}
                
                free(gpus);
                exit(0);
                break;
            }
            
            
            
            
            
            
            // here set the pheromone
            // double pheromone = 0;
            
            // if(pheromone>ph){
            //     ph = pheromone;
            // }
            // if(pheromone > global_best)
            //    global_best = pheromone;
            // setting the pheromone for the current network
            // set_local_pheromone(s, pheromone);
            // replace the best local network if the pheromone is better
            // check_local_best_pheromone(s);
            
        }
        
        break;
        
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


    
    
    // freeing all the resources
    s->bit_arrays[0]->data = temp0;
    s->bit_arrays[1]->data = temp1;
    s->bit_arrays[2]->data = temp2;
    free(ksource[0]);
    free(ksource);
    free(inputs);
    free(outputs);
    free_aco_superstruct_gpu(super);
    for(i = 0; i < threads; i++){
        free_aco_superstruct(ss[i]);
    }
    free(paths_ok);
    free(ss);
    free_matrix((void**)cpus,n_paths);
    
}
