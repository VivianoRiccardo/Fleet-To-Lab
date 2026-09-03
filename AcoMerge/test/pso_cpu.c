#include "../src/ezspark.h"

double value_the_network(float* network_output, float* outputs, int instances, int size){
    int i,j;
    int network_index, real_index;
    float network_best, real_best;
    double accuracy = 0;
    for(i = 0; i < instances; i++){
        network_index = 0;
        real_index = 0;
        network_best = network_output[i*size];
        real_best = outputs[i*size];
        for(j = 1; j < size; j++){
            if(network_output[i*size+j] > network_best){
                network_best = network_output[i*size+j];
                network_index = j;
            }
            if(outputs[i*size+j] > real_best){
                real_best = outputs[i*size+j];
                real_index = j;
            }
        }
        if(real_index == network_index){
            accuracy+=1;
        }
    }
    return accuracy/((double)(instances));
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
    int batch_size = 200;
    
    // number of threads to use
    int threads = 16;
    
    // instances of the training set
    int instances = 50000;
    
    // parameters for reading the file data
    char** ksource = (char**)malloc(sizeof(char*));
    char* filename = "./data/train.bin";
    char temp[2];
    temp[1] = '\0';
    int file_size = 0;
    
    
    int i,j,z,w;
    
    // the network
    aco_node** nodes = (aco_node**)malloc(sizeof(aco_node*)*n_nodes);
    
    aco_fcl* node0 = init_aco_fcl(fcl_input1_size,fcl_output1_size,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node1 = init_aco_activation(RELU,0,fcl_output1_size,fcl_output1_size,1,0,0,0,NULL,NULL,NULL,NULL);
    aco_fcl* node2 = init_aco_fcl(fcl_input2_size,fcl_output2_size,2,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL);
    aco_activation* node3 = init_aco_activation(RELU,0,fcl_output2_size,fcl_output2_size,3,0,0,0,NULL,NULL,NULL,NULL);
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
    
    
    aco_superstruct* s = init_aco_superstruct(nodes,adjacency_matrix,same_params,n_nodes);// auto generate where to store the output 0
    generate_parameters_array_to_superstruct(s);// generate where the arrays to store weights and biases 1
    generate_pso_parameters_array_to_superstruct(s, 100);// generate the weights and biases particles 2
    generate_aco_pheromone_parameters_array_to_superstruct(s, 0.0);// generate the array for the pheromone for each particle 3 
    generate_aco_superstruct_bit_arrays(s);// generate the structs to store 3 subnetworks path 
    generate_aco_pso_y_alphas_array_to_superstruct(s, 0.0);// 4
    
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
    
    // for the heuristic check always the best global ever got and the best before
    double global_best = -1, global_best_before = -2;
    
    s->number_of_iterations = 100;
    s->n_ants = 100;
    set_aco_superstruct_array_to_null(s, 1);
    
    // training
    for(i = 0; i < s->number_of_iterations; i++){
        // reset the current pheromone best
        double ph = -1;
        s->bit_arrays[1]->pheromone = 0;
        s->bit_arrays[2]->pheromone = 0;
        /*for(j = 0; j < s->n_ants; j++){
            aco_superstruct_select_params_according_to_index_pso(s, j);

            float* network_output = get_multi_output_from_multi_input_aco_superstruct_ff(ss,inputs,threads,instances/divisor_of_instances);
            ph = value_the_network(network_output, outputs, instances/divisor_of_instances, fcl_output3_size);
            free(network_output);
            printf("Accuracy: %lf\n",ph);
        }*/
        
        
        printf("iteration number: %d/%d\n",i,s->number_of_iterations);
        for(w = 0; w < instances/divisor_of_instances; w+=batch_size){
			double best_error = 9999999999;
            printf("%d\n",w);
            for(j = 0; j < s->n_ants; j++){
                // generating a subnetwork
                aco_superstruct_select_params_according_to_index_pso(s, j);
                double pheromone = get_multi_output_from_multi_input_aco_superstruct_ff_focal_loss(ss,inputs+fcl_input1_size*w,outputs+fcl_output3_size*w,instances/divisor_of_instances - w < threads?instances/divisor_of_instances - w :threads,instances/divisor_of_instances - w < batch_size?instances/divisor_of_instances - w :batch_size);
                //float* network_output = get_multi_output_from_multi_input_aco_superstruct_ff(ss,inputs+fcl_input1_size*w,instances/divisor_of_instances - w < threads?instances/divisor_of_instances - w :threads,instances/divisor_of_instances - w < batch_size?instances/divisor_of_instances - w :batch_size);
                //double pheromone = value_the_network(network_output, outputs+fcl_output3_size*w,instances/divisor_of_instances - w < threads?instances/divisor_of_instances - w :threads,instances/divisor_of_instances - w < batch_size?instances/divisor_of_instances - w :batch_size);
                //pheromone = (100.0-pheromone)/100;
                s->losses[j] = pheromone/batch_size;
                if(s->losses[j] < best_error)
					best_error = s->losses[j];
                //printf("%f\n",s->losses[j]);
                //free(network_output);
                
            }
            printf("%lf\n",best_error);
            aco_update_pso(s);
        }
        
        // increasing the index iteration
        s->iteration_index++;
        
        aco_update_pso_parameters(s);
        
        if(i == s->number_of_iterations-1){
            for(j = 0; j < s->n_ants; j++){
                aco_superstruct_select_params_according_to_index_pso(s, j);
                float* network_output = get_multi_output_from_multi_input_aco_superstruct_ff(ss,inputs,threads,instances/divisor_of_instances);
                ph = value_the_network(network_output, outputs, instances/divisor_of_instances, fcl_output3_size);
                free(network_output);
                printf("Accuracy: %lf\n",ph);
            }
        }
        
        

        
        

        
    }
    // freeing all the resources
    free(ksource[0]);
    free(ksource);
    free(inputs);
    free(outputs);
    for(i = 0; i < threads; i++){
        free_aco_superstruct(ss[i]);
    }
    free(ss);
    
}
