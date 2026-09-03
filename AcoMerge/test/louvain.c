#include "../src/ezspark_gpu.h"


int main(){
    srand(time(NULL));
    int state_size = 20;
    int number_of_states = 100;
    int k = 10;
    int i,j;
    float** states = (float**)malloc(sizeof(float*)*number_of_states);
    for(i = 0; i < number_of_states; i++){
        states[i] = (float*)calloc(state_size,sizeof(float));
        for(j = 0; j < state_size; j++){
            if(normalized_random() < 0.5){
                states[i][j] = 1;
            }
        }
    }
    
    knnGraph* knn = init_knnGraph(states,number_of_states,state_size,EUCLIDIAN_DISTANCE,k);
    
    
    LouvainGraph* lg = init_louvainGraph(knn);
    printf("Modularity: %lf\n",get_modularity(lg));
    run_Louvain(lg);
    printf("Modularity: %lf\n",get_modularity(lg));
    
    free_LouvainGraph(lg);
    free_matrix((void**)states,number_of_states);
}
