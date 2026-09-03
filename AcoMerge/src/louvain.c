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


void fullfill_edges_for_vector_i(float** vectors,float* distances, int* neighbours_per_vector,  int n_vectors, int size_vectors, int index, int metric_distance, int k, int starting_point){
    
    int i, n, counter, j;
    for(counter = 0; counter < k; counter++){
        if(neighbours_per_vector[index*k+counter] == -1)
            break;
    }
    for(i = starting_point; i < n_vectors; i++){
        if(i == index)
            continue;
        float dist = 0;
        if(metric_distance == EUCLIDIAN_DISTANCE){
            dist = euclidean_distance(vectors[index],vectors[i],size_vectors);
        }
        else if(metric_distance == COSINE_SIMILARITY){
            dist = cosine_similarity(vectors[index],vectors[i],size_vectors);
        }
        if(counter < k){
            neighbours_per_vector[index*k+counter] = i;
            distances[index*k+counter] = dist;
            counter++;
            if(counter == k){
                int* proper_indices = (int*)calloc(k,sizeof(int));
                for(j = 0; j < k; j++){
                    proper_indices[j] = j;
                }
                merge_sort(distances+index*k, proper_indices, 0, k-1);
                
                for(j = 0; j < k; j++){
                    proper_indices[j] = neighbours_per_vector[index*k+proper_indices[j]];
                }
                
                copy_int_array(proper_indices,neighbours_per_vector+index*k,k);
                free(proper_indices);
                
                merge_sort_pure(distances+index*k, 0, k-1);
            }
        }
        else{
            int n = set_value_of_just_lowest_in_vector(distances+index*k, dist, k);
            if(n != -1){
                neighbours_per_vector[index*k+n] = i;
            }
        }
    }
    
}

void fullfill_edges_for_vector_i_single(float** vectors,float* new_vector, float* distances, int* neighbours_per_vector,  int n_vectors, int size_vectors, int metric_distance, int k, int starting_point){
    
    int i, n, counter, j;
    for(counter = 0; counter < k; counter++){
        if(neighbours_per_vector[counter] == -1)
            break;
    }
    for(i = starting_point; i < n_vectors; i++){
        float dist = 0;
        if(metric_distance == EUCLIDIAN_DISTANCE){
            dist = euclidean_distance(new_vector,vectors[i],size_vectors);
        }
        else if(metric_distance == COSINE_SIMILARITY){
            dist = cosine_similarity(new_vector,vectors[i],size_vectors);
        }
        if(counter < k){
            neighbours_per_vector[counter] = i;
            distances[counter] = dist;
            counter++;
            if(counter == k){
                int* proper_indices = (int*)calloc(k,sizeof(int));
                for(j = 0; j < k; j++){
                    proper_indices[j] = j;
                }
                merge_sort(distances, proper_indices, 0, k-1);
                
                for(j = 0; j < k; j++){
                    proper_indices[j] = neighbours_per_vector[proper_indices[j]];
                }
                
                copy_int_array(proper_indices,neighbours_per_vector,k);
                free(proper_indices);
                
                merge_sort_pure(distances, 0, k-1);
            }
        }
        else{
            int n = set_value_of_just_lowest_in_vector(distances, dist, k);
            if(n != -1){
                neighbours_per_vector[n] = i;
            }
        }
    }
    
}

double get_sigma_from_distance(float* distances, int* neighbours_per_vector,  int n_vectors, int k){
    double sigma = 0, count;
    int i,j;
    for(count = 0, j = 0; j < n_vectors; j++){
        for(i = 0; i < k; i++){
            if(neighbours_per_vector[j*k+i] == -1)
                break;
            sigma+=distances[j*k+i];
            count++;
        }
    }
    return (sigma/count)/sqrtf(2);
}

void set_gaussian_kernels(float* distances, int* neighbours_per_vector, float* gaussian_kernels,  int n_vectors, int k, double sigma){
    int i,j;
    double sigma_squared = sigma*sigma;
    for(j = 0; j < n_vectors; j++){
        for(i = 0; i < k; i++){
            if(neighbours_per_vector[j*k+i] == -1)
                break;
            gaussian_kernels[j*k+i] = exp(-(distances[j*k+i]*distances[j*k+i])/(2.0*sigma_squared));
        }
    }
}

void set_gaussian_kernels_single(float* distances, int* neighbours_per_vector, float* gaussian_kernels,  int n_vectors, int k, double sigma){
    int i,j;
    double sigma_squared = sigma*sigma;
    for(i = 0; i < k; i++){
        if(neighbours_per_vector[i] == -1)
            break;
        gaussian_kernels[i] = exp(-(distances[i]*distances[i])/(2.0*sigma_squared));
    }
}

knnGraph* init_knnGraph(float** vectors, int n_vectors, int size_vectors, int metric_distance, int k){
    knnGraph* knn = (knnGraph*)malloc(sizeof(knnGraph));
    knn->vectors = vectors;
    knn->n_vectors = n_vectors;
    knn->size_vectors = size_vectors;
    knn->k = k;
    knn->metric_distance = metric_distance;
    float* distances = (float*)calloc(k*n_vectors,sizeof(float));
    float* gaussian_kernels = (float*)calloc(k*n_vectors,sizeof(float));
    set_vector_with_value(FLT_MAX,distances,k*n_vectors);
    int* neighbours_per_vector = (int*)calloc(k*n_vectors,sizeof(int));
    set_int_vector_with_value(-1,neighbours_per_vector,k*n_vectors);
    knn->neighbours_per_vector = neighbours_per_vector;
    knn->distances = distances;
    int i;
    for(i = 0; i < n_vectors; i++){
        fullfill_edges_for_vector_i(vectors,distances,neighbours_per_vector,  n_vectors, size_vectors, i, metric_distance, k, 0);
    }
    knn->gaussian_kernels = gaussian_kernels;
    knn->sigma = get_sigma_from_distance( distances, neighbours_per_vector,  n_vectors, k);
    set_gaussian_kernels(distances, neighbours_per_vector, gaussian_kernels,  n_vectors, k, knn->sigma);

    return knn;
}



void modify_knnGraph(knnGraph* knn, int latest_n_newest_vectors){
    int i;
    for(i = 0; i < knn->n_vectors-latest_n_newest_vectors; i++){
        fullfill_edges_for_vector_i(knn->vectors,knn->distances,knn->neighbours_per_vector,  knn->n_vectors, knn->size_vectors, i, knn->metric_distance, knn->k,knn->n_vectors-latest_n_newest_vectors );
    }
    for(; i < knn->n_vectors; i++){
        fullfill_edges_for_vector_i(knn->vectors,knn->distances,knn->neighbours_per_vector,  knn->n_vectors, knn->size_vectors, i, knn->metric_distance, knn->k,0);
    }
    knn->sigma = get_sigma_from_distance( knn->distances, knn->neighbours_per_vector,  knn->n_vectors, knn->k);
    set_gaussian_kernels(knn->distances, knn->neighbours_per_vector, knn->gaussian_kernels,  knn->n_vectors, knn->k, knn->sigma);
} 

void free_knnGraph(knnGraph* knn){
    free(knn->distances);
    free(knn->neighbours_per_vector);
    free(knn->gaussian_kernels);
    free(knn);
    return;
}

double get_node_edges_sum(float* distances,  int* neighbours_per_vector, int k, int index){
    int i;
    double sum = 0;
    for(i = 0; i < k; i++){
        if(neighbours_per_vector[index*k+i] == -1){
            break;
        }
        sum+=distances[index*k+i];
    }
    return sum;
}

LouvainGraph* init_louvainGraph(knnGraph* knn){
    LouvainGraph* lg = (LouvainGraph*)malloc(sizeof(LouvainGraph));
    lg->knn = knn;
    int i;
    double sum = 0;
    for(i = 0; i < knn->n_vectors; i++){
        sum+=get_node_edges_sum(knn->gaussian_kernels,  knn->neighbours_per_vector, knn->k, i);
    }
    lg->sum_all_edges = sum;
    lg->n_levels = 1;
    lg->n_communities = (int*)malloc(sizeof(int));
    lg->n_communities[0] = knn->n_vectors;
    lg->c = (Community***)malloc(sizeof(Community**));
    lg->c[0] = (Community**)malloc(sizeof(Community*)*knn->n_vectors);
    for(i = 0; i < knn->n_vectors; i++){
        lg->c[0][i] = create_community(0, i, i, knn->k, knn->gaussian_kernels, knn->neighbours_per_vector, lg->sum_all_edges);
    }
    return lg;
}

Community* create_community(int level, int index, int state_index, int k, float* edges, int* neighbours_per_vector, double sum_all_edges){
    Community* c = (Community*)malloc(sizeof(Community));
    c->n_nodes = 1;
    c->level = level;
    c->index = index;
    c->node_indices = (int*)malloc(sizeof(int));
    c->node_indices[0] = state_index;
    c->sum_internal_edges = 0;
    c->sum_all_edges = 0;
    sum_internal_and_all_edges(&c->sum_internal_edges, &c->sum_all_edges, c->n_nodes, c->node_indices, edges, neighbours_per_vector, k);
    c->modularity = c->sum_internal_edges/(2.0*sum_all_edges) - (c->sum_all_edges*c->sum_all_edges)/(4.0*sum_all_edges*sum_all_edges);
    c->n_children = 0;
    c->children = NULL;
    c->parent = NULL;
    return c;
}

// it merges only if modularity increases, returns a new community if merges, null otherwise
// a is where we move, b is to where a is moved, if b has a parent obviously the parent of b is where we are moving it
double  get_modularity_from_community_merging(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges,  int n_visited_communities_for_this_node, int* visited_communities_for_this_node, int* current_community_for_node){
    if(b->parent != NULL)
        b = b->parent;
    int i,j,z, w;

    for(i = 0; i < n_visited_communities_for_this_node; i++){
        if(b->index == visited_communities_for_this_node[i])
            return -100;
    }
    
    if(a->parent != NULL && a->parent->index == b->index){
        return -100;
    }
    
    (*current_community_for_node) = b->index;
    double internal_sum = 0;
    for(i = 0; i < a->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[a->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < b->n_nodes; j++){
                 if(w == b->node_indices[j]){
                     internal_sum += edges[a->node_indices[i]*k+z];
                 }
            }
        }
    }
    for(i = 0; i < b->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[b->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < a->n_nodes; j++){
                 if(w == a->node_indices[j]){
                     internal_sum += edges[b->node_indices[i]*k+z];
                 }
            }
        }
    }
    
    // new modularity of b
    double modularity = (a->sum_internal_edges+b->sum_internal_edges+internal_sum)/(2.0*sum_all_edges) - ((a->sum_all_edges+b->sum_all_edges)*(a->sum_all_edges+b->sum_all_edges)/(4.0*sum_all_edges*sum_all_edges));
    return modularity-b->modularity;
}

double  get_modularity_from_community_merging_single(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges,  int n_visited_communities_for_this_node, int* visited_communities_for_this_node, int* current_community_for_node){
    if(b->parent != NULL)
        b = b->parent;
    int i,j,z, w;

    for(i = 0; i < n_visited_communities_for_this_node; i++){
        if(b->index == visited_communities_for_this_node[i])
            return -100;
    }
    
    if(a->parent != NULL && a->parent->index == b->index){
        return -100;
    }
    
    (*current_community_for_node) = b->index;
    double internal_sum = 0;
    for(i = 0; i < a->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[z]; z < k && w != -1; z++){
            for(j = 0; j < b->n_nodes; j++){
                 if(w == b->node_indices[j]){
                     internal_sum += edges[z];
                 }
            }
        }
    }
    
    // new modularity of b
    double modularity1 = (b->sum_internal_edges)/(2.0*(sum_all_edges+a->sum_all_edges)) - ((b->sum_all_edges)*(b->sum_all_edges)/(4.0*(sum_all_edges+a->sum_all_edges)*(sum_all_edges+a->sum_all_edges)));
    double modularity2 = (b->sum_internal_edges+internal_sum)/(2.0*(sum_all_edges+a->sum_all_edges)) - ((a->sum_all_edges+b->sum_all_edges)*(a->sum_all_edges+b->sum_all_edges)/(4.0*(sum_all_edges+a->sum_all_edges)*(sum_all_edges+a->sum_all_edges)));
    return modularity1-modularity2;
}


double get_modularity_of_community_without_a(Community* a, float* edges, int* neighbours_per_vector, int k, double sum_all_edges){
    if(a->parent == NULL)
        return a->modularity;
    int i,j,z,w;
    Community* b = a->parent;
    double internal_sum = 0;
    for(i = 0; i < a->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[a->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < b->n_nodes; j++){
                if(b->node_indices[j] == a->node_indices[i])
                    continue;
                 if(w == b->node_indices[j]){
                     internal_sum += edges[a->node_indices[i]*k+z];
                 }
            }
        }
    }
    for(i = 0; i < b->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[b->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < a->n_nodes; j++){
                if(b->node_indices[i] == a->node_indices[j])
                    continue;
                 if(w == a->node_indices[j]){
                     internal_sum += edges[b->node_indices[i]*k+z];
                 }
            }
        }
    }
    double modularity = (-a->sum_internal_edges+b->sum_internal_edges-internal_sum)/(2.0*sum_all_edges) - ((-a->sum_all_edges+b->sum_all_edges)*(-a->sum_all_edges+b->sum_all_edges)/(4.0*sum_all_edges*sum_all_edges));

    return b->modularity - modularity;
}

void sum_internal_and_all_edges(float* internal_edges, float* all_edges, int n_nodes, int* node_indices, float* edges, int* neighbours_per_vector, int k){
    int i,j, z, w;
    for(i = 0; i < n_nodes; i++){
        for(j = 0, z = neighbours_per_vector[node_indices[i]*k+j]; z != -1 && j < k; j++){
            if(z == node_indices[i])
                continue;
            (*all_edges)+=edges[node_indices[i]*k+j];
            for(w = 0; w < n_nodes; w++){
                if(i == w)
                    continue;
                if(z == node_indices[w])
                    (*internal_edges)+=edges[node_indices[i]*k+j];
            }
        }
    }
}

void free_community(Community* c){
    if(c == NULL)
        return;
    free(c->node_indices);
    free(c->children);
    free(c);
    return;
}

void free_LouvainGraph(LouvainGraph* lg){
    int i,j;
    for(i = lg->n_levels-1; i >= 0; i--){
        for(j = 0; j < lg->n_communities[i]; j++){
            if(i == lg->c[i][j]->level)
                free_community(lg->c[i][j]);
        }
        free(lg->c[i]);
    }
    free(lg->n_communities);
    free(lg->c);
    free_knnGraph(lg->knn);
    free(lg);
}

Community* merge_communities(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges){
    int i,j,z, w;
    if(a->parent != NULL){
        Community* a_parent = a->parent;
        Community** children = (Community**)malloc(sizeof(Community*)*(a_parent->n_children-1));
        for(i = 0, j = 0; i < a_parent->n_children-1; i++, j++){
            if(a_parent->children[i]->index == a->index){
                j++;
            }
            children[i] = a_parent->children[j];
        }
        free(a_parent->children);
        a_parent->children = children;
    }
    
    Community* ret = NULL;
    
    if(b->parent != NULL){
        b = b->parent;
        b->node_indices = (int*)realloc(b->node_indices,sizeof(int)*(b->n_nodes+a->n_nodes));
        copy_int_array(a->node_indices,b->node_indices+b->n_nodes,a->n_nodes);
        b->n_children++;
        b->children = (Community**)realloc(b->children,sizeof(Community*)*b->n_children);
        b->children[b->n_children-1] = a;
        b->sum_internal_edges += a->sum_internal_edges;
        b->sum_all_edges += a->sum_all_edges;
    }
    else{
        ret = (Community*)malloc(sizeof(Community));
        ret->n_nodes = b->n_nodes;
        ret->node_indices = (int*)calloc(b->n_nodes+a->n_nodes,sizeof(int));
        copy_int_array(b->node_indices,ret->node_indices,b->n_nodes);
        copy_int_array(a->node_indices,ret->node_indices+b->n_nodes,a->n_nodes);
        ret->parent = NULL;
        ret->n_children = 2;
        ret->children = (Community**)malloc(sizeof(Community*)*2);
        ret->children[0] = a;
        ret->children[1] = b;
        ret->sum_internal_edges = a->sum_internal_edges+b->sum_internal_edges;
        ret->sum_all_edges = a->sum_all_edges+b->sum_all_edges;
        b->parent = ret;
        b = ret;
    }

    
    double internal_sum = 0;
    for(i = 0; i < a->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[a->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < b->n_nodes; j++){
                 if(w == b->node_indices[j]){
                     internal_sum += edges[a->node_indices[i]*k+z];
                 }
            }
        }
    }
    for(i = 0; i < b->n_nodes; i++){
        for(z = 0, w = neighbours_per_vector[b->node_indices[i]*k+z]; z < k && w != -1; z++){
            for(j = 0; j < a->n_nodes; j++){
                 if(w == a->node_indices[j]){
                     internal_sum += edges[b->node_indices[i]*k+z];
                 }
            }
        }
    }
    
    b->n_nodes+=a->n_nodes;
    b->sum_internal_edges+=internal_sum;
    b->modularity = (b->sum_internal_edges)/(2.0*sum_all_edges) - ((b->sum_all_edges)*(b->sum_all_edges)/(4.0*sum_all_edges*sum_all_edges));
    a->parent = b;
    return ret;
    
    
}


    

// it returns n+1 size float array where n is the number of communities at the biggest level
// it returns for each community the modularity gain
float* get_community_belonging_modularity_for_new_state(LouvainGraph* lg, float* state, int k){
    int i;
    int* neighbours_per_vector = (int*)calloc(k,sizeof(int));
    set_int_vector_with_value(-1,neighbours_per_vector,k);
    float* distances = (float*)calloc(k,sizeof(int));
    float* gaussian_kernels = (float*)calloc(k,sizeof(int));
    set_vector_with_value(FLT_MAX,distances, k);
    fullfill_edges_for_vector_i_single(lg->knn->vectors,state, distances, neighbours_per_vector,  lg->knn->n_vectors, lg->knn->size_vectors, lg->knn->metric_distance, k, 0);
    set_gaussian_kernels_single(distances, neighbours_per_vector, gaussian_kernels,  lg->knn->n_vectors, k, lg->knn->sigma);
    double sum = 0;
    for(i = 0; i < k && neighbours_per_vector[i] != -1; i++){
        sum+=gaussian_kernels[i];
    }
    Community* c = create_community(lg->n_levels-1,lg->n_levels*lg->knn->n_vectors+1,lg->n_levels*lg->knn->n_vectors+1,k, gaussian_kernels, neighbours_per_vector, lg->sum_all_edges+sum);
    float* ret = (float*)calloc(lg->n_communities[lg->n_levels-1]+1,sizeof(float));
    int* visited_communities_for_this_node = (int*)calloc(lg->n_communities[lg->n_levels-1],sizeof(int));
    double mod = c->modularity;
    for(i = 0; i < lg->n_communities[lg->n_levels-1]; i++){
        int current_community_for_node = 0;
        double incr = get_modularity_from_community_merging_single(c, lg->c[lg->n_levels-1][i], k, gaussian_kernels, neighbours_per_vector, lg->sum_all_edges,  i, visited_communities_for_this_node, &current_community_for_node);
        visited_communities_for_this_node[i] = current_community_for_node;
        ret[i] = incr;
    }
    ret[i] = mod;
    


    free(visited_communities_for_this_node);
    free(gaussian_kernels);
    free(neighbours_per_vector);
    free(distances);
    return ret;

}



void run_Louvain(LouvainGraph* lg){
    int i,j, k,n_communities;
    for(n_communities = 0, i = 0; i < lg->n_levels; i++){
        n_communities+=lg->n_communities[i];
    }
    for(i = 0; i < lg->n_levels; i++){
        int new_community = 0;
        int* communities = (int*)calloc(lg->n_communities[i],sizeof(int));
        for(j = 0; j < lg->n_communities[i]; j++){
            communities[j] = j;
        }
        
        shuffle_int_array(communities,lg->n_communities[i]);
        
        for(j = 0; j < lg->n_communities[i]; j++){
            double mod = get_modularity_of_community_without_a(lg->c[i][communities[j]], lg->knn->gaussian_kernels, lg->knn->neighbours_per_vector, lg->knn->k, lg->sum_all_edges);
            double max = -100;
            int index = -1;
            int n_visited_communities_for_this_node = 0;
            int* visited_communities_for_this_node = (int*)calloc(lg->n_communities[i],sizeof(int));
            for(k = 0; k < lg->n_communities[i]; k++){
                
                if(k == communities[j])
                    continue;
                int current_community = 0;

                double incr = get_modularity_from_community_merging(lg->c[i][communities[j]],lg->c[i][k], lg->knn->k, lg->knn->gaussian_kernels,lg->knn->neighbours_per_vector, lg->sum_all_edges, n_visited_communities_for_this_node, visited_communities_for_this_node, &current_community);
                if(incr == -100)
                    continue;
                visited_communities_for_this_node[n_visited_communities_for_this_node] = current_community;
                n_visited_communities_for_this_node++;
                if(incr > mod && incr > max){
                    index = k;
                    max = incr;
                }
            }
            free(visited_communities_for_this_node);
            if(index != -1 && max != -100){
                new_community = 1;
                Community* new_community = merge_communities(lg->c[i][communities[j]], lg->c[i][index], lg->knn->k, lg->knn->gaussian_kernels, lg->knn->neighbours_per_vector, lg->sum_all_edges);
                if(new_community != NULL){
                    new_community->level = i+1;
                    new_community->index = n_communities;
                    n_communities++;
                    if(i == lg->n_levels-1){
                        lg->n_levels++;
                        lg->n_communities = (int*)realloc(lg->n_communities,sizeof(int)*lg->n_levels);
                        lg->n_communities[i+1] = 1;
                        lg->c = (Community***)realloc(lg->c,sizeof(Community**)*lg->n_levels);
                        lg->c[i+1] = (Community**)malloc(sizeof(Community*));
                        lg->c[i+1][0] = new_community;
                    }
                    else{
                        lg->n_communities[i+1]++;
                        lg->c[i+1] = (Community**)realloc(lg->c[i+1],sizeof(Community*)*lg->n_communities[i+1]);
                        lg->c[i+1][lg->n_communities[i+1]-1] = new_community;
                    }
                    
                }
            }
            
            else{
                if(i == lg->n_levels-1){
                    lg->n_levels++;
                    lg->n_communities = (int*)realloc(lg->n_communities,sizeof(int)*lg->n_levels);
                    lg->n_communities[i+1] = 1;
                    lg->c = (Community***)realloc(lg->c,sizeof(Community**)*lg->n_levels);
                    lg->c[i+1] = (Community**)malloc(sizeof(Community*));
                    lg->c[i+1][0] = lg->c[i][communities[j]];
                }
                else{
                    lg->n_communities[i+1]++;
                    lg->c[i+1] = (Community**)realloc(lg->c[i+1],sizeof(Community*)*lg->n_communities[i+1]);
                    lg->c[i+1][lg->n_communities[i+1]-1] = lg->c[i][communities[j]];
                }
            }
        }
        
        free(communities);
        if(new_community == 0)
            break;
        
    }
}

double get_modularity(LouvainGraph* lg){
    int i;
    double modularity = 0;
    for(i = 0; i < lg->n_communities[lg->n_levels-1]; i++){
        modularity+=lg->c[lg->n_levels-1][i]->modularity;
    }
    return modularity;
}






