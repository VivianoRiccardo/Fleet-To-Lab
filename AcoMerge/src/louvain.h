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

#ifndef __LOUVAIN_H__
#define __LOUVAIN_H__




typedef struct knnGraph {
	int n_vectors, size_vectors, k, metric_distance;
	double sigma;
	float** vectors;//n_vectors X vectors_size
	float* distances; // n_vectors * k, the edges for each vector
	float* gaussian_kernels; // n_vectors * k, the edges for each vector
	int* neighbours_per_vector; // n_vectors * k (indicate the index of the most adjacient vectors)
}knnGraph;

typedef struct Community Community;

struct Community {
    int n_nodes, level, index, n_children;
    int* node_indices;
    float sum_internal_edges;
    float sum_all_edges;
    float modularity;
    Community** children;
    Community* parent;
};

typedef struct LouvainGraph{
	int n_levels;
	float sum_all_edges;
	int* n_communities; // n_levels
	Community*** c; //n_levels X n_communities[i]
	knnGraph* knn;
}LouvainGraph;

void fullfill_edges_for_vector_i(float** vectors,float* distances, int* neighbours_per_vector,  int n_vectors, int size_vectors, int index, int metric_distance, int k, int starting_point);
double get_sigma_from_distance(float* distances, int* neighbours_per_vector,  int n_vectors, int k);
void set_gaussian_kernels(float* distances, int* neighbours_per_vector, float* gaussian_kernels,  int n_vectors, int k, double sigma);
knnGraph* init_knnGraph(float** vectors, int n_vectors, int size_vectors, int metric_distance, int k);
void modify_knnGraph(knnGraph* knn, int latest_n_newest_vectors);
void free_knnGraph(knnGraph* knn);
double get_node_edges_sum(float* distances,  int* neighbours_per_vector, int k, int index);
LouvainGraph* init_louvainGraph(knnGraph* knn);
Community* create_community(int level, int index, int state_index, int k, float* edges, int* neighbours_per_vector, double sum_all_edges);
double  get_modularity_from_community_merging(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges,  int n_visited_communities_for_this_node, int* visited_communities_for_this_node, int* current_community_for_node);
double get_modularity_of_community_without_a(Community* a, float* edges, int* neighbours_per_vector,int k, double sum_all_edges);
void sum_internal_and_all_edges(float* internal_edges, float* all_edges, int n_nodes, int* node_indices, float* edges, int* neighbours_per_vector, int k);
void free_community(Community* c);
void free_LouvainGraph(LouvainGraph* lg);
Community* merge_communities(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges);
void run_Louvain(LouvainGraph* lg);
double get_modularity(LouvainGraph* lg);
void fullfill_edges_for_vector_i_single(float** vectors,float* new_vector, float* distances, int* neighbours_per_vector,  int n_vectors, int size_vectors, int metric_distance, int k, int starting_point);
void set_gaussian_kernels_single(float* distances, int* neighbours_per_vector, float* gaussian_kernels,  int n_vectors, int k, double sigma);
double  get_modularity_from_community_merging_single(Community* a, Community* b, int k, float* edges, int* neighbours_per_vector, double sum_all_edges,  int n_visited_communities_for_this_node, int* visited_communities_for_this_node, int* current_community_for_node);



#endif
