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


edge_popup_trainer* init_edge_popup_trainer(aco_superstruct** s, int batch_size, int optimizer_flag,int mini_batch_size, int* error_flags, float lr, float epsilon,float m0, float n0){
	int i;
	edge_popup_trainer* e = (edge_popup_trainer*)malloc(sizeof(edge_popup_trainer));
	e->batch_size = batch_size;
	e->mini_batch_size = mini_batch_size;
	e->optimizer_flag = optimizer_flag;
	e->lr = lr;
	e->epsilon = epsilon;
	e->m0 = m0;
	e->n0 = n0;
	e->mt = m0;
	e->nt = n0;
	e->s = s;
	e->n_arrays = (int*)calloc(batch_size,sizeof(int));
	e->int_n_arrays = (int*)calloc(batch_size,sizeof(int));
	e->array_sizes = (uint64_t**)malloc(batch_size*sizeof(uint64_t*));
	e->int_array_sizes = (uint64_t**)malloc(batch_size*sizeof(uint64_t*));
	e->arrays = (float***)malloc(sizeof(float**)*batch_size);
	e->int_arrays = (int***)malloc(sizeof(int**)*batch_size);
	e->error_flags = error_flags;
	for(i = 0; i < batch_size; i++){
		e->array_sizes[i] = NULL;
		e->int_array_sizes[i] = NULL;
		e->arrays[i] = NULL;
		e->int_arrays[i] = NULL;
	}
	
	return e;
}



void free_edge_popup_trainer(edge_popup_trainer* s){
	int i;
	for(i = 0; i < s->batch_size; i++){
		if(s->int_n_arrays[i]){
			free_matrix((void**)s->int_arrays[i],s->int_n_arrays[i]);
		}
		if(s->n_arrays[i]){
			free_matrix((void**)s->arrays[i],s->n_arrays[i]);
		}
	}
	free_matrix((void**)s->array_sizes,s->batch_size);
	free_matrix((void**)s->int_array_sizes,s->batch_size);
	free(s->int_arrays);
	free(s->arrays);
	free(s->n_arrays);
	free(s->int_n_arrays);
	free(s->error_flags);
	free(s);
}


void add_superstruct_storing_error_array_according_to_index(edge_popup_trainer* e, int index){
	if(index < 0 || index >= e->batch_size){
		fprintf(stderr,"Error, the indxe is out of range\n");
		exit(1);
	}
    float* params = (float*)calloc(e->s[0]->array_sizes[0],sizeof(float));
    e->n_arrays[index]++;
    e->array_sizes[index] = (uint64_t*)realloc(e->array_sizes[index],e->n_arrays[index]*sizeof(uint64_t));
    e->arrays[index] = (float**)realloc(e->arrays[index],e->n_arrays[index]*sizeof(float*));
    e->array_sizes[index][e->n_arrays[index]-1] = e->s[0]->array_sizes[0];
    e->arrays[index][e->n_arrays[index]-1] = params;
}

void add_superstruct_weight_like_array_according_to_index(edge_popup_trainer* e, int index){
	if(index < 0 || index >= e->batch_size){
		fprintf(stderr,"Error, the indxe is out of range\n");
		exit(1);
	}
    float* params = (float*)calloc(e->s[0]->array_sizes[1],sizeof(float));
    e->n_arrays[index]++;
    e->array_sizes[index] = (uint64_t*)realloc(e->array_sizes[index],e->n_arrays[index]*sizeof(uint64_t));
    e->arrays[index] = (float**)realloc(e->arrays[index],e->n_arrays[index]*sizeof(float*));
    e->array_sizes[index][e->n_arrays[index]-1] = e->s[0]->array_sizes[1];
    e->arrays[index][e->n_arrays[index]-1] = params;
}

void add_superstruct_weight_like_int_array_according_to_index(edge_popup_trainer* e, int index){
	if(index < 0 || index >= e->batch_size){
		fprintf(stderr,"Error, the indxe is out of range\n");
		exit(1);
	}
    int* params = (int*)calloc(e->s[0]->array_sizes[1],sizeof(int));
    e->int_n_arrays[index]++;
    e->int_array_sizes[index] = (uint64_t*)realloc(e->int_array_sizes[index],e->int_n_arrays[index]*sizeof(uint64_t));
    e->int_arrays[index] = (int**)realloc(e->int_arrays[index],e->int_n_arrays[index]*sizeof(int*));
    e->int_array_sizes[index][e->int_n_arrays[index]-1] = e->s[0]->array_sizes[1];
    e->int_arrays[index][e->int_n_arrays[index]-1] = params;
}

void sort_scores(edge_popup_trainer* e){
	merge_sort_aco_superstruct(e->s[0]);
	set_int_vector_with_value(0,e->int_arrays[0][e->int_n_arrays[0]-1],e->int_array_sizes[0][e->int_n_arrays[0]-1]);
	set_total_aco_superstruct_used_output(e->s[0]);
}


void multi_thread_sort_scores(edge_popup_trainer* e, int threads){
	merge_sort_aco_superstruct_multi_thread(e->s[0], threads);
	set_int_vector_with_value(0,e->int_arrays[0][e->int_n_arrays[0]-1],e->int_array_sizes[0][e->int_n_arrays[0]-1]);
	set_total_aco_superstruct_used_output(e->s[0]);
}
// 0 is bp errors
// 1 is partial derivatives
void assign_bp_storing_params_according_to_index(edge_popup_trainer* e, int index){
	if(e == NULL)
		return;
	if (index < 0 || index >= e->batch_size){
		fprintf(stderr,"Error, index out of range\n");
		exit(1);
	}
	assign_aco_superstruct_storing_partial_derivatives(e->s[index], e->arrays[index][0]);
}

void assign_edge_popup_params_according_to_index(edge_popup_trainer* e, int index){
	if(e == NULL)
		return;
	if (index < 0 || index >= e->batch_size){
		fprintf(stderr,"Error, index out of range\n");
		exit(1);
	}
	assign_aco_superstruct_learnable_params_scores(e->s[index], e->arrays[index][1],e->int_arrays[0][0],e->arrays[0][2]);
}


void set_partial_derivative_to_zero_according_to_index(edge_popup_trainer* e, int index){
	if(e == NULL)	
		return;
	if(index < 0 || index >= e->batch_size){
		return;
	}
	
	set_vector_with_value(0,e->arrays[index][1],e->array_sizes[index][1]);
}

void update_model(edge_popup_trainer* e){
	if(e == NULL)
		return;
	int i;
	for(i = 0; i < e->s[0]->array_sizes[1]; i++){
		nadam(e->arrays[0][2]+i, e->arrays[0][3]+i, e->arrays[0][4]+i, e->arrays[0][2][i],  e->mt, e->nt, e->m0, e->n0, e->mini_batch_size, e->epsilon, e->lr);
	}
}

void update_edge_popup_parameters(edge_popup_trainer* e){
	e->mt*=e->m0;
	e->nt*=e->n0;
}

void add_superstruct_storing_used_output(edge_popup_trainer* e){
	if(e == NULL)
		return;
	int index = 0;
	int* params = (int*)calloc(e->s[0]->array_sizes[0],sizeof(int));
    e->int_n_arrays[index]++;
    e->int_array_sizes[index] = (uint64_t*)realloc(e->int_array_sizes[index],e->int_n_arrays[index]*sizeof(uint64_t));
    e->int_arrays[index] = (int**)realloc(e->int_arrays[index],e->int_n_arrays[index]*sizeof(int*));
    e->int_array_sizes[index][e->int_n_arrays[index]-1] = e->s[0]->array_sizes[0];
    e->int_arrays[index][e->int_n_arrays[index]-1] = params;
}

void assign_superstruct_storing_used_output_according_to_index(edge_popup_trainer* e, int index){
	if(e == NULL)
		return;
	if(index < 0 || index >= e->batch_size)
		return;
	assign_aco_superstruct_storing_params_output_used(e->s[index],e->int_arrays[0][e->int_n_arrays[0]-1]);

}


