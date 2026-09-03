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
#ifndef __UTILS_H__
#define __UTILS_H__

void set_vector_with_value(float value, float* v, int dimension);
void sum_vectors(float* input1, float* input2,float* output, int size);
void copy_array(float* input, float* output, int size);
void print_vector(float* vector, int size);
void print_matrix(float* vector, int rows, int cols);
void print_tensor(float* vector, int channels, int rows, int cols);
void print_4dtensor(float* vector,int as, int channels, int rows, int cols);
void merge_aco_nodes(aco_node** arr, int l, int m, int r);
void merge_sort_aco_nodes(aco_node** arr, int l, int r);
void put_concatenation_first(aco_node** nodes, int n_nodes);                      
void free_matrix(void** matrix, int rows);
void copy_int_array(int* v, int* copy, int size);
int int_min(int a, int b);
int int_max(int a, int b);
void convert_data(void* ptr, uint64_t size, uint64_t len);
int is_little_endian();
void swap_array_bytes_order(void* ptr, uint64_t size, uint64_t len);
void reverse_ptr(void* ptr, uint64_t size);
void set_char_vector_with_zero(char* c, uint64_t size);
double sum_over_input(float* input, int size);
void copy_char_array(char* input, char* output, int size);
int read_file_in_char_vector(char** ksource, char* fname, int* size);
void merge_sort(float* arr, int* index, int left, int right);
void merge(float* arr, int* index, int left, int mid, int right);
void set_int_vector_with_value(int value, int* v, int dimension);
void set_vector_according_to_indices(float* input, float* output, uint8_t* indices, uint64_t input_size, uint64_t output_and_indices_size);
int shuffle_int_array(int* m,int n);
char* itoa_n(int i, char b[]);
void min_max_normalization(float* input, float* output, int size);
void set_frequencies_superstruct(float* output, float* frequencies, int size);
void copy_uint8_t_array(uint8_t* v, uint8_t* copy, int size);
void set_uint8_t_vector_with_value(uint8_t value, uint8_t* v, int dimension);
void set_vector_according_to_index(float* input, float* output, uint8_t index, uint64_t input_size, uint64_t output_and_indices_size);
void get_randomness_seed(unsigned int seed);
void get_randomness();
void print_int_vector(int* vector, int size);
void print_int_matrix2d(int** vector, int rows, int cols);
int there_are_missing_nodes(aco_node** nodes, int n_nodes);
int after_splitter_there_is_splitter_or_concatenation(aco_node* n);
void merge_sort_pure(float* arr, int left, int right);
void merge_pure(float* arr, int left, int mid, int right);
int set_value_of_just_lowest_in_vector(float* vector, float value, int size);

#endif
