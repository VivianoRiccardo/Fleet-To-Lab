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

char* itoa_n(int i, char b[]){
    char const digit[] = "0123456789";
    char* p = b;
    if(i<0){
        *p++ = '-';
        i *= -1;
    }
    int shifter = i;
    do{
        ++p;
        shifter = shifter/10;
    }while(shifter);
    *p = '\0';
    do{
        *--p = digit[i%10];
        i = i/10;
    }while(i);
    return b;
}


void set_vector_according_to_indices(float* input, float* output, uint8_t* indices, uint64_t input_size, uint64_t output_and_indices_size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i, size = input_size/output_and_indices_size;
    for(i = 0; i < input_size; i+= size){
        output[i/size] = input[i+indices[i/size]];
    }
}

void set_vector_according_to_index(float* input, float* output, uint8_t index, uint64_t input_size, uint64_t output_and_indices_size){
    if(input == NULL || output == NULL )
        return;
    int i, size = input_size/output_and_indices_size;
    for(i = 0; i < input_size; i+= size){
        output[i/size] = input[i+index];
    }
}

void min_max_normalization(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    float max = input[0];
    float min = input[0];
    for(i = 1; i < size; i++){
        if(input[i] > max)
            max = input[i];
        if(input[i] < min)
            min = input[i];
    }
    
    for(i = 0; i < size; i++){
        output[i] = (input[i]-min)/(max-min);
    }
} 

void set_vector_with_value(float value, float* v, int dimension){
    if(v == NULL)
        return;
    int i;
    for(i = 0; i < dimension; i++){
        v[i] = value;
    }
}

void set_int_vector_with_value(int value, int* v, int dimension){
    if(v == NULL)
        return;
    int i;
    for(i = 0; i < dimension; i++){
        v[i] = value;
    }
}

void set_uint8_t_vector_with_value(uint8_t value, uint8_t* v, int dimension){
    if(v == NULL)
        return;
    int i;
    for(i = 0; i < dimension; i++){
        v[i] = value;
    }
}

void sum_vectors(float* input1, float* input2,float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]+input2[i];
    }
}

void copy_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    memcpy(output, input, size*sizeof(float));
}

void print_vector(float* vector, int size){
    if(vector == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        printf("%f , ",vector[i]);
    }
    printf("\n");
}

void print_int_vector(int* vector, int size){
    if(vector == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        printf("%d , ",vector[i]);
    }
    printf("\n");
}

void print_matrix(float* vector, int rows, int cols){
    if(vector == NULL)
        return;
    int i,j;
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            printf("%f , ",vector[i*cols+j]);
        }
        printf("\n");
    }
}

void print_int_matrix2d(int** vector, int rows, int cols){
    if(vector == NULL)
        return;
    int i,j;
    for(i = 0; i < rows; i++){
        for(j = 0; j < cols; j++){
            printf("%d , ",vector[i][j]);
        }
        printf("\n");
    }
}

void print_tensor(float* vector, int channels, int rows, int cols){
    if(vector == NULL)
        return;
    int i,j,c;
    
    for(c = 0; c < channels; c++){
        printf("channel: %d\n",c);
        for(i = 0; i < rows; i++){
            for(j = 0; j < cols; j++){
                printf("%f , ",vector[c*rows*cols+i*cols+j]);
            }
            printf("\n");
        }
    }
}

void print_4dtensor(float* vector,int as, int channels, int rows, int cols){
    if(vector == NULL)
        return;
    int i,j,c,a;
    
    for(a = 0; a < as; a++){
        printf("first dim: %d\n",a);
        for(c = 0; c < channels; c++){
            printf("channel: %d\n",c);
            for(i = 0; i < rows; i++){
                for(j = 0; j < cols; j++){
                    printf("%f , ",vector[c*rows*cols+i*cols+j]);
                }
                printf("\n");
            }
        }
    }    
}

void put_concatenation_first(aco_node** nodes, int n_nodes){
    int i,j;
    aco_node* temp;
    for(i = 0; i < n_nodes; i++){
        if(nodes[i]->concatenation != NULL){
            for(j = i-1; j >= 0; j--){
                temp = nodes[j];
                nodes[j] = nodes[i];
                nodes[j+1] = temp;
            }
        break;
        }
    }
}


void merge_aco_nodes(aco_node** arr, int l, int m, int r){ 
    int i, j, k; 
    int n1 = m - l + 1; 
    int n2 = r - m; 
  
    aco_node* L[n1], *R[n2]; 
  
    for (i = 0; i < n1; i++) 
        L[i] = arr[l + i]; 
    for (j = 0; j < n2; j++) 
        R[j] = arr[m + 1 + j]; 
  
    i = 0; 
  
    j = 0; 
  
    k = l; 
    while (i < n1 && j < n2) {
        if (L[i]->identifier <= R[j]->identifier) { 
            arr[k] = L[i]; 
            i++; 
        } 
        else { 
            arr[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 
  
    while (i < n1) { 
        arr[k] = L[i]; 
        i++; 
        k++; 
    } 
  
    while (j < n2) { 
        arr[k] = R[j]; 
        j++; 
        k++; 
    } 
} 
  

  
void merge_sort_aco_nodes(aco_node** arr, int l, int r){ 
    if (l < r) { 
        int m = l + (r - l) / 2; 
  
        merge_sort_aco_nodes(arr, l, m); 
        merge_sort_aco_nodes(arr, m + 1, r); 
  
        merge_aco_nodes(arr, l, m, r); 
    } 
} 

void free_matrix(void** matrix, int rows){
    if(matrix == NULL)
        return;
    int i;
    for(i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}

void copy_int_array(int* v, int* copy, int size){
    if(v == NULL || copy == NULL)
        return;
    memcpy(copy, v, size*sizeof(int));
}

void copy_uint8_t_array(uint8_t* v, uint8_t* copy, int size){
    if(v == NULL || copy == NULL)
        return;
    memcpy(copy, v, size*sizeof(uint8_t));
}

int int_min(int a, int b){
    if(a < b) return a; return b;
}
int int_max(int a, int b){
    if(a > b) return a; return b;
}

int is_little_endian(){
    unsigned int x = 1;
    return ((int) (((char *)&x)[0]) == 1);
}


void convert_data(void* ptr, uint64_t size, uint64_t len){
    if(!is_little_endian())
        swap_array_bytes_order(ptr,size,len);
}

void swap_array_bytes_order(void* ptr, uint64_t size, uint64_t len){
    if(size <= 1 || !len)
        return;
    char* array = (char*) &ptr;
    uint64_t i;
    for(i = 0; i < len; i++){
        reverse_ptr(array + i*size,size);
    }
}

void reverse_ptr(void* ptr, uint64_t size){
    if(size <= 1)
        return;
   char* array = (char*) ptr;
   uint64_t i, len = size/2;
   for(i = 0; i < len; i++){
       char temp = array[i];
       array[i] = array[size-1-i];
       array[size-1-i] = temp;
   }
   return;
}

void set_char_vector_with_zero(char* c, uint64_t size){
    if(c == NULL)    
        return;
    uint64_t i;
    for(i = 0; i < size; i++){
        c[i] = c[i] & 0x00;
    }
}

double sum_over_input(float* input, int size){
    if(input == NULL)
        return 0;
    int i;
    double sum = 0;
    for(i = 0; i < size; i++){
        sum+=input[i];
    }
    return sum;
}

void copy_char_array(char* input, char* output, int size){
    if(input == NULL || output == NULL || !size) return;
    memcpy(output,input,(sizeof(char)*size));
}

int read_file_in_char_vector(char** ksource, char* fname, int* size){
    int i;
    FILE *kfile;
    size_t kfilesize;
  
    kfile = fopen(fname, "r" );
    
    
    
    if(kfile == NULL){
        fprintf(stderr,"Error opening file %s\n",fname);
        return 1;
    }
    
    
    
    
    fseek( kfile, 0, SEEK_END );
    kfilesize = ((size_t)ftell(kfile));
    rewind( kfile );
    
    (*ksource) = (char*)malloc(kfilesize*sizeof(char));
    i = fread((*ksource), 1, kfilesize, kfile );
    fclose( kfile );
    (*size) = kfilesize;
    return 0;
}


void merge(float* arr, int* index, int left, int mid, int right) {
    int i, j, k; 
    int n1 = mid - left + 1; 
    int n2 = right - mid; 
  
    // Create temp arrays 
    int L[n1], R[n2]; 
  
    // Copy data to temp arrays 
    // L[] and R[] 
    for (i = 0; i < n1; i++) 
        L[i] = index[left + i]; 
    for (j = 0; j < n2; j++) 
        R[j] = index[mid + 1 + j]; 
  
    // Merge the temp arrays back 
    // into arr[l..r] 
    // Initial index of first subarray 
    i = 0; 
  
    // Initial index of second subarray 
    j = 0; 
  
    // Initial index of merged subarray 
    k = left; 
    while (i < n1 && j < n2) { 
        if (arr[L[i]] <= arr[R[j]]) { 
            index[k] = L[i]; 
            i++; 
        } 
        else { 
            index[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 
  
    // Copy the remaining elements 
    // of L[], if there are any 
    while (i < n1) { 
        index[k] = L[i]; 
        i++; 
        k++; 
    } 
  
    // Copy the remaining elements of 
    // R[], if there are any 
    while (j < n2) { 
        index[k] = R[j]; 
        j++; 
        k++; 
    } 
}

void merge_pure(float* arr, int left, int mid, int right) {
    int i, j, k; 
    int n1 = mid - left + 1; 
    int n2 = right - mid; 
  
    // Create temp arrays 
    float L[n1], R[n2]; 
  
    // Copy data to temp arrays 
    // L[] and R[] 
    for (i = 0; i < n1; i++) 
        L[i] = arr[left + i]; 
    for (j = 0; j < n2; j++) 
        R[j] = arr[mid + 1 + j]; 
  
    // Merge the temp arrays back 
    // into arr[l..r] 
    // Initial index of first subarray 
    i = 0; 
  
    // Initial index of second subarray 
    j = 0; 
  
    // Initial index of merged subarray 
    k = left; 
    while (i < n1 && j < n2) { 
        if (L[i] <= R[j]) { 
            arr[k] = L[i]; 
            i++; 
        } 
        else { 
            arr[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 
  
    // Copy the remaining elements 
    // of L[], if there are any 
    while (i < n1) { 
        arr[k] = L[i]; 
        i++; 
        k++; 
    } 
  
    // Copy the remaining elements of 
    // R[], if there are any 
    while (j < n2) { 
        arr[k] = R[j]; 
        j++; 
        k++; 
    } 
}

void merge_sort(float* arr, int* index, int left, int right) {
    if (left < right) { 
        // Same as (l+r)/2, but avoids 
        // overflow for large l and r 
        int m = left + (right - left) / 2; 
  
        // Sort first and second halves 
        merge_sort(arr,index, left, m); 
        merge_sort(arr,index, m + 1, right); 
  
        merge(arr, index, left, m, right); 
    } 
}

void merge_sort_pure(float* arr, int left, int right) {
    if (left < right) { 
        // Same as (l+r)/2, but avoids 
        // overflow for large l and r 
        int m = left + (right - left) / 2; 
  
        // Sort first and second halves 
        merge_sort_pure(arr, left, m); 
        merge_sort_pure(arr, m + 1, right); 
  
        merge_pure(arr, left, m, right); 
    } 
}

int shuffle_int_array(int* m,int n){
    if (n > 1) {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = m[j];
          m[j] = m[i];
          m[i] = t;
        }
    
    }
    return 0;
}


void set_frequencies_superstruct(float* output, float* frequencies, int size){
    if(output == NULL || frequencies == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        if(output[i] > 0)
            frequencies[i]+=1.0;
    }
}

void get_randomness(){
    srand(time(NULL));
}

void get_randomness_seed(unsigned int seed){
    srand(seed);
}

int there_are_missing_nodes(aco_node** nodes, int n_nodes){
    if(nodes == NULL || !n_nodes)
        return 1;
    int i;
    for(i = 0; i < n_nodes; i++){
        if(nodes[i]->identifier != i)
            return 1;
    }
    return 0;
}

int after_splitter_there_is_splitter_or_concatenation(aco_node* n){
    if(n == NULL || n->splitter == NULL)
        return 0;
    int i;
    for(i = 0; i < n->n_output; i++){
        if(n->output[i]->splitter != NULL || n->output[i]->concatenation != NULL)
            return 1;
    }
    return 0;
}

int set_value_of_just_lowest_in_vector(float* vector, float value, int size){
    if(size == 1){
        if(vector[0] > value){
            vector[0] = value;
            return 0;
        }
        return -1;
    }
    int index = size/2;
    
    if(vector[index] <= value){
        int n = set_value_of_just_lowest_in_vector(vector+index,value,size-index);
        if(n != -1){
            n+=index;
        }
        return n;
    }
    else if(vector[index] > value){
        return set_value_of_just_lowest_in_vector(vector,value,index);
    }
}


