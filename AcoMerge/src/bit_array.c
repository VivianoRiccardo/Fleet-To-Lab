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


BitArray* initBitArray() {
    BitArray* bitArray = (BitArray*)malloc(sizeof(BitArray));
    bitArray->data = NULL;
    bitArray->size = 0;
    bitArray->currentBit = 0;
    bitArray->pheromone = 0;
    return bitArray;
}

BitArray* copyBitArray(BitArray* a) {
    BitArray* bitArray = (BitArray*)malloc(sizeof(BitArray));
    bitArray->data = NULL;
    bitArray->size = 0;
    bitArray->currentBit = 0;
    bitArray->pheromone = 0;
    
    bitArray->size = a->size;
    bitArray->data = (uint8_t*)calloc(a->size,sizeof(uint8_t));
    copy_uint8_t_array(a->data,bitArray->data,a->size);
    bitArray->currentBit = a->currentBit;
    bitArray->pheromone = a->pheromone;
    
    return bitArray;
}

void free_BitArray(BitArray* a){
	if(a == NULL)
		return;
    free(a->data);
    free(a);
}

void reset_BitArray(BitArray* a){
    set_uint8_t_vector_with_value(0,a->data,a->size);
    a->currentBit = 0;
    a->pheromone = -1;
}

void set_BitArray_size(BitArray* a, uint64_t size){
    a->size = size;
    free(a->data);
    a->data = (uint8_t*)calloc(a->size,sizeof(uint8_t));
}

void setBits(BitArray *a, uint8_t value) {
    a->data[a->currentBit] = value;
    a->currentBit++;

}


uint8_t convertToNumbers(BitArray *a, int number_index) {
    return a->data[number_index];
}

