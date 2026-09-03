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


uint64_t get_aco_lstm_weights_or_pheromone_weights_size(int input_size, int output_size){
    return 3*output_size*(input_size + output_size) + 1;
}

uint64_t get_aco_lstm_biases_or_pheromone_biases_size(int output_size){
    return 3*output_size;
}

int aco_lstm_input_size(int input_size, int output_size){
    return input_size + 3*output_size;
}

int aco_lstm_output_size(int input_size, int output_size){
    return 6*output_size;
}

uint64_t aco_lstm_get_x_input_offset(){
    return 0;
}

uint64_t aco_lstm_get_c_input_offset(int input_size, int output_size){
    return input_size;
}

uint64_t aco_lstm_get_k_input_offset(int input_size, int output_size){
    return input_size+output_size;
}

uint64_t aco_lstm_get_h_input_offset(int input_size, int output_size){
    return input_size+2*output_size;
}

int aco_lstm_get_output_size(int output_size){
    return 6*output_size;
}

void init_aco_lstm_weights_xavier_init(int input_size, int output_size, float* weights){
    int i, size1 = 3*input_size*output_size, size2 = 3*output_size*output_size;
    float* w = weights + size1;
    for(i = 0; i < size1; i++){
        weights[i] = random_general_gaussian_xavier_init((float)(input_size));
    }
    for(i = 0; i < size2; i++){
        w[i] = random_general_gaussian_xavier_init((float)(output_size));
    }
}

void init_aco_lstm_weights_kaiming_constant_init(int input_size, int output_size, float* weights){
    int i, size1 = 3*input_size*output_size, size2 = 3*output_size*output_size;
    float* w = weights + size1;
    for(i = 0; i < size1; i++){
        weights[i] = signed_kaiming_constant((float)(input_size));
    }
    for(i = 0; i < size2; i++){
        w[i] = signed_kaiming_constant((float)(output_size));
    }
}

void init_aco_lstm_weights_signed_kaiming_constant_init_parameters(int input_size, int output_size, float* weights, int number_of_parameters){
    int i, size1 = 3*input_size*output_size*number_of_parameters, size2 = 3*output_size*output_size*number_of_parameters;
    float* w = weights + size1;
    for(i = 0; i < size1; i++){
        weights[i] = signed_kaiming_constant((float)(input_size));
    }
    for(i = 0; i < size2; i++){
        w[i] = random_general_gaussian_kaiming_init((float)(output_size));
    }
}

void init_aco_lstm_weights_chaos_init_parameters(int input_size, int output_size, float* weights, int number_of_parameters){
    int i, size1 = 3*input_size*output_size, size2 = 3*output_size*output_size,j;
    float* w = weights + size1*number_of_parameters;
    for(i = 0; i < size1; i++){
        float a = normalized_random();
        for(j = 0; j < number_of_parameters; j++){
            weights[i*number_of_parameters+j] = a;
            a = chaos_next_sequence(a);
        }
    }
    for(i = 0; i < size2; i++){
        float a = normalized_random();
        for(j = 0; j < number_of_parameters; j++){
            w[i*number_of_parameters+j] = a;
            a = chaos_next_sequence(a);
        }
    }
}

void init_aco_lstm_p(int input_size, int output_size, float* weights){
    uint64_t sum = get_aco_lstm_weights_or_pheromone_weights_size(input_size, output_size) -1;
    weights[sum] = normalized_random();
}

void init_aco_lstm_weights_with_value(int input_size, int output_size, float* weights, float pheromone_value){
    set_vector_with_value(pheromone_value,weights,3*output_size*(input_size+output_size));
}


void init_aco_lstm_biases_zeros(int output_size, float* biases){
    set_vector_with_value(0,biases,output_size*3);
}

void init_aco_lstm_biases_normalized_random(int output_size, float* biases){
    int i, size = 3*output_size;
    for(i = 0; i < size; i++){
        biases[i] = signed_normalized_random(1.0);
    }
}

void init_aco_lstm_biases_signed_kaiming_constant(int output_size, float* biases){
    int i, size = 3*output_size;
    for(i = 0; i < size; i++){
        biases[i] = signed_kaiming_constant((float)output_size);
    }
}


void init_aco_lstm_biases_with_value(int output_size, float* biases, float pheromone_value){
    set_vector_with_value(pheromone_value,biases,3*output_size);
}

void init_aco_lstm_weights_pheromones(int input_size, int output_size, float* weights_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,weights_pheromone,output_size*3*(input_size+output_size)+1);
}

void init_aco_lstm_biases_pheromones(int output_size, float* biases_pheromone, float pheromone_value){
    set_vector_with_value(pheromone_value,biases_pheromone,3*output_size);
}

aco_lstm* init_aco_lstm(int timestamp, int input_size, int output_size, int window, int identifier, uint64_t weights_array_offset, uint64_t  biases_array_offset,
                      uint64_t  pheromone_weights_array_offset, uint64_t  pheromone_biases_array_offset, uint64_t  output_array_offset,
                      uint64_t  input_array_offset,uint64_t  input_h_array_offset,uint64_t  input_k_array_offset,uint64_t  input_c_array_offset,uint64_t  output_h_array_offset,uint64_t  output_k_array_offset,uint64_t  output_c_array_offset,
                      float* weights, float* biases, float* pheromone_weights, float* pheromone_biases, float* output, float* input, float* input_h, float* input_k, float* input_c, float* output_h, float* output_k, float* output_c){
    
    
    aco_lstm* f = (aco_lstm*)malloc(sizeof(aco_lstm));
    f->k_percentage = 1;
    f->input_size = input_size;
    f->output_size = output_size;
    f->window = window;
    f->identifier = identifier;
    f->weights = weights;
    f->biases = biases;
    f->pheromone_weights = pheromone_weights;
    f->pheromone_biases = pheromone_biases;
    f->output = output;
    f->input = input;
    f->weights_array_offset = weights_array_offset;
    f->biases_array_offset = biases_array_offset;
    f->pheromone_weights_array_offset = pheromone_weights_array_offset;
    f->pheromone_biases_array_offset = pheromone_biases_array_offset;
    f->output_array_offset = output_array_offset;
    f->output_h_array_offset = output_h_array_offset;
    f->output_c_array_offset = output_c_array_offset;
    f->input_array_offset = input_array_offset;
    f->input_h_array_offset = input_h_array_offset;
    f->input_c_array_offset = input_c_array_offset;
    f->input_h = input_h;
    f->input_c = input_c;
    f->output_h = output_h;
    f->output_c = output_c;
    f->input_k = input_k;
    f->output_k = output_k;
    f->input_k_array_offset = input_k_array_offset;
    f->output_k_array_offset = output_k_array_offset;
    f->timestamp = timestamp;
    f->input_error = NULL;
    f->input_h_error = NULL;
    f->output_h_error = NULL;
    f->input_k_error = NULL;
    f->output_k_error = NULL;
    f->input_c_error = NULL;
    f->output_c_error = NULL;
    f->d_weights = NULL;
    f->d_biases = NULL;
    f->all_weights = NULL;
    f->all_biases = NULL;
    f->indices = NULL;
    f->output_used = NULL;
    
    return f;
}

aco_lstm* copy_aco_lstm(aco_lstm* a){
    if(a == NULL)    
        return NULL;
    aco_lstm* n = (aco_lstm*)malloc(sizeof(aco_lstm));
    n->k_percentage = a->k_percentage;
    n->indices = a->indices;
    n->input_size = a->input_size;
    n->output_size = a->output_size;
    n->window = a->window;
    n->identifier = a->identifier;
    n->weights = a->weights;
    n->biases = a->biases;
    n->pheromone_weights = a->pheromone_weights;
    n->pheromone_biases = a->pheromone_biases;
    n->output = a->output;
    n->input = a->input;
    n->weights_array_offset = a->weights_array_offset;
    n->biases_array_offset = a->biases_array_offset;
    n->pheromone_weights_array_offset = a->pheromone_weights_array_offset;
    n->pheromone_biases_array_offset = a->pheromone_biases_array_offset;
    n->output_array_offset = a->output_array_offset;
    n->output_h_array_offset = a->output_h_array_offset;
    n->output_c_array_offset = a->output_c_array_offset;
    n->input_array_offset = a->input_array_offset;
    n->input_h_array_offset = a->input_h_array_offset;
    n->input_c_array_offset = a->input_c_array_offset;
    n->input_h = a->input_h;
    n->input_c = a->input_c;
    n->output_h = a->output_h;
    n->output_c = a->output_c;
    n->input_k = a->input_k;
    n->output_k = a->output_k;
    n->input_k_array_offset = a->input_k_array_offset;
    n->output_k_array_offset = a->output_k_array_offset;
    n->timestamp = a->timestamp;
    n->output_used = a->output_used;
    return n;
}

void assign_aco_lstm_storing_params(aco_lstm* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output = storing_params+a->output_array_offset;
    a->input = storing_params+a->input_array_offset;
    a->input_c = storing_params+a->input_c_array_offset;
    a->input_k = storing_params+a->input_k_array_offset;
    a->input_h = storing_params+a->input_h_array_offset;
    a->output_c = storing_params+a->output_c_array_offset;
    a->output_k = storing_params+a->output_k_array_offset;
    a->output_h = storing_params+a->output_h_array_offset;
}

void assign_aco_lstm_storing_params_output_used(aco_lstm* a, int* storing_params){
    if(a == NULL)    
        return;
    a->output_used = storing_params+a->output_array_offset;
    a->output_h_used = storing_params+a->output_h_array_offset;
}

void assign_aco_lstm_storing_partial_derivatives(aco_lstm* a, float* storing_params){
    if(a == NULL)    
        return;
    a->output_error = storing_params+a->output_array_offset;
    a->input_error = storing_params+a->input_array_offset;
    a->input_c_error = storing_params+a->input_c_array_offset;
    a->input_k_error = storing_params+a->input_k_array_offset;
    a->input_h_error = storing_params+a->input_h_array_offset;
    a->output_c_error = storing_params+a->output_c_array_offset;
    a->output_k_error = storing_params+a->output_k_array_offset;
    a->output_h_error = storing_params+a->output_h_array_offset;
}

void assign_aco_lstm_learnable_params(aco_lstm* a, float* params){
    if(a == NULL)
        return;
    a->weights = params+a->weights_array_offset;
    a->biases = params+a->biases_array_offset;
}

void assign_aco_lstm_learnable_params_scores(aco_lstm* a, float* params, int* indices, float* scores){
    if(a == NULL)
        return;
    a->d_weights = params+a->weights_array_offset;
    a->scores = scores+a->weights_array_offset;
    a->indices = indices+a->weights_array_offset;
}


// TO TEST
void lstm_feed_forward(float timestamp, float* x, float* h, float* k, float* c, float* cell_state, float* k_state, float* hidden_state, float* z, float* w, float* u, float* b, float p, int input_size, int output_size){
    
    int i,j; 
    int size1 = output_size, size2 = 2*output_size;
    int size3 = input_size, size4 = 2*input_size;
    int size5 = input_size*output_size, size6 = size5*2;
    int size7 = output_size*output_size, size8 = size7*2;
    int temp, temp2, temp3, temp4;
    
    for(i = 0; i < output_size; i++){
        temp = i*input_size;
        temp2 = i*output_size;
        for(j = 0; j < input_size; j++){
            temp3 = temp+j;
            z[i] += w[temp3]*x[j]; //r_t
            z[size1+i] += w[size5+temp3]*x[j]; //z_o
            z[size2+i] += w[size6+temp3]*x[j]; //z_c
        }
        for(j = 0; j < output_size; j++){
            temp4 = temp2+j;
            z[i] += u[temp4]*h[j]; //r_t
            z[size1+i] += u[size7+temp4]*h[j]; //z_o
            z[size2+i] +=u[size8+temp4]*h[j]; //z_c
        }
        
        //z[i] += b[i];//r_t
        //z[size1+i] += b[size1+i];//z_o
        //z[size2+i] += b[size2+i];//z_c
        
        
        z[i] = sigmoid(z[i]); //r_t
        z[size1+i] = sigmoid(z[size1+i]); //o_t
        z[size2+i] = tanhh(z[size2+i]); //tanhh(z_c)
        
        k_state[i] = z[i]*timestamp + (1-z[i])*k[i];
        z[i] = pow((timestamp-k_state[i]+1.0)/(timestamp-k_state[i]+0.001),-p);// f_t
        cell_state[i] = z[size2+i]*(1-z[i]) + c[i]*z[i]; //cell state of output we calculate c is the previous c state
        hidden_state[i] = z[size1+i]*tanhh(cell_state[i]); //hidden state of output we calculate
    }
}


void lstm_feed_forward_edge_popup(float timestamp, float* x, float* h, float* k, float* c, float* cell_state, float* k_state, float* hidden_state, float* z, float* w, float* u, float* b, float p, int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used){
    
    int i,j;
    int size1 = output_size, size2 = 2*output_size;
    int size3 = input_size, size4 = 2*input_size;
    int size5 = input_size*output_size, size6 = size5*2;
    int size7 = output_size*output_size, size8 = size7*2;
    int temp, temp2, temp3, temp4;
    int last_n1 = input_size*output_size*k_percentage;
    int last_n2 = output_size*output_size*k_percentage;
    int p1 = input_size*output_size-last_n1;
    int p2 = output_size*output_size-last_n2;
    int p3 = output_size*output_size;
    int p4 = output_size*output_size;
    /*
    float* z1_output_used = (float*)calloc(output_size, sizeof(float));
    float* z2_output_used = (float*)calloc(output_size, sizeof(float));
    float* z3_output_used = (float*)calloc(output_size, sizeof(float));
    * */
    int temp_output1;
    int temp_output2;
    int temp_output3;
    int temp_input1;
    int temp_input2;
    int temp_input3;
    
    
    int* z1_output_used = output_used;
    int* z2_output_used = output_used+output_size;
    int* z3_output_used = output_used+2*output_size;
    
    
    for(i = p1; i < p3; i++){
		
		temp_output1 = (int)(indices_w[i]/input_size);
        temp_input1 = (indices_w[i]%input_size);
        temp_output2 = (int)(indices_w[size5+i]/input_size);
        temp_input2 = (indices_w[size5+i]%input_size);
        temp_output3 = (int)(indices_w[size6+i]/input_size);
        temp_input3 = (indices_w[size6+i]%input_size);
        /*
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
        * */
        z[temp_output1] += w[indices_w[i]]*x[temp_input1]; //r_t
        z[size1+temp_output2] += w[size5+indices_w[size5+i]]*x[temp_input2]; //z_o
        z[size2+temp_output3] += w[size6+indices_w[size6+i]]*x[temp_input3]; //z_c
    }
    
    for(i = p2; i < p4; i++){
        temp_output1 = (int)(indices_u[i]/output_size);
        temp_input1 = (indices_u[i]%output_size);
        temp_output2 = (int)(indices_u[size7+i]/output_size);
        temp_input2 = (indices_u[size7+i]%output_size);
        temp_output3 = (int)(indices_u[size8+i]/output_size);
        temp_input3 = (indices_u[size8+i]%output_size);
        /*
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
        */
        z[temp_output1] += u[indices_u[i]]*h[temp_input1]; //r_t
        z[size1+temp_output2] += u[size7+indices_u[size7+i]]*h[temp_input2]; //z_o
        z[size2+temp_output3] +=u[size8+indices_u[size8+i]]*h[temp_input3]; //z_c
        
        
    }
    float temp_z1;
    float temp_z2;
    float temp_z3;
    for(i = 0; i < output_size; i++){
        temp_z1 = z[i];
        temp_z2 = z[size1+i];
        temp_z3 = z[size2+i];
        if(z1_output_used[i])
            z[i] = sigmoid(z[i]); //r_t
        
        if(z2_output_used[i])
            z[size1+i] = sigmoid(z[size1+i]); //o_t
        
        if(z3_output_used[i])
            z[size2+i] = tanhh(z[size2+i]); //tanhh(z_c)
        
        k_state[i] = z[i]*timestamp + (1-z[i])*k[i];
        z[i] = pow((timestamp-k_state[i]+1.0)/(timestamp-k_state[i]+0.001),-p);// f_t
        cell_state[i] = z[size2+i]*(1-z[i]) + c[i]*z[i]; /*cell state of output we calculate c is the previous c state*/
        hidden_state[i] = z[size1+i]*tanhh(cell_state[i]); /*hidden state of output we calculate*/ 
        


        
        z[i] = temp_z1;
        z[size1+i] = temp_z2;
        z[size2+i] = temp_z3;
    }
    
    /*
    free(z1_output_used);
    free(z2_output_used);
    free(z3_output_used);
    * */
}


void lstm_back_propagation_edge_popup(float timestamp, float* x,float* input_error, float* h,float* h_error, float* k,float* k_error, float* c,float* c_error, float* cell_state,float* c_output_error, float* k_state,float* k_output_error, float* hidden_state,float* h_output_error, float* z,float* output_error, float* w, float* d_w, float* u, float* d_u, float* b, float p, int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used){
    
    int i,j,zz; 
    int size1 = output_size, size2 = 2*output_size;
    int size3 = input_size, size4 = 2*input_size;
    int size5 = input_size*output_size, size6 = size5*2;
    int size7 = output_size*output_size, size8 = size7*2;
    int temp, temp2, temp3, temp4,temp5,temp6,temp7;
    int last_n1 = input_size*output_size*k_percentage;
    int last_n2 = output_size*output_size*k_percentage;
    int p1 = input_size*output_size-last_n1;
    int p2 = output_size*output_size-last_n2;
    int p3 = output_size*output_size;
    int p4 = output_size*output_size;
    /*
    float* z1_output_used = (float*)calloc(output_size, sizeof(float));
    float* z2_output_used = (float*)calloc(output_size, sizeof(float));
    float* z3_output_used = (float*)calloc(output_size, sizeof(float));
    * */
    int* z1_output_used = output_used;
    int* z2_output_used = output_used+output_size;
    int* z3_output_used = output_used+2*output_size;
    int temp_output1;
    int temp_output2;
    int temp_output3;
    int temp_input1;
    int temp_input2;
    int temp_input3;
    
    /*
    for(i = 0; i < output_size; i++){
        z[i] = 0;
        z[size1+i] = 0;
        z[size2+i] = 0;
    }
    
    for(i = p1; i < p3; i++){
        temp_output1 = (int)(indices_w[i]/input_size);
        temp_input1 = (indices_w[i]%input_size);
        temp_output2 = (int)(indices_w[size5+i]/input_size);
        temp_input2 = (indices_w[size5+i]%input_size);
        temp_output3 = (int)(indices_w[size6+i]/input_size);
        temp_input3 = (indices_w[size6+i]%input_size);
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
    }
    for(i = p2; i < p4; i++){
        temp_output1 = (int)(indices_u[i]/output_size);
        temp_input1 = (indices_u[i]%output_size);
        temp_output2 = (int)(indices_u[size7+i]/output_size);
        temp_input2 = (indices_u[size7+i]%output_size);
        temp_output3 = (int)(indices_u[size8+i]/output_size);
        temp_input3 = (indices_u[size8+i]%output_size);
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
    }
    * */
    float temp_val;
    float temp2_val;
    float temp3_val;
    for(i = 0; i < output_size; i++){
        
        if(z1_output_used[i])
            temp_val = sigmoid(z[i]);
        else
            temp_val = 0;
        temp2_val = pow((timestamp-k_state[i]+1.0)/(timestamp-k_state[i]+0.001),-p);
        
        if(z2_output_used[i]){
            output_error[size1+i] += h_output_error[i]*tanhh(cell_state[i]);
            output_error[size1+i]*=derivative_sigmoid(z[size1+i]);
            c_output_error[i]+=h_output_error[i]*sigmoid(z[size1+i]);
            c_output_error[i]*=derivative_tanhh(cell_state[i]);// c_output_error is from next cell state
        }
        if(z3_output_used[i]){
            output_error[size2+i] += c_output_error[i]*(1-temp2_val);
            output_error[size2+i] *= derivative_tanhh(z[size2+i]);
        }
        
        temp3_val = c_output_error[i]*(c[i]-tanhh(z[size2+i]))*(temp2_val*-0.999*p)/((k_state[i]-timestamp-1.0)*(k_state[i]-timestamp-0.001));
        c_error[i]+=c_output_error[i]*temp2_val;
        k_output_error[i]+=temp3_val;
        k_error[i] += temp3_val*(1-temp_val);
        if(z1_output_used[i]){
            temp3_val*=(timestamp-k[i])*derivative_sigmoid(z[i]);
            output_error[i]+=temp3_val;
        }
        
        temp = i*input_size;
        temp2 = i*output_size;
        for(j = 0; j < input_size; j++){
            temp3 = temp+j;
            /*input_error[j]+=output_error[i]*w[temp3];
            input_error[j]+=output_error[size1+i]*w[size5+temp3];
            input_error[j]+=output_error[size2+i]*w[size6+temp3];
            * */
            d_w[temp3]+=x[j]*output_error[i]*w[temp3];
            d_w[size5+temp3]+=x[j]*output_error[size1+i]*w[size5+temp3];
            d_w[size6+temp3]+=x[j]*output_error[size2+i]*w[size6+temp3];
        }
        for(j = 0; j < output_size; j++){
            temp4 = temp2+j;
            /*h_error[j]+=output_error[i]*u[temp4];
            h_error[j]+=output_error[size1+i]*u[size7+temp4];
            h_error[j]+=output_error[size2+i]*u[size8+temp4];
            * */
            d_u[temp4]+=h[j]*output_error[i]*u[temp4];
            d_u[size5+temp4]+=h[j]*output_error[size1+i]*u[size7+temp4];
            d_u[size6+temp4]+=h[j]*output_error[size2+i]*u[size8+temp4];
        }
        
    }
    
    for(i = p1; i < p3; i++){
		
        temp_output1 = (int)(indices_w[i]/input_size);
        temp_input1 = (indices_w[i]%input_size);
        temp_output2 = (int)(indices_w[size5+i]/input_size);
        temp_input2 = (indices_w[size5+i]%input_size);
        temp_output3 = (int)(indices_w[size6+i]/input_size);
        temp_input3 = (indices_w[size6+i]%input_size);
        /*
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
        * */
        input_error[temp_input1]+=output_error[temp_output1]*w[indices_w[i]]; //r_t
        input_error[temp_input2]+=output_error[size1+temp_output2]*w[size5+indices_w[size5+i]]; //z_o
        input_error[temp_input3]+=output_error[size2+temp_output3]*w[size6+indices_w[size6+i]]; //z_c
        
    }
    
    for(i = p2; i < p4; i++){
        temp_output1 = (int)(indices_u[i]/output_size);
        temp_input1 = (indices_u[i]%output_size);
        temp_output2 = (int)(indices_u[size7+i]/output_size);
        temp_input2 = (indices_u[size7+i]%output_size);
        temp_output3 = (int)(indices_u[size8+i]/output_size);
        temp_input3 = (indices_u[size8+i]%output_size);
        /*
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
        */
        h_error[temp_input1]+=output_error[temp_output1]*u[indices_u[i]]; //r_t
        h_error[temp_input2]+=output_error[size1+temp_output2]*u[size7+indices_u[size7+i]]; //z_o
        h_error[temp_input3]+=output_error[size2+temp_output3]*u[size8+indices_u[size8+i]]; //z_c
        
        
    }
    
    
    /*
    free(z1_output_used);
    free(z2_output_used);
    free(z3_output_used);
    * */
}


void aco_lstm_feed_forward(aco_lstm* lstm){
    if(lstm == NULL)
        return;
    lstm_feed_forward((float)lstm->timestamp, lstm->input, lstm->input_h, lstm->input_k, lstm->input_c, lstm->output_c, lstm->output_k, lstm->output_h, lstm->output, lstm->weights, lstm->weights+(lstm->input_size*lstm->output_size*3), lstm->biases,lstm->weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3], lstm->input_size, lstm->output_size);

}

void aco_lstm_feed_forward_edge_popup(aco_lstm* lstm){
    if(lstm == NULL)
        return;
    lstm_feed_forward_edge_popup((float)lstm->timestamp, lstm->input, lstm->input_h, lstm->input_k, lstm->input_c, lstm->output_c, lstm->output_k, lstm->output_h, lstm->output, lstm->weights, lstm->weights+(lstm->input_size*lstm->output_size*3), lstm->biases,lstm->weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3], lstm->input_size, lstm->output_size,lstm->indices,lstm->indices+(lstm->input_size*lstm->output_size*3),lstm->k_percentage, lstm->output_used);

}

void aco_lstm_back_propagation_edge_popup(aco_lstm* lstm){
    if(lstm == NULL)
        return;
    lstm_back_propagation_edge_popup((float)lstm->timestamp, lstm->input, lstm->input_error, lstm->input_h, lstm->input_h_error, lstm->input_k,lstm->input_k_error, lstm->input_c, lstm->input_c_error,lstm->output_c,lstm->output_c_error, lstm->output_k, lstm->output_k_error, lstm->output_h,lstm->output_h_error, lstm->output, lstm->output_error, lstm->weights, lstm->d_weights, lstm->weights+(lstm->input_size*lstm->output_size*3),lstm->d_weights+(lstm->input_size*lstm->output_size*3), lstm->biases,lstm->weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3], lstm->input_size, lstm->output_size, lstm->indices,lstm->indices+(lstm->input_size*lstm->output_size*3),lstm->k_percentage, lstm->output_used);

}

/*
void aco_lstm_back_propagation(aco_lstm* lstm, int n_parameters){
    if(lstm == NULL)
        return;
    lstm_back_propagation((float)lstm->timestamp, lstm->input, lstm->input_error,lstm->input_h,lstm->input_h_error, lstm->input_k,lstm->input_k_error, lstm->input_c,lstm->input_c_error, lstm->output_c,lstm->output_c_error, lstm->output_k,lstm->output_k_error, lstm->output_h,lstm->output_h_error, lstm->output,lstm->output_error, lstm->weights,lstm->all_weights,lstm->d_weights, lstm->weights+(lstm->input_size*lstm->output_size*3),lstm->all_weights+(lstm->input_size*lstm->output_size*3),lstm->d_weights+(lstm->input_size*lstm->output_size*3), lstm->biases,lstm->weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3],&lstm->all_weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3],&lstm->d_weights[(lstm->input_size+lstm->output_size)*lstm->output_size*3], lstm->input_size, lstm->output_size,n_parameters);

}
*/

void print_aco_lstm_offsets(aco_lstm* f){
    printf("input array offset: %lud\n",f->input_array_offset);
    printf("input_c array offset: %lud\n",f->input_c_array_offset);
    printf("input_k array offset: %lud\n",f->input_k_array_offset);
    printf("input_h array offset: %lud\n",f->input_h_array_offset);
    printf("output_c array offset: %lud\n",f->output_c_array_offset);
    printf("output_k array offset: %lud\n",f->output_k_array_offset);
    printf("output_h array offset: %lud\n",f->output_h_array_offset);
    printf("output array offset: %lud\n",f->output_array_offset);
}

void merge_sort_aco_lstm(aco_lstm* c){
    if(c == NULL)
        return;
    merge_sort(c->scores,c->indices,0,c->input_size*c->output_size-1);
    merge_sort(c->scores+c->input_size*c->output_size,c->indices+c->input_size*c->output_size,0,c->input_size*c->output_size-1);
    merge_sort(c->scores+c->input_size*c->output_size*2,c->indices+c->input_size*c->output_size*2,0,c->input_size*c->output_size-1);
    merge_sort(c->scores+c->input_size*c->output_size*3,c->indices+c->input_size*c->output_size*3,0,c->output_size*c->output_size-1);
    merge_sort(c->scores+c->input_size*c->output_size*3+c->output_size*c->output_size,c->indices+c->input_size*c->output_size*3+c->output_size*c->output_size,0,c->output_size*c->output_size-1);
    merge_sort(c->scores+c->input_size*c->output_size*3+c->output_size*c->output_size*2,c->indices+c->input_size*c->output_size*3+c->output_size*c->output_size*2,0,c->output_size*c->output_size-1);
}

float* get_aco_lstm_d_output(aco_lstm* lstm){
    if(lstm == NULL)
        return NULL;
    return lstm->output_h_error;
}

void init_lstm_scores(aco_lstm* lstm){
    if(lstm == NULL)
        return;
    int i;
    for(i = 0; i < lstm->output_size*lstm->input_size; i++){
        lstm->indices[i] = i;
        lstm->indices[lstm->output_size*lstm->input_size+i] = i;
        lstm->indices[2*lstm->output_size*lstm->input_size+i] = i;
    }
    for(i = 0; i < lstm->output_size*lstm->output_size; i++){
        lstm->indices[3*lstm->output_size*lstm->input_size+i] = i;
        lstm->indices[3*lstm->output_size*lstm->input_size+lstm->output_size*lstm->output_size+i] = i;
        lstm->indices[3*lstm->output_size*lstm->input_size+2*lstm->output_size*lstm->output_size+i] = i;
    }
}

void set_lstm_k_percentage(aco_lstm* c, float k){
	if(c == NULL)
		return;
	if(k < 0 || k > 1)
		return;
	c->k_percentage = k;
}

void lstm_used_output(float timestamp,int input_size, int output_size, int* indices_w, int* indices_u, float k_percentage, int* output_used, int* output_h_used){
    
    int i,j;
    int size1 = output_size, size2 = 2*output_size;
    int size3 = input_size, size4 = 2*input_size;
    int size5 = input_size*output_size, size6 = size5*2;
    int size7 = output_size*output_size, size8 = size7*2;
    int temp, temp2, temp3, temp4;
    int last_n1 = input_size*output_size*k_percentage;
    int last_n2 = output_size*output_size*k_percentage;
    int p1 = input_size*output_size-last_n1;
    int p2 = output_size*output_size-last_n2;
    int p3 = output_size*output_size;
    int p4 = output_size*output_size;
    
    int temp_output1;
    int temp_output2;
    int temp_output3;
    int temp_input1;
    int temp_input2;
    int temp_input3;
    
    
    int* z1_output_used = output_used;
    int* z2_output_used = output_used+output_size;
    int* z3_output_used = output_used+2*output_size;
    
    
    for(i = p1; i < p3; i++){
        temp_output1 = (int)(indices_w[i]/input_size);
        temp_input1 = (indices_w[i]%input_size);
        temp_output2 = (int)(indices_w[size5+i]/input_size);
        temp_input2 = (indices_w[size5+i]%input_size);
        temp_output3 = (int)(indices_w[size6+i]/input_size);
        temp_input3 = (indices_w[size6+i]%input_size);
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
    }
    for(i = p2; i < p4; i++){
        temp_output1 = (int)(indices_u[i]/output_size);
        temp_input1 = (indices_u[i]%output_size);
        temp_output2 = (int)(indices_u[size7+i]/output_size);
        temp_input2 = (indices_u[size7+i]%output_size);
        temp_output3 = (int)(indices_u[size8+i]/output_size);
        temp_input3 = (indices_u[size8+i]%output_size);
        z1_output_used[temp_output1] = 1;
        z2_output_used[temp_output2] = 1;
        z3_output_used[temp_output3] = 1;
    }
    
    for(i = 0; i < output_size; i++){
		output_h_used[i] = (z1_output_used[i]||z2_output_used[i]||z3_output_used[i])?1:0;
	}
}


void set_aco_lstm_used_output(aco_lstm* lstm){
	if(lstm == NULL)
		return;

	lstm_used_output((float)lstm->timestamp,lstm->input_size,lstm->output_size,lstm->indices,lstm->indices+(lstm->input_size*lstm->output_size*3),lstm->k_percentage, lstm->output_used, lstm->output_h_used);
}

int* get_aco_lstm_used_outputs(aco_lstm* a){
	if(a == NULL)
		return NULL;
	return a->output_h_used;
}


