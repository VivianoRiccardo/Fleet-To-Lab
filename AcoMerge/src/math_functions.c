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

float sigmoid(float x){
    return 1.0/(1.0+exp(-x));
}

void sigmoid_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = sigmoid(input[i]);
    }
}

void sigmoid_array_with_flags(float* input, float* output,int* flags, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = ((float)((int)(flags[i])))*sigmoid(input[i]);
    }
}

float tanhh(float x){
    double y = exp(2*x);
    return (y-1)/(y+1);
}

void tanh_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = tanhh(input[i]);
    }
}

float float_max(float a, float b){
    if(a > b)return a;return b;
}

float relu(float x){
    return float_max(0.0,x);
}

float derivative_relu(float x){
    if(x > 0)
        return 1.0;
    return 0.0;
}

void relu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = relu(input[i]);
    }
}

float elu(float x, float alpha){
    if(x > 0)
        return x;
    return alpha*(exp(x)-1);
}

float derivative_elu(float x, float alpha){
    if(x > 0)
        return 1.0;
    return alpha*(exp(x));
}

void elu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = elu(input[i],0.1);
    }
}

float leaky_relu(float x){
    return float_max(x*0.01,x);
}

float derivative_leaky_relu(float x){
    if(x > 0)
        return 1.0;
    return 0.01;
}

void leaky_relu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = leaky_relu(input[i]);
    }
}


float gelu(float x){
    return x*sigmoid(1.702*x);
}

float derivative_gelu(float x){
    float t = exp(x*1.702);
    float tt = t+1;
    tt*=tt;
    return t*(t+1.702*x+1)/tt;
}

void derivative_softmax(float* output,float* softmax_arr,float* error, int size){
    int i,j;
    
    for(j = 0; j < size; j++){
        for(i = 0; i < size; i++){
            if (i == j)
                output[j] += (softmax_arr[i]*(1-softmax_arr[j]))*error[i];
            else
                output[j] -= softmax_arr[j]*softmax_arr[i]*error[i];
            
        }
        
    }
}

void derivative_softmax_with_flags(float* output,float* softmax_arr,float* error, int* flags, int size){
    int i,j;
    
    for(j = 0; j < size; j++){
        if(!flags[j]) continue;
        for(i = 0; i < size; i++){
            if(!flags[i]) continue;
            if (i == j)
                output[j] += (softmax_arr[i]*(1-softmax_arr[j]))*error[i];
            else
                output[j] -= softmax_arr[j]*softmax_arr[i]*error[i];
            
        }
        
    }
}

float derivative_focal_loss(float y_hat, float y, float gamma){
    float temp,log_one,power;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);

    log_one = log(temp);
    power = pow((double)(1-temp),(double)gamma)/temp;

    
    float temp2 = gamma*pow((double)(1-temp),(double)gamma-1)*log_one-power;
    if(y == 1)
        return temp2;
    else
        return -temp2;
}


void gelu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = gelu(input[i]);
    }
}

void softmax(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    float sum = 0;
    for(i = 0; i < size; i++){
        output[i] = exp(input[i]);
        sum+=output[i];
    }
    for(i = 0; i < size; i++){
        output[i]/=sum;
    }
}

void softmax_with_flags(float* input, float* output,int* flags, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    float sum = 0;
    for(i = 0; i < size; i++){
        output[i] = ((float)flags[i])*exp(input[i]);
        sum+=output[i];
    }
    for(i = 0; i < size; i++){
        output[i]/=sum;
    }
}

float float_abs(float a){
    return (a > 0) ? a : -a;
}

float focal_loss(float y_hat, float y, float gamma){
    float temp,log_one;
    if(y == 1)
        temp = (y_hat);
    else
        temp = (1-y_hat);
    
    if(!temp || temp != temp){
        temp = 0;
        log_one = -999999;
    }
    else{
        log_one = log(temp);
    }
    return -pow((double)(1-temp),(double)gamma)*log_one;
        
}

void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size){
    int i;
    for(i = 0; i < size; i++){
        output[i] = focal_loss(y_hat[i],y[i],gamma);
    }
}
void compute_prob_average_double(float* input, float* output, int size){
    double sum = sum_over_input(input,size);
    int i;
    for(i = 0; i < size; i++){
        double out = input[i]/sum;
        output[i] = out;
    }
}

float derivative_sigmoid(float x){
    float y = sigmoid(x);
    return y*(1-y);
}

float derivative_tanhh(float x){
    float y = tanhh(x);
    return 1-y*y;
}


void derivative_sigmoid_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_sigmoid(input[i]);
    }
}

void derivative_sigmoid_array_with_flags(float* input, float* output,int* flags, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += ((float)flags[i])*derivative_sigmoid(input[i]);
    }
}

void derivative_tanh_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_tanhh(input[i]);
    }
}

void derivative_tanh_array_with_flags(float* input, float* output,int* flags, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += ((float)flags[i])*derivative_tanhh(input[i]);
    }
}


void derivative_relu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_relu(input[i]);
    }
}

void derivative_leaky_relu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_leaky_relu(input[i]);
    }
}

void derivative_elu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_elu(input[i],0.01);
    }
}

void derivative_elu_array_with_flags(float* input, float* output,int* flags, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += ((float)((int)(flags[i])))*derivative_elu(input[i],0.01);
    }
}

void derivative_gelu_array(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_gelu(input[i]);
    }
}

double cdf_normal(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

double pdf_normal(double x) {
    return exp(-x * x / 2) / sqrt(2 * PI);
}

double d_acquisition_function_d_mu(double mu, double sigma, double best_so_far) {
    if(sigma == 0.0)
        return 0.0;
    double z = (mu - best_so_far) / sigma;
    return cdf_normal(z) - z*pdf_normal(z) + (mu-best_so_far)*exp(-z*z)/(sigma*sqrtf(2*PI));
}

void update_hyperparameters(float *hyperparameters, double step_size, double mu, double sigma, double best_so_far) {
    double gradient = d_acquisition_function_d_mu(mu,sigma,best_so_far);
    *hyperparameters += step_size * gradient;
}

float b_spline(float x, float* knots, int degree, int index, int first_time){
    
    if(!degree)
        return 1;
    float first_part = 1;
    float second_part = 1;
    if(first_time){
        if(x > knots[index+1] || x < knots[index])
            first_part = 0;
        if(x > knots[index+2] || x < knots[index+1])
            second_part = 0;
    }
    float ret = 0;
    if(first_part){
        if(x-knots[index] != 0.0) {
            float t = knots[index+degree]-knots[index];
            if(t == 0.0)
                t = EPSILON;
            ret+=((x-knots[index])/t)*b_spline(x, knots, degree-1, index, 0);
        }
    }
    if(second_part){
        if((knots[index+degree+1]-x) != 0.0) {
            float t = (knots[index+degree+1]-knots[index+1]);
            if(t == 0.0)
                t = EPSILON;
            ret+=((knots[index+degree+1]-x)/t)*b_spline(x, knots, degree-1, index+1, 0);
        }
    }
    return ret;
    
}


float silu(float x){
    return x/(1+exp(-x));
}

void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size){
    if(y_hat == NULL || y == NULL || output == NULL || size < 1)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] += derivative_focal_loss(y_hat[i],y[i],gamma);
    }
}

void compute_general_derivative_loss(float* y_hat, float* y, float* output, float gamma, int size, int flag){
    if(y_hat == NULL || y == NULL || output == NULL || size < 0)
        return;
    if(flag == FOCAL_LOSS){
        derivative_focal_loss_array(y_hat, y, output, gamma,size);
    }
}

void computer_derivative_loss(float* y, float gamma, edge_popup_trainer* e, int index){
    if(e == NULL || y == NULL || index < 0 || index >= e->batch_size)
        return;
    int i;
    uint64_t sum = 0;
    for(i = 0; i < e->s[index]->n_leaves; i++){
        compute_general_derivative_loss(get_aco_node_output(e->s[index]->leaves[i]), y+sum, get_aco_node_output_error(e->s[index]->leaves[i]),gamma,get_aco_node_output_size(e->s[index]->leaves[i]), e->error_flags[i]);
        sum+=get_aco_node_output_size(e->s[index]->leaves[i]);
    }
}

void compute_inverse(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        if(input[i] != 0.0)
            output[i] = 1.0/input[i];
    }
}

void compute_inverse_with_indices(float* input, float* output, int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        if(indices[i] && input[i] != 0.0)
            output[i] = 1.0/input[i];
    }
}

void compute_inverse_derivative_with_indices(float* input, float* output, int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        if(indices[i] && input[i] != 0.0)
            output[i] = -1.0/(input[i]*input[i]);
    }
}

void dot1D_float_with_int(float* input1, int* input2, float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]*((float)(input2[i]));
    }
}

void dot1D(float* input1, float* input2, float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]*(input2[i]);
    }
}

void div1D(float* input1, float* input2, float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]/(input2[i]);
    }
}

void dif1D(float* input1, float* input2, float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input1[i]-(input2[i]);
    }
}

void exp1D(float* input1, float* input2, float* output, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = pow(input1[i],(input2[i]));
    }
}

void exp1D_with_flags(float* input1, float* input2, float* output, int* flags, int size){
    if(input1 == NULL || input2 == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        if(flags[i])
        output[i] = pow(input1[i],(input2[i]));
    }
}

void mul_value_float_with_int(int* input, float value, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = ((float)(input[i]))*value;
    }
}

void mul_value(float* input, float value, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = (input[i])*value;
    }
}

void compute_square(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input[i]*input[i];
    }
}

void compute_square_with_indices(float* input, float* output,int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = input[i]*input[i]*((float)indices[i]);
    }
}

void compute_square_derivative_with_indices(float* input, float* output,int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = 2*input[i]*((float)indices[i]);
    }
}

void compute_square_root(float* input, float* output, int size){
    if(input == NULL || output == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = sqrtf(input[i]);
    }
}

void compute_square_root_with_indices(float* input, float* output,int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = sqrtf(input[i])*((float)indices[i]);
    }
}

void compute_square_root_derivative_with_indices(float* input, float* output,int* indices, int size){
    if(input == NULL || output == NULL || indices == NULL)
        return;
    int i;
    for(i = 0; i < size; i++){
        output[i] = (1.0/(2*sqrtf(input[i])))*((float)indices[i]);
    }
}
