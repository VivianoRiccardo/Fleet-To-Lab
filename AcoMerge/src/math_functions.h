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
#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__

float sigmoid(float x);
void sigmoid_array(float* input, float* output, int size);
float tanhh(float x);
void tanh_array(float* input, float* output, int size);
float float_max(float a, float b);
float relu(float x);
void relu_array(float* input, float* output, int size);
float elu(float x, float alpha);
void elu_array(float* input, float* output, int size);
float leaky_relu(float x);
void leaky_relu_array(float* input, float* output, int size);
float gelu(float x);
void gelu_array(float* input, float* output, int size);
void softmax(float* input, float* output, int size);
float focal_loss(float y_hat, float y, float gamma);
void focal_loss_array(float* y_hat, float* y,float* output, float gamma, int size);
float float_abs(float a);
void compute_prob_average_double(float* input, float* output, int size);
float derivative_sigmoid(float x);
float derivative_tanhh(float x);
void derivative_sigmoid_array(float* input, float* output, int size);
void derivative_tanh_array(float* input, float* output, int size);
float derivative_relu(float x);
float derivative_elu(float x, float alpha);
float derivative_leaky_relu(float x);
float derivative_gelu(float x);
void derivative_softmax(float* output,float* softmax_arr,float* error, int size);
float derivative_focal_loss(float y_hat, float y, float gamma);
void derivative_focal_loss_array(float* y_hat, float* y, float* output, float gamma, int size);
void derivative_relu_array(float* input, float* output, int size);
void derivative_leaky_relu_array(float* input, float* output, int size);
void derivative_elu_array(float* input, float* output, int size);
void derivative_gelu_array(float* input, float* output, int size);
double cdf_normal(double x);
double pdf_normal(double x);
double d_acquisition_function_d_mu(double mu, double sigma, double best_so_far);
void update_hyperparameters(float *hyperparameters, double step_size, double mu, double sigma, double best_so_far);
float b_spline(float x, float* knots, int degree, int index, int first_time);
float silu(float x);
void compute_general_derivative_loss(float* y, float* y_hat, float* output, float gamma, int size, int flag);
void computer_derivative_loss(float* y, float gamma, edge_popup_trainer* e, int index);
void sigmoid_array_with_flags(float* input, float* output,int* flags, int size);
void softmax_with_flags(float* input, float* output,int* flags, int size);
void derivative_sigmoid_array_with_flags(float* input, float* output,int* flags, int size);
void derivative_softmax_with_flags(float* output,float* softmax_arr,float* error, int* flags, int size);
void compute_inverse(float* input, float* output, int size);
void compute_inverse_with_indices(float* input, float* output, int* indices, int size);
void compute_inverse_derivative_with_indices(float* input, float* output, int* indices, int size);
void dot1D_float_with_int(float* input1, int* input2, float* output, int size);
void mul_value_float_with_int(int* input, float value, float* output, int size);
void compute_square(float* input, float* output, int size);
void compute_square_with_indices(float* input, float* output,int* indices, int size);
void compute_square_derivative_with_indices(float* input, float* output,int* indices, int size);
void compute_square_root(float* input, float* output, int size);
void compute_square_root_with_indices(float* input, float* output,int* indices, int size);
void compute_square_root_derivative_with_indices(float* input, float* output,int* indices, int size);
void mul_value(float* input, float value, float* output, int size);
void dot1D(float* input1, float* input2, float* output, int size);
void derivative_tanh_array_with_flags(float* input, float* output,int* flags, int size);
void derivative_elu_array_with_flags(float* input, float* output,int* flags, int size);
void div1D(float* input1, float* input2, float* output, int size);
void dif1D(float* input1, float* input2, float* output, int size);
void exp1D(float* input1, float* input2, float* output, int size);
void exp1D_with_flags(float* input1, float* input2, float* output, int* flags, int size);

#endif
