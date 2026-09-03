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

/*random number between 0 and 1*/
float normalized_random(){
    return (float)((float)rand() / (float)RAND_MAX);
}

double random_beta(double alpha, double beta) {
    double u = normalized_random();
    double v = normalized_random();

    double x = pow(u, 1.0 / alpha);
    double y = pow(v, 1.0 / beta);

    return x / (x + y);
}

float signed_normalized_random(float n){
    if (normalized_random() >= 0.5)
            return normalized_random()*n;
    return -normalized_random()*n;
}

/* a random number from a gaussian distribution with mean 0 and std 1*/
float random_normal (){
  return sqrtf(-2.0 * log (normalized_random ())) * cos (2.0 * M_PI * normalized_random ());
  }

/* a random number from a gaussian distribution with mean 0 and std = n
*/
float random_general_gaussian(float mean, float std){
    return mean + std*random_normal();
}

float chaos_next_sequence(float a){
    if(a < 0.5){
        return 16.0*a*(0.5-a);
    }
    return 1.0-16.0*a*(0.5-a)*(1.0-a);
}

/* a random number from a gaussian distribution with mean 0 and std = sqrtf(1/n)
 * where n is the number of neuron of layer l-1*/
float random_general_gaussian_xavier_init(float n){
    return random_general_gaussian(0, sqrtf(1.0/n));
}

// suggested when input neurons and output neurons are the same, n1 are the input neurons, n2 the output
float random_general_gaussian_xavier_init2(float n1,float n2){
    return random_general_gaussian(0, sqrtf(2.0/(n1+n2)));
}

// n is the number of input neurons
float random_general_gaussian_kaiming_init(float n){
    return random_general_gaussian(0, sqrtf(2.0/n));
}

// n is the number of input neurons
float signed_kaiming_constant(float n){
    return signed_normalized_random(sqrtf(2.0/n));
}
