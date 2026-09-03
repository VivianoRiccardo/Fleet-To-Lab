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


void nadam(float* w, float* m, float* n, float g,  float ut, float vt, float u0, float v0, float batch_size, float epsilon, float lr){
	float ut1 = ut*u0;
	g/=batch_size;
	(*m) = u0*(*m)+(1.0-u0)*g;
	(*n) = v0*(*n)+(1.0-v0)*g*g;
	float m_hat = (u0*(*m)/(1.0-ut*u0))+((1-u0)*g/(1.0-ut));
	float n_hat = v0*(*n)/(1.0-vt);
	(*w)-=lr*m_hat/sqrtf(n_hat+epsilon);
}
