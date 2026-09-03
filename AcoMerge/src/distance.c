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

float euclidean_distance(float* a, float* b, int n) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float cosine_similarity(float *a, float *b, int n) {
    float dot = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;
	int i;
    for (i = 0; i < n; i++) {
        dot += a[i] * b[i];
        normA += a[i] * b[i];
        normB += a[i] * b[i];
    }

    if (normA == 0.0f || normB == 0.0f) return 0.0f;  // Prevent divide by zero
    return dot / (sqrtf(normA) * sqrtf(normB));
}
