#pragma once

#include "tensor.h"

#define NUM_SENTENCES 16384

/* Operations (layers) */
void Embedding(int *in, Tensor *w, Tensor *out);
void Permute(Tensor *in, Tensor *out);
void Conv1D(Tensor *in, HalfTensor *w, Tensor *b, Tensor *out, 
            HalfTensor *in_spread, Tensor *out_padded, cudaStream_t stream);
void ReLU(Tensor *inout);
void GetMax(Tensor *in, Tensor *out);
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out);
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
void Linear_narrow(Tensor *in, Tensor *w, Tensor *b, Tensor *out);
