#include "layer.h"

// Embedding Kernel

#define EMB_BLOCKDIM 256

__global__ void embedding_kernel(int *in, float *w, float *out, int n, int s, int H) {
  int word = in[blockIdx.z * s + blockIdx.y];
  int emb_idx = blockIdx.x * EMB_BLOCKDIM + threadIdx.x;
  out[blockIdx.z * s * H + blockIdx.y * H + emb_idx] = w[word * H + emb_idx]; 
}

/* Embedding
 * @param [in1]  in: [n, s]
 * @param [in2]   w: [NUM_VOCAB, H]
 * @param [out] out: [n, s, H]
 * 'n' is NUM_SENTENCES / NUM_GPUS (4096)
 * 's' is the sequence length (16)
 * 'H' is the embedding dimension (4096)
 */
void Embedding(int *in, Tensor* w, Tensor *out) {
  size_t n = out->shape[0];
  size_t s = out->shape[1];
  size_t H = out->shape[2];

  dim3 blockDim(EMB_BLOCKDIM, 1, 1);
  dim3 gridDim(H / EMB_BLOCKDIM, s, n);
  embedding_kernel<<<gridDim, blockDim>>>(in, w->buf, out->buf, n, s, H);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Permute Kernel

#define PERM_BLOCKDIM 256

__global__ void permute_kernel(float *in, float *out, int n, int s, int H) {
  int x = blockIdx.x * PERM_BLOCKDIM + threadIdx.x;
  int y = blockIdx.y;
  int z = blockIdx.z;
  out[z * s * H + x * s + y] = in[z * s * H + y * H + x];
}

/* Permute
 * @param [in]   in: [n, s, H] = [4096, 16, 4096]
 * @param [out] out: [n, H, s] = [4096, 4096, 16]
 */
void Permute(Tensor *in, Tensor *out) {
  size_t n = in->shape[0];
  size_t s = in->shape[1];
  size_t H = in->shape[2];

  dim3 blockDim(PERM_BLOCKDIM, 1, 1);
  dim3 gridDim(H / PERM_BLOCKDIM, s, n);

  permute_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, n, s, H);
  CHECK_CUDA(cudaDeviceSynchronize());
}


// Conv1D Kernel

#define CONV_TS 64
// Number of Tile Iterations = C / CONV_TS = 4096 / 64 

__global__ void conv1d_kernel(
  float *in, float *w, float *b, float *out,
  int n, int C, int s, int OC, int K, int os) {

  __shared__ float inTile[CONV_TS * s];
  __shared__ float wTile[CONV_TS * K];

  float outReg[os] = {0.0f};

  int ocIdx = blockIdx.x;
  int nIdx = blockIdx.y;

  w += ocIdx * C * K;
  in += nIdx * C * s;

  // Tile Iteration
  for(int t = 0; t < C; t += CONV_TS) {

    // Copy from GMEM to SMEM
    // threadIdx.x <= K - 1  && threadIdx.y <= CONV_TS - 1
    int wIdx = threadIdx.y * K + threadIdx.x;
    wTile[wIdx] = w[wIdx];
    for(int i = wIdx; i < CONV_TS * s; i += CONV_TS * K) {
      inTile[i] = in[i];
    }

    __syncthreads();

    float wVal = wTile[threadIdx.y * K + threadIdx.x];
    for(int i = 0; i < os; i++) {
      outReg[i] += inTile[threadIdx.y * s + (threadIdx.x + i)] * wVal;
    }

    // Transfer Tile
    w += CONV_TS * K;
    in += CONV_TS * s;

    __syncthreads();
  }

  for(int i = 0; i < os; i++) {
    out[nIdx * OC * os + ocIdx * os + i] = outReg[i] + b[ocIdx];
  }
}

/* Conv1D 
 * @param [in1]  in: [n, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [n, OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'n' is the number of samples (4096)
 * 'C' is the input channel size (4096)
 * 's' is the input sequence length (16)
 * 'OC' is the output channel size (1024)
 * 'os' is the output sequence length (14 or 12 or 10 or 8)
 * 'K' is the kernel (or filter) size (3 or 5 or 7 or 9)
 */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t n = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim(K, CONV_TS, 1); // CONV_TS * K
  dim3 gridDim(OC, n, 1); // n * OC

  conv1d_kernel<<<gridDim, blockDim>>>(in->buf, w->buf, b->buf, out->buf, n, C, s, OC, K, os);
  CHECK_CUDA(cudaDeviceSynchronize());
}


// ReLU Kernel

__global__ void relu_kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  dim3 blockDim(256, 1, 1);
  dim3 gridDim(N/256, 1, 1);

  relu_kernel<<<gridDim, blockDim>>>(inout->buf, N);
  CHECK_CUDA(cudaDeviceSynchronize());
}


// GetMax Kernel

#define GETMAX_BLOCKDIM 128

__global__ void getmax_kernel(float *in, float *out, int C, int s) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float result = -1.0f; // Tensor elements >= 0 after ReLU
  in += blockIdx.y * C * s + tid * s;
  for (int i = 0; i < s; i++) {
    result = fmaxf(result, in[i]);
  }
  out[blockIdx.y * C + tid] = result;
}

/* GetMax
 * @param [in]   in: [n, C, s]
 * @param [out] out: [n, C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'n' is the number of sentences
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t n = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];

  dim3 blockDim(GETMAX_BLOCKDIM, 1, 1);
  dim3 gridDim(C / GETMAX_BLOCKDIM, n, 1);

  getmax_kernel<<<gridDim, blockDim>>>(in->buf, out->buf, C, s);
  CHECK_CUDA(cudaDeviceSynchronize());
}


// Concat Kernel

#define CONCAT_BLOCKDIM 128

__global__ void concat_kernel(
  float *in1, float *in2, float *in3, float *in4, float *out, int C) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  out += blockIdx.y * (4 * C);
  in1 += blockIdx.y * C;
  in2 += blockIdx.y * C;
  in3 += blockIdx.y * C;
  in4 += blockIdx.y * C;

  out[tid] = in1[tid];
  out[tid + C] = in2[tid];
  out[tid + 2 * C] = in3[tid];
  out[tid + 3 * C] = in4[tid];
}

/* Concat
 * @param [in1] in1: [N1]
 * @param [in2] in2: [N2]
 * @param [in3] in3: [N3]
 * @param [in4] in4: [N4]
 * @param [out] out: [N1 + N2 + N3 + N4]
 * 'N1', 'N2', 'N3', and 'N4' are the num of elems in the tensors.
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {
  // size_t N1 = in1->shape[0];
  // size_t N2 = in2->shape[0];
  // size_t N3 = in3->shape[0];
  // size_t N4 = in4->shape[0];

  // for (size_t i = 0; i < N1; i++) {
  //   out->buf[i] = in1->buf[i];
  // }
  // for (size_t i = 0; i < N2; i++) {
  //   out->buf[N1 + i] = in2->buf[i];
  // }
  // for (size_t i = 0; i < N3; i++) {
  //   out->buf[N1 + N2 + i] = in3->buf[i];
  // }
  // for (size_t i = 0; i < N4; i++) {
  //   out->buf[N1 + N2 + N3 + i] = in4->buf[i];
  // }

  size_t n = in1->shape[0];
  size_t C = in1->shape[1];

  dim3 blockDim(CONCAT_BLOCKDIM, 1, 1);
  dim3 gridDim(C / CONCAT_BLOCKDIM, n, 1);

  concat_kernel<<<gridDim, blockDim>>>(in1->buf, in2->buf, in3->buf, in4->buf, out->buf, C);
  CHECK_CUDA(cudaDeviceSynchronize());
}

/* Linear 
 * @param [in1]  in: [N]
 * @param [in2]   w: [M, N]
 * @param [in3]   b: [M]
 * @param [out] out: [M]
 * 'N' is the input feature size
 * 'M' is the output feature size
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t N = in->shape[0];
  size_t M = w->shape[0];

  for (size_t i = 0; i < M; i++) {
    float val = 0.f;
    for (size_t j = 0; j < N; j++) {
      val += in->buf[j] * w->buf[i * N + j];
    }
    out->buf[i] = val + b->buf[i];
  }
}


