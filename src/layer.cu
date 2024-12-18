#include "layer.h"

#define SEQ_LEN 16
#define N_CLASSES 2

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


// Conv1D Kernel = Input Spread Kernel + Matmul WMMA Kernel

using namespace nvcuda;

#define SPREAD_BLOCKDIM 512

__global__ void spread_input_kernel(
  const float *in, half *in_spread, int n, int C, int s, int K, int os
) {
  // in = [n, C, s]
  // in_spread = [n, C * K, padded_os (==16)]

  // blockDim = (SPREAD_BLOCKDIM, 1, 1)
  // gridDim = (n, 1, 1)

  in += blockIdx.x * C * s;
  in_spread += blockIdx.x * C * K * 16;

  for (int i = threadIdx.x; i < C * K * os; i += blockDim.x) {
    int spread_row = i / os;
    int spread_col = i % os;
    int in_row = i / (os * K);
    int in_col = ((i % (os * K)) / os) + (i % os);
    in_spread[spread_row * 16 + spread_col] = in[in_row * s + in_col];
  }
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WMMA_BLOCKDIM 1024
#define WARPS_PER_BLOCK (WMMA_BLOCKDIM / 32) // 32
#define WMMA_BLOCK_ROWS (WMMA_M * WARPS_PER_BLOCK) // 512
#define WMMA_TSKA 32
#define WMMA_TSKB 512

__global__ void matmul_wmma_kernel(
  const half *a, const half *b, float *c, const float *bias, float *c_padded, int lm, int ln, int lk, int os
) {
  // a = [OC, C * K] (==w)
  // b = [n, C * K, padded_os] (Zero-padded in_spread)
  // c = [n, OC, os] (c without zero padding)
  // bias = [OC]
  // lm = OC
  // ln = 16 (padded os)
  // lk = C * K
  // blockDim = (WMMA_BLOCKDIM, 1, 1) -> If WMMA_BLOCKDIM == 512, then 16 warps
  // gridDim = (lm / WMMA_BLOCK_ROWS, n, 1)

  // Move to the appropriate batch
  b += blockIdx.y * lk * ln;

  __shared__ __align__(32) half a_shared[WMMA_BLOCK_ROWS * WMMA_TSKA];
  __shared__ __align__(32) half b_shared[WMMA_TSKB * WMMA_N];

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  int localWarpIdx = threadIdx.x / 32;

  for (int bt = 0; bt < lk; bt += WMMA_TSKB) {
    for (int i = threadIdx.x; i < WMMA_TSKB * WMMA_N; i += WMMA_BLOCKDIM) {
      int bRow = i / WMMA_N;
      int bCol = i % WMMA_N;
      b_shared[i] = b[(bt + bRow) * ln + bCol];
    }

    for (int at = 0; at < WMMA_TSKB; at += WMMA_TSKA) {

      for (int i = threadIdx.x; i < WMMA_BLOCK_ROWS * WMMA_TSKA; i += WMMA_BLOCKDIM) {
        int aRow = i / WMMA_TSKA;
        int aCol = i % WMMA_TSKA;
        a_shared[i] = a[((warpIdx / WARPS_PER_BLOCK) * WMMA_BLOCK_ROWS + aRow) * lk + (bt + at) + aCol];
      }

      __syncthreads();

      #pragma unroll
      for (int wt = 0; wt < WMMA_TSKA; wt += WMMA_K) {
        int t = bt + at + wt;

        wmma::load_matrix_sync(a_frag, a_shared + (localWarpIdx * WMMA_M) * WMMA_TSKA + (t % WMMA_TSKA), WMMA_TSKA);
        wmma::load_matrix_sync(b_frag, b_shared + (t % WMMA_TSKB) * WMMA_N, WMMA_N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }

      __syncthreads();
    }
  }

  // Move to the appropriate threadblock in the appropriate batch
  c_padded += blockIdx.y * lm * WMMA_N + blockIdx.x * WMMA_BLOCK_ROWS * WMMA_N;
  c += blockIdx.y * lm * os + blockIdx.x * WMMA_BLOCK_ROWS * os;
  bias += blockIdx.x * WMMA_BLOCK_ROWS;

  wmma::store_matrix_sync(c_padded + (localWarpIdx * WMMA_M) * WMMA_N, c_frag, WMMA_N, wmma::mem_row_major);

  __syncthreads();
  
  // Copy from c_shared to c except for zero padding
  for (int i = threadIdx.x; i < WMMA_BLOCK_ROWS * os; i += WMMA_BLOCKDIM) {
    int cRow = i / os;
    int cCol = i % os;
    c[i] = c_padded[cRow * WMMA_N + cCol] + bias[cRow];
  } 
}

/* Conv1D 
 * @param [in1]  in: [n, C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [n, OC, os]
 * @param [in]  in_spread  : [n, C * K, padded_os]
 * @param [in]  out_padded : [n, OC, padded_os]
 * padded_os = 16
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
void Conv1D(
  Tensor *in, HalfTensor *w, Tensor *b, Tensor *out, 
  HalfTensor *in_spread, Tensor *out_padded,
  cudaStream_t stream, int p, int num_pipelines) {
  size_t n = in->shape[0];
  size_t C = in->shape[1];
  size_t s = in->shape[2];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  size_t os = s - K + 1;

  dim3 blockDim_spread(SPREAD_BLOCKDIM, 1, 1);
  dim3 gridDim_spread(n / num_pipelines, 1, 1);

  spread_input_kernel<<<gridDim_spread, blockDim_spread, 0, stream>>>(
    in->buf + p * n / num_pipelines * C * s, 
    in_spread->buf + p * n / num_pipelines * C * K * 16, 
    n / num_pipelines, C, s, K, os);

  int lm = OC;
  int ln = 16; // padded_os
  int lk = C * K;

  dim3 blockDim_wmma(WMMA_BLOCKDIM, 1, 1);
  dim3 gridDim_wmma(lm / WMMA_BLOCK_ROWS, n / num_pipelines, 1);

  matmul_wmma_kernel<<<gridDim_wmma, blockDim_wmma, 0, stream>>>(
    w->buf, 
    in_spread->buf + p * n / num_pipelines * C * K * 16, 
    out->buf + p * n / num_pipelines * OC * os, 
    b->buf, 
    out_padded->buf + p * n / num_pipelines * OC * 16, 
    lm, ln, lk, os);
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
  float result = 0.0f; // Tensor elements >= 0 after ReLU
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
 * @param [in1] in1: [n, C]
 * @param [in2] in2: [n, C]
 * @param [in3] in3: [n, C]
 * @param [in4] in4: [n, C]
 * @param [out] out: [n, 4 * C]
 */
void Concat(Tensor *in1, Tensor *in2, Tensor *in3, Tensor *in4, 
            Tensor *out) {

  size_t n = in1->shape[0];
  size_t C = in1->shape[1];

  dim3 blockDim(CONCAT_BLOCKDIM, 1, 1);
  dim3 gridDim(C / CONCAT_BLOCKDIM, n, 1);

  concat_kernel<<<gridDim, blockDim>>>(in1->buf, in2->buf, in3->buf, in4->buf, out->buf, C);
  CHECK_CUDA(cudaDeviceSynchronize());
}


// Linear Kernel (except for last layer)

#define TS 128
#define TSK 4
#define W 8
#define VEC 4

__global__ void linear_kernel(float *A, float *B, float *C, float *bias, int M, int N,
                              int K) {
  // C [M * N]
  // A [M * K]
  // B [N * K]
  // bias [N]                        
  
  __shared__ __align__(16) float As[TS * TSK];
  __shared__ __align__(16) float Bs[TS * TSK];

  A += blockIdx.y * TS * K;
  B += blockIdx.x * TS * K;

  float sums[W][W] = {0.0f};

  for(int t = 0; t < K; t += TSK) {

    if (threadIdx.y < 8) { // threadIdx.y range = [0, 7]
      int r = threadIdx.y * 16 + threadIdx.x;
      reinterpret_cast<float4 *>(&As[r * TSK])[0] = reinterpret_cast<float4 *>(&A[r * K])[0];
    }
    else { // threadIdx.y range = [8, 15]
      int r = (threadIdx.y - 8) * 16 + threadIdx.x;
      reinterpret_cast<float4 *>(&Bs[r * TSK])[0] = reinterpret_cast<float4 *>(&B[r * K])[0];
    }

    __syncthreads();
    
    for(int w1 = 0; w1 < W; w1++) {
      int r1 = threadIdx.y + w1 * (TS/W);
      #pragma unroll
      for(int w2 = 0; w2 < W; w2++) {
        int r2 = threadIdx.x + w2 * (TS/W);
        float temp = 0.0f;

        #pragma unroll
        for(int k = 0; k < TSK; k++) {
          temp += As[r1 * TSK + k] * Bs[r2 * TSK + k];
        }

        sums[w1][w2] += temp;
      }
    }

    A += TSK;
    B += TSK;

    __syncthreads();
  }

  int globalCol = blockIdx.x * TS + threadIdx.x;
  int globalRow = blockIdx.y * TS + threadIdx.y;

  for(int w1 = 0; w1 < W; w1++) {
    for(int w2 = 0; w2 < W; w2++) {
      C[(globalRow + w1 * (TS/W)) * N + (globalCol + w2 * (TS/W))] = sums[w1][w2] + bias[globalCol + w2 * (TS/W)];
    }
  }
}

/* Linear 
 * @param [in1]  in: [M, K] // M = NUM_SENTENCES / NUM_GPUS
 * @param [in2]   w: [N, K]
 * @param [in3]   b: [N]
 * @param [out] out: [M, N]
 */
void Linear(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = in->shape[0];
  size_t K = in->shape[1];
  size_t N = w->shape[0];

  dim3 blockDim(TS/W, TS/W, 1);
  dim3 gridDim(N/TS, M/TS, 1);

  linear_kernel<<<gridDim, blockDim>>>(in->buf, w->buf, out->buf, b->buf, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Linear Kernel for last layer (N = 2)

#define NTS 64
#define NTSK 16

__global__ void linear_narrow_kernel(float *A, float *B, float *C, float *bias, int M, int N,
                              int K) {
  
  // N = 2, K = 512
  // C [M * N]
  // A [M * K]
  // B [N * K]
  // bias [N]                        
  
  __shared__ float As[NTS * NTSK];
  __shared__ float Bs[1024]; // N * K = 2 * 512

  int tid = threadIdx.y * N + threadIdx.x;

  // Copy from the entire B to Bs (GMEM -> SMEM)
  for(int i = tid; i < N * K; i += N * NTS) {
    Bs[i] = B[i];
  }

  A += blockIdx.x * NTS * K;

  float sum = 0.0f;

  for(int t = 0; t < K; t += NTSK) {

    for(int i = tid; i < NTS * NTSK; i += N * NTS) {
      int r = i / NTSK;
      int c = i % NTSK;
      As[i] = A[r * K + c];
    }

    __syncthreads();

    for(int k = 0; k < NTSK; k++) {
      sum += As[threadIdx.y * NTSK + k] * Bs[threadIdx.x * K + (t + k)];
    }
    
    A += NTSK;

    __syncthreads();
  }

  int globalRow = blockIdx.x * NTS + threadIdx.y;
  C[globalRow * N + threadIdx.x] = sum + bias[threadIdx.x];
}

void Linear_narrow(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t M = in->shape[0];
  size_t K = in->shape[1];
  size_t N = w->shape[0]; // 2 (Narrow N)

  dim3 blockDim(N, NTS, 1);
  dim3 gridDim(M/NTS, 1, 1);

  linear_narrow_kernel<<<gridDim, blockDim>>>(in->buf, w->buf, out->buf, b->buf, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());
}
