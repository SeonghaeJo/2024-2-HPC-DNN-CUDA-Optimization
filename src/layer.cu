#include "layer.h"

/*
  One input element (4096 output elements) per thread
  2D thread block size = EMB_SENTENCES * 16
  Total number of threads = n * 16
*/

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


#define PERM_BLOCKDIM 256

__global__ void permute_kernel(float *in, float *out, int n, int s, int H) {
  int x = blockIdx.x * PERM_BLOCKDIM + threadIdx.x;
  int y = blockIdx.y;
  int z = blockIdx.z;
  out[z * s * H + x * s + y] = in[z * s * H + y * H + x];
}

/* Permute
 * @param [in]   in: [n, s, H] = [4096, 16, 4096]
 * @param [out] out: [n, H, s]
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

/* Conv1D 
 * @param [in1]  in: [C, s]
 * @param [in2]   w: [OC, C, K] 
 * @param [in3]   b: [OC]
 * @param [out] out: [OC, os]
 *    
 *    In this model, K is 3, 5, 7, or 9, 
 *    with stride = 1, pad = 0, dilation = 1.
 *    The formula for the output sequence length:
 *      os = (in - K + 2 * pad) / stride + 1
 *          = (s - K + 2 * 0) / 1 + 1
 *          = s - K + 1
 *
 * 'C' is the input channel size
 * 's' is the input sequence length
 * 'OC' is the output channel size
 * 'os' is the output sequence length
 * 'K' is the kernel (or filter) size
 */
void Conv1D(Tensor *in, Tensor *w, Tensor *b, Tensor *out) {
  size_t s = in->shape[1];
  size_t C = in->shape[0];
  size_t OC = w->shape[0];
  size_t K = w->shape[2];
  
  size_t os = s - K + 1;

  for (size_t i = 0; i < OC; i++) {
    for (size_t j = 0; j < os; j++) {
      float val = 0.f;
      for (size_t k = 0; k < C; k++) {
        for (size_t l = 0; l < K; l++) {
          val += in->buf[k * s + j + l] * 
                  w->buf[i * C * K + k * K + l];
        }
      }
      out->buf[i * os + j] = val + b->buf[i];
    }
  }
}

/* ReLU
 * @param [in & out] inout: [N]
 * 'N' is the number of elements in the tensor.
 */
void ReLU(Tensor *inout) {
  size_t N = inout->num_elem();

  for (size_t i = 0; i < N; i++) {
    inout->buf[i] = inout->buf[i] > 0 ? inout->buf[i] : 0;
  }
}
/* ReLU CUDA kernel */
__global__ void ReLU_Kernel(float *inout, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    inout[i] = inout[i] > 0 ? inout[i] : 0;
  }
}
/* ReLU using CUDA */
void ReLU_CUDA(Tensor *inout) {
  size_t N = inout->num_elem();

  float *d_inout;
  CHECK_CUDA(cudaMalloc(&d_inout, N * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inout, inout->buf, N * sizeof(float), 
                        cudaMemcpyHostToDevice));

  ReLU_Kernel<<<(N + 255) / 256, 256>>>(d_inout, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(inout->buf, d_inout, N * sizeof(float), 
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_inout));
}

/* GetMax
 * @param [in]   in: [C, s]
 * @param [out] out: [C]
 *    
 *    This layer is to get the max value along the sequence dim.
 *    The formula for this layer: out = max(in, dim=-1)
 * 
 * 'C' is the channel size
 * 's' is the sequence length
 */
void GetMax(Tensor *in, Tensor *out) {
  size_t C = in->shape[0];
  size_t s = in->shape[1];

  for (size_t i = 0; i < C; i++) {
    out->buf[i] = in->buf[i * s];
    for (size_t j = 1; j < s; j++) {
      out->buf[i] = in->buf[i * s + j] > out->buf[i] ? 
        in->buf[i * s + j] : out->buf[i];
    }
  }
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
  size_t N1 = in1->shape[0];
  size_t N2 = in2->shape[0];
  size_t N3 = in3->shape[0];
  size_t N4 = in4->shape[0];

  for (size_t i = 0; i < N1; i++) {
    out->buf[i] = in1->buf[i];
  }
  for (size_t i = 0; i < N2; i++) {
    out->buf[N1 + i] = in2->buf[i];
  }
  for (size_t i = 0; i < N3; i++) {
    out->buf[N1 + N2 + i] = in3->buf[i];
  }
  for (size_t i = 0; i < N4; i++) {
    out->buf[N1 + N2 + N3 + i] = in4->buf[i];
  }
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


