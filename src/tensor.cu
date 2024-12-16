#include "model.h"


/* [Tensor Structure] */
/* Tensor
 * @brief - A multi-dimensional matrix containing elements of a single data
 type.
 * @member - buf  : Data buffer containing elements
 * @member - shape: Shape of tensor from outermost dimension to innermost
 dimension e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
 */
Tensor::Tensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(float)));
  CHECK_CUDA(cudaMemset(buf, 0.0f, N_ * sizeof(float)));
}

Tensor::Tensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::~Tensor() {
  if (buf != nullptr) CHECK_CUDA(cudaFree(buf));
}

size_t Tensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}

// Half Tensor

HalfTensor::HalfTensor(const vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(half)));
  CHECK_CUDA(cudaMemset(buf, 0, N_ * sizeof(half)));
}

HalfTensor::HalfTensor(const vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();
  half *temp = (half *)malloc(N_ * sizeof(half));
  for (size_t n = 0; n < N_; n++) {
    temp[n] = (half) (buf_[n]);
  }
  CHECK_CUDA(cudaMalloc(&buf, N_ * sizeof(half)));
  CHECK_CUDA(cudaMemcpy(buf, temp, N_ * sizeof(half), cudaMemcpyHostToDevice));
  free(temp);
}

HalfTensor::~HalfTensor() {
  if (buf != nullptr) CHECK_CUDA(cudaFree(buf));
}

size_t HalfTensor::num_elem() {
  size_t size = 1;
  for (size_t i = 0; i < ndim; i++) { size *= shape[i]; }
  return size;
}
