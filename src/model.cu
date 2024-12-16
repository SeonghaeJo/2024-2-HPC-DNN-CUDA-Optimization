#include <mpi.h>

#include <cstdio>
#include <omp.h>

#include "layer.h"
#include "model.h"

#define NUM_GPUS 4
#define NUM_NODES 4

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w[NUM_GPUS];
HalfTensor *conv0_w[NUM_GPUS];
HalfTensor *conv1_w[NUM_GPUS];
HalfTensor *conv2_w[NUM_GPUS];
HalfTensor *conv3_w[NUM_GPUS];
Parameter *conv0_b[NUM_GPUS];
Parameter *conv1_b[NUM_GPUS];
Parameter *conv2_b[NUM_GPUS];
Parameter *conv3_b[NUM_GPUS];
Parameter *linear0_w[NUM_GPUS], *linear0_b[NUM_GPUS];
Parameter *linear1_w[NUM_GPUS], *linear1_b[NUM_GPUS];
Parameter *linear2_w[NUM_GPUS], *linear2_b[NUM_GPUS];
Parameter *linear3_w[NUM_GPUS], *linear3_b[NUM_GPUS];

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos[4] = {0};

  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int g = 0; g < NUM_GPUS; g++) {

    CHECK_CUDA(cudaSetDevice(g));

    emb_w[g] = new Parameter({21635, 4096}, param + pos[g]);
    pos[g] += 21635 * 4096; 

    conv0_w[g] = new HalfTensor({1024, 4096, 3}, param + pos[g]);
    pos[g] += 1024 * 4096 * 3; 
    conv0_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv1_w[g] = new HalfTensor({1024, 4096, 5}, param + pos[g]);
    pos[g] += 1024 * 4096 * 5; 
    conv1_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv2_w[g] = new HalfTensor({1024, 4096, 7}, param + pos[g]);
    pos[g] += 1024 * 4096 * 7;
    conv2_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv3_w[g] = new HalfTensor({1024, 4096, 9}, param + pos[g]);
    pos[g] += 1024 * 4096 * 9;
    conv3_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    linear0_w[g] = new Parameter({2048, 4096}, param + pos[g]);
    pos[g] += 2048 * 4096;
    linear0_b[g] = new Parameter({2048}, param + pos[g]);
    pos[g] += 2048;

    linear1_w[g] = new Parameter({1024, 2048}, param + pos[g]);
    pos[g] += 1024 * 2048;
    linear1_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    linear2_w[g] = new Parameter({512, 1024}, param + pos[g]);
    pos[g] += 512 * 1024;
    linear2_b[g] = new Parameter({512}, param + pos[g]);
    pos[g] += 512;

    linear3_w[g] = new Parameter({2, 512}, param + pos[g]);
    pos[g] += 2 * 512;
    linear3_b[g] = new Parameter({2}, param + pos[g]);
    pos[g] += 2;
  }

  if (pos[0] != param_size) {
    fprintf(stderr, "Parameter size mismatched: %zu != %zu\n", 
            pos[0], param_size);
    exit(EXIT_FAILURE);
  }
}

void free_parameters() {
  for (int g = 0; g < NUM_GPUS; g++) {
    CHECK_CUDA(cudaSetDevice(g));

    delete emb_w[g];
    delete conv0_w[g];
    delete conv0_b[g];
    delete conv1_w[g];
    delete conv1_b[g];
    delete conv2_w[g];
    delete conv2_b[g];
    delete conv3_w[g];
    delete conv3_b[g];
    delete linear0_w[g];
    delete linear0_b[g];
    delete linear1_w[g];
    delete linear1_b[g];
    delete linear2_w[g];
    delete linear2_b[g];
    delete linear3_w[g];
    delete linear3_b[g];
  }
}

/* [Model Activations] 
 * _a: Activation buffer
 */
Activation *emb_a[NUM_GPUS];
Activation *permute_a[NUM_GPUS];
HalfTensor *conv0_in_spread[NUM_GPUS];
HalfTensor *conv1_in_spread[NUM_GPUS];
HalfTensor *conv2_in_spread[NUM_GPUS];
HalfTensor *conv3_in_spread[NUM_GPUS];
Activation *conv0_a[NUM_GPUS], *pool0_a[NUM_GPUS];
Activation *conv1_a[NUM_GPUS], *pool1_a[NUM_GPUS];
Activation *conv2_a[NUM_GPUS], *pool2_a[NUM_GPUS];
Activation *conv3_a[NUM_GPUS], *pool3_a[NUM_GPUS];
Activation *concat_a[NUM_GPUS];
Activation *linear0_a[NUM_GPUS], *linear1_a[NUM_GPUS], *linear2_a[NUM_GPUS], *linear3_a[NUM_GPUS];

int *inputs_mpi;
float *outputs_mpi;

int *inputs_d[NUM_GPUS];

void alloc_activations() {
  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int g = 0; g < NUM_GPUS; g++) {
    CHECK_CUDA(cudaSetDevice(g));

    emb_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, SEQ_LEN, 4096});
    permute_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096, SEQ_LEN});
    conv0_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024, SEQ_LEN - 2});
    pool0_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024});
    conv1_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024, SEQ_LEN - 4});
    pool1_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024});
    conv2_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024, SEQ_LEN - 6});
    pool2_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024});
    conv3_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024, SEQ_LEN - 8});
    pool3_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024});
    concat_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096});
    linear0_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 2048});
    linear1_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 1024});
    linear2_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 512});
    linear3_a[g] = new Activation({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 2});

    conv0_in_spread[g] = new HalfTensor({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096 * 3, 16});
    conv1_in_spread[g] = new HalfTensor({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096 * 5, 16});
    conv2_in_spread[g] = new HalfTensor({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096 * 7, 16});
    conv3_in_spread[g] = new HalfTensor({NUM_SENTENCES / NUM_NODES / NUM_GPUS, 4096 * 9, 16});

    CHECK_CUDA(cudaMalloc(&inputs_d[g], NUM_SENTENCES / NUM_NODES / NUM_GPUS * SEQ_LEN * sizeof(int)));
  }

  CHECK_CUDA(cudaMallocHost((void **)&inputs_mpi, NUM_SENTENCES * SEQ_LEN * sizeof(int)));
  memset(inputs_mpi, 0, NUM_SENTENCES * SEQ_LEN * sizeof(int));

  CHECK_CUDA(cudaMallocHost((void **)&outputs_mpi, NUM_SENTENCES * 2 * sizeof(float)));
  memset(outputs_mpi, 0, NUM_SENTENCES * 2 * sizeof(float));
}

void free_activations() {
  for (int g = 0; g < NUM_GPUS; g++) {

    CHECK_CUDA(cudaSetDevice(g));

    delete emb_a[g];
    delete permute_a[g];
    delete conv0_a[g];
    delete pool0_a[g];
    delete conv1_a[g];
    delete pool1_a[g];
    delete conv2_a[g];
    delete pool2_a[g];
    delete conv3_a[g];
    delete pool3_a[g];
    delete concat_a[g];
    delete linear0_a[g];
    delete linear1_a[g];
    delete linear2_a[g];
    delete linear3_a[g];

    delete conv0_in_spread[g];
    delete conv1_in_spread[g];
    delete conv2_in_spread[g];
    delete conv3_in_spread[g];

    CHECK_CUDA(cudaFree(inputs_d[g]));
  }

  CHECK_CUDA(cudaFreeHost(inputs_mpi));
  CHECK_CUDA(cudaFreeHost(outputs_mpi));
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {

  if (n_samples != 1 && n_samples != NUM_SENTENCES) {
    printf("predict_sentiment : n_sample (%lu) is not equal to NUM_SENTENCES (%d)\n", n_samples, NUM_SENTENCES);
    exit(1);
  }

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_rank != 0) {
    inputs = inputs_mpi;
    outputs = outputs_mpi;
  }

  int mpi_in_count = (NUM_SENTENCES / NUM_NODES) * SEQ_LEN;
  MPI_Scatter((void *)inputs, mpi_in_count, MPI_INT, 
              (void *)(&inputs[mpi_in_count * mpi_rank]), mpi_in_count, MPI_INT,
              0, MPI_COMM_WORLD);

  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int g = 0; g < NUM_GPUS; g++) {

    CHECK_CUDA(cudaSetDevice(g));

    // inputs = [num_sentences * SEQ_LEN]
    int input_start = (NUM_SENTENCES / NUM_NODES / NUM_GPUS) * SEQ_LEN * (NUM_GPUS * mpi_rank + g);
    CHECK_CUDA(cudaMemcpy(inputs_d[g], &inputs[input_start], 
                          NUM_SENTENCES / NUM_NODES / NUM_GPUS * SEQ_LEN * sizeof(int), 
                          cudaMemcpyHostToDevice));

    Embedding(inputs_d[g], emb_w[g], emb_a[g]);

    Permute(emb_a[g], permute_a[g]);

    Conv1D(permute_a[g], conv0_w[g], conv0_b[g], conv0_a[g], conv0_in_spread[g]);

    ReLU(conv0_a[g]);
    GetMax(conv0_a[g], pool0_a[g]);

    Conv1D(permute_a[g], conv1_w[g], conv1_b[g], conv1_a[g], conv1_in_spread[g]);
    ReLU(conv1_a[g]);
    GetMax(conv1_a[g], pool1_a[g]);

    Conv1D(permute_a[g], conv2_w[g], conv2_b[g], conv2_a[g], conv2_in_spread[g]);
    ReLU(conv2_a[g]);
    GetMax(conv2_a[g], pool2_a[g]);

    Conv1D(permute_a[g], conv3_w[g], conv3_b[g], conv3_a[g], conv3_in_spread[g]);
    ReLU(conv3_a[g]);
    GetMax(conv3_a[g], pool3_a[g]);

    Concat(pool0_a[g], pool1_a[g], pool2_a[g], pool3_a[g], concat_a[g]);

    if (n_samples == 1) {
      continue;
    }

    Linear(concat_a[g], linear0_w[g], linear0_b[g], linear0_a[g]);
    ReLU(linear0_a[g]);

    Linear(linear0_a[g], linear1_w[g], linear1_b[g], linear1_a[g]);
    ReLU(linear1_a[g]);

    Linear(linear1_a[g], linear2_w[g], linear2_b[g], linear2_a[g]);
    ReLU(linear2_a[g]);

    Linear_narrow(linear2_a[g], linear3_w[g], linear3_b[g], linear3_a[g]);

    int output_start = (NUM_SENTENCES / NUM_NODES / NUM_GPUS) * 2 * (NUM_GPUS * mpi_rank + g);
    // outputs = [num_sentences * N_CLASSES]
    CHECK_CUDA(cudaMemcpy(&outputs[output_start], linear3_a[g]->buf,
                          (NUM_SENTENCES / NUM_NODES / NUM_GPUS) * 2 * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  int mpi_out_count = (NUM_SENTENCES / NUM_NODES) * 2;
  MPI_Gather((void *)(&outputs[mpi_out_count * mpi_rank]), mpi_out_count, MPI_FLOAT,
              (void *)(&outputs[mpi_out_count * mpi_rank]), mpi_out_count, MPI_FLOAT, 
              0, MPI_COMM_WORLD);
}