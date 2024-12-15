#include <mpi.h>

#include <cstdio>
#include <omp.h>

#include "layer.h"
#include "model.h"

// #define DEBUG 1

#define NUM_GPUS 4

/* [Model Parameters]
 * _w: Weight parameter
 * _b: Bias parameter
 */
Parameter *emb_w[NUM_GPUS];
Parameter *conv0_w[NUM_GPUS], *conv0_b[NUM_GPUS];
Parameter *conv1_w[NUM_GPUS], *conv1_b[NUM_GPUS];
Parameter *conv2_w[NUM_GPUS], *conv2_b[NUM_GPUS];
Parameter *conv3_w[NUM_GPUS], *conv3_b[NUM_GPUS];
Parameter *linear0_w[NUM_GPUS], *linear0_b[NUM_GPUS];
Parameter *linear1_w[NUM_GPUS], *linear1_b[NUM_GPUS];
Parameter *linear2_w[NUM_GPUS], *linear2_b[NUM_GPUS];
Parameter *linear3_w[NUM_GPUS], *linear3_b[NUM_GPUS];

void alloc_and_set_parameters(float *param, size_t param_size) {
  size_t pos[4] = {0};

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  #ifdef DEBUG
  if (mpi_rank == 0) printf("alloc_and_set_parameters\n");
  #endif

  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int g = 0; g < NUM_GPUS; g++) {

    CHECK_CUDA(cudaSetDevice(g));

    emb_w[g] = new Parameter({21635, 4096}, param + pos[g]);
    pos[g] += 21635 * 4096; 

    conv0_w[g] = new Parameter({1024, 4096, 3}, param + pos[g]);
    pos[g] += 1024 * 4096 * 3; 
    conv0_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv1_w[g] = new Parameter({1024, 4096, 5}, param + pos[g]);
    pos[g] += 1024 * 4096 * 5; 
    conv1_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv2_w[g] = new Parameter({1024, 4096, 7}, param + pos[g]);
    pos[g] += 1024 * 4096 * 7;
    conv2_b[g] = new Parameter({1024}, param + pos[g]);
    pos[g] += 1024;

    conv3_w[g] = new Parameter({1024, 4096, 9}, param + pos[g]);
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
Activation *conv0_a[NUM_GPUS], *pool0_a[NUM_GPUS];
Activation *conv1_a[NUM_GPUS], *pool1_a[NUM_GPUS];
Activation *conv2_a[NUM_GPUS], *pool2_a[NUM_GPUS];
Activation *conv3_a[NUM_GPUS], *pool3_a[NUM_GPUS];
Activation *concat_a[NUM_GPUS];
Activation *linear0_a[NUM_GPUS], *linear1_a[NUM_GPUS], *linear2_a[NUM_GPUS], *linear3_a[NUM_GPUS];

int *inputs_d[NUM_GPUS];

void alloc_activations() {
  #pragma omp parallel for num_threads(NUM_GPUS)
  for (int g = 0; g < NUM_GPUS; g++) {
    CHECK_CUDA(cudaSetDevice(g));

    emb_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, SEQ_LEN, 4096});
    permute_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 4096, SEQ_LEN});
    conv0_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024, SEQ_LEN - 2});
    pool0_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024});
    conv1_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024, SEQ_LEN - 4});
    pool1_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024});
    conv2_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024, SEQ_LEN - 6});
    pool2_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024});
    conv3_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024, SEQ_LEN - 8});
    pool3_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024});
    concat_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 4096});
    linear0_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 2048});
    linear1_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 1024});
    linear2_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 512});
    linear3_a[g] = new Activation({NUM_SENTENCES / NUM_GPUS, 2});

    CHECK_CUDA(cudaMalloc(&inputs_d[g], NUM_SENTENCES / NUM_GPUS * SEQ_LEN * sizeof(int)));
  }
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

    CHECK_CUDA(cudaFree(inputs_d[g]));
  }
}

/* [Model Computation: Sentiment Analysis Task] */
void predict_sentiment(int *inputs, float *outputs, size_t n_samples) {

  if (n_samples != 1 && n_samples != NUM_SENTENCES) {
    printf("predict_sentiment : n_sample (%lu) is not equal to NUM_SENTENCES (%d)\n", n_samples, NUM_SENTENCES);
    exit(1);
  }

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {

    #pragma omp parallel for num_threads(NUM_GPUS)
    for (int g = 0; g < NUM_GPUS; g++) {

      CHECK_CUDA(cudaSetDevice(g));

      // inputs = [num_sentences * SEQ_LEN]
      CHECK_CUDA(cudaMemcpy(inputs_d[g], &inputs[(NUM_SENTENCES / NUM_GPUS) * SEQ_LEN * g], 
                            NUM_SENTENCES / NUM_GPUS * SEQ_LEN * sizeof(int), 
                            cudaMemcpyHostToDevice));

      #ifdef DEBUG
      if (g == 0) {
        printf("input[%d, %d] = %d, %d\n", 0, 1, inputs[0], inputs[1]);
        printf("input[%d, %d] = %d, %d\n", 16, 17, inputs[16], inputs[17]);
      }
      #endif

      Embedding(inputs_d[g], emb_w[g], emb_a[g]);

      #ifdef DEBUG
      if (g == 0) {
        printf("Embedding\n");
        size_t emb_w_size = sizeof(float) * 21635 * 4096;
        float *emb_w_debug = (float *)malloc(emb_w_size);
        CHECK_CUDA(cudaMemcpy(emb_w_debug, emb_w[g]->buf, emb_w_size, cudaMemcpyDeviceToHost));
        printf("emb_w[%d][%d] = %f\n", 12, 0, emb_w_debug[12 * 4096]);
        printf("emb_w[%d][%d] = %f\n", 12, 1, emb_w_debug[12 * 4096 + 1]);
        printf("emb_w[%d][%d] = %f\n", 119, 0, emb_w_debug[119 * 4096]);
        printf("emb_w[%d][%d] = %f\n", 119, 1, emb_w_debug[119 * 4096 + 1]);
        printf("emb_w[%d][%d] = %f\n", 308, 0, emb_w_debug[308 * 4096]);
        printf("emb_w[%d][%d] = %f\n", 308, 1, emb_w_debug[308 * 4096 + 1]);
        printf("emb_w[%d][%d] = %f\n", 2, 0, emb_w_debug[2 * 4096]);
        printf("emb_w[%d][%d] = %f\n", 2, 1, emb_w_debug[2 * 4096 + 1]);
        size_t emb_a_size = sizeof(float) * (NUM_SENTENCES / NUM_GPUS) * 16 * 4096;
        float *emb_a_debug = (float *)malloc(emb_a_size);
        CHECK_CUDA(cudaMemcpy(emb_a_debug, emb_a[g]->buf, emb_a_size, cudaMemcpyDeviceToHost));
        printf("emb_a[%d][%d][%d] = %f\n", 0, 0, 0, emb_a_debug[0]);
        printf("emb_a[%d][%d][%d] = %f\n", 0, 0, 1, emb_a_debug[1]);
        printf("emb_a[%d][%d][%d] = %f\n", 0, 1, 0, emb_a_debug[4096]);
        printf("emb_a[%d][%d][%d] = %f\n", 0, 1, 1, emb_a_debug[4097]);
        printf("emb_a[%d][%d][%d] = %f\n", 1, 0, 0, emb_a_debug[16 * 4096]);
        printf("emb_a[%d][%d][%d] = %f\n", 1, 0, 1, emb_a_debug[16 * 4096 + 1]);
        printf("emb_a[%d][%d][%d] = %f\n", 1, 1, 0, emb_a_debug[16 * 4096 + 4096]);
        printf("emb_a[%d][%d][%d] = %f\n", 1, 1, 1, emb_a_debug[16 * 4096 + 4096 + 1]);
        free(emb_w_debug);
        free(emb_a_debug);
      }
      #endif

      Permute(emb_a[g], permute_a[g]);

      #ifdef DEBUG
      if (g == 0) {
        printf("Permute\n");
        size_t perm_a_size = sizeof(float) * (NUM_SENTENCES / NUM_GPUS) * 4096 * 16;
        float *perm_a_debug = (float *)malloc(perm_a_size);
        CHECK_CUDA(cudaMemcpy(perm_a_debug, permute_a[g]->buf, perm_a_size, cudaMemcpyDeviceToHost));
        printf("perm_a[%d][%d][%d] = %f\n", 0, 0, 0, perm_a_debug[0]);
        printf("perm_a[%d][%d][%d] = %f\n", 0, 1, 0, perm_a_debug[16]);
        printf("perm_a[%d][%d][%d] = %f\n", 0, 0, 1, perm_a_debug[1]);
        printf("perm_a[%d][%d][%d] = %f\n", 0, 1, 1, perm_a_debug[17]);
        printf("perm_a[%d][%d][%d] = %f\n", 1, 0, 0, perm_a_debug[4096 * 16]);
        printf("perm_a[%d][%d][%d] = %f\n", 1, 1, 0, perm_a_debug[4096 * 16 + 16]);
        printf("perm_a[%d][%d][%d] = %f\n", 1, 0, 1, perm_a_debug[4096 * 16 + 1]);
        printf("perm_a[%d][%d][%d] = %f\n", 1, 1, 1, perm_a_debug[4096 * 16 + 16 + 1]);
        free(perm_a_debug);
      }
      #endif

      if (n_samples == 1) {
        continue;
      }

      Conv1D(permute_a[g], conv0_w[g], conv0_b[g], conv0_a[g]);

      #ifdef DEBUG
      if (g == 0) {
        size_t OC = 1024;
        size_t os = 14;
        size_t C = 4096;
        size_t K = 3;
        size_t s = 16;

        size_t n_cuda = permute_a[g]->shape[0];
        size_t C_cuda = permute_a[g]->shape[1];
        size_t s_cuda = permute_a[g]->shape[2];
        size_t OC_cuda = conv0_w[g]->shape[0];
        size_t K_cuda = conv0_w[g]->shape[2];
        size_t os_cuda = conv0_a[g]->shape[2];
        printf("[Conv1D] n = %lu, C = %lu, s = %lu, OC = %lu, K = %lu, os = %lu\n", 
                n_cuda, C_cuda, s_cuda, OC_cuda, K_cuda, os_cuda);

        float *conv0_in_debug = (float *)malloc(sizeof(float) * C * s);
        float *conv0_w_debug = (float *)malloc(sizeof(float) * OC * C * K);
        float *conv0_b_debug = (float *)malloc(sizeof(float) * OC);
        float *conv0_a_debug = (float *)malloc(sizeof(float) * OC * os);
        float *conv0_a_ans = (float *)malloc(sizeof(float) * OC * os);
        CHECK_CUDA(cudaMemcpy(conv0_in_debug, permute_a[g]->buf, sizeof(float) * C * s, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(conv0_w_debug, conv0_w[g]->buf, sizeof(float) * OC * C * K, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(conv0_b_debug, conv0_b[g]->buf, sizeof(float) * OC, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(conv0_a_debug, conv0_a[g]->buf, sizeof(float) * OC * os, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < OC; i++) {
          for (size_t j = 0; j < os; j++) {
            float val = 0.f;
            for (size_t k = 0; k < C; k++) {
              for (size_t l = 0; l < K; l++) {
                val += conv0_in_debug[k * s + j + l] * 
                        conv0_w_debug[i * C * K + k * K + l];
              }
            }
            conv0_a_ans[i * os + j] = val + conv0_b_debug[i];
            if ((i == 0 || i == 1) && conv0_a_ans[i * os + j] != conv0_a_debug[i * os + j]) {
              printf("[i = %lu, j = %lu] ans: %f, cuda: %f, bias: %f\n", i, j, 
              conv0_a_ans[i * os + j], conv0_a_debug[i * os + j], conv0_b_debug[i]);
            }
          }
        }
        free(conv0_in_debug);
        free(conv0_w_debug);
        free(conv0_b_debug);
        free(conv0_a_debug);
        free(conv0_a_ans);
      }    
      #endif

      ReLU(conv0_a[g]);
      GetMax(conv0_a[g], pool0_a[g]);

      Conv1D(permute_a[g], conv1_w[g], conv1_b[g], conv1_a[g]);
      ReLU(conv1_a[g]);
      GetMax(conv1_a[g], pool1_a[g]);

      Conv1D(permute_a[g], conv2_w[g], conv2_b[g], conv2_a[g]);
      ReLU(conv2_a[g]);
      GetMax(conv2_a[g], pool2_a[g]);

      Conv1D(permute_a[g], conv3_w[g], conv3_b[g], conv3_a[g]);
      ReLU(conv3_a[g]);
      GetMax(conv3_a[g], pool3_a[g]);

      Concat(pool0_a[g], pool1_a[g], pool2_a[g], pool3_a[g], concat_a[g]);

      Linear(concat_a[g], linear0_w[g], linear0_b[g], linear0_a[g]);
      #ifdef DEBUG
      if (g == 0) {
        size_t N0 = concat_a[g]->shape[1]; // 4096
        size_t M0 = linear0_w[g]->shape[0]; // 2048
        printf("[Linear 0] N0 = %lu, M0 = %lu\n", N0, M0);

        float *concat_a_debug = (float *)malloc(4096 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(concat_a_debug, concat_a[g]->buf + 4096, 4096 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear0_w_debug = (float *)malloc(2048 * 4096 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear0_w_debug, linear0_w[g]->buf, 2048 * 4096 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear0_b_debug = (float *)malloc(2048 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear0_b_debug, linear0_b[g]->buf, 2048 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear0_a_debug = (float *)malloc(2048 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear0_a_debug, linear0_a[g]->buf + 2048, 2048 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear0_a_ans = (float *)malloc(2048 * sizeof(float));

        for (size_t i = 0; i < M0; i++) {
          float val = 0.f;
          for (size_t j = 0; j < N0; j++) {
            val += concat_a_debug[j] * linear0_w_debug[i * N0 + j];
          }
          linear0_a_ans[i] = val + linear0_b_debug[i];
          if ((i < 3 || i > 2044)) {
            printf("[i = %lu] ans = %f, cuda = %f\n", i, linear0_a_ans[i], linear0_a_debug[i]);
          }
        }

        free(concat_a_debug);
        free(linear0_w_debug);
        free(linear0_b_debug);
        free(linear0_a_debug);
        free(linear0_a_ans);
      }
      #endif

      ReLU(linear0_a[g]);

      Linear(linear0_a[g], linear1_w[g], linear1_b[g], linear1_a[g]);
      ReLU(linear1_a[g]);

      Linear(linear1_a[g], linear2_w[g], linear2_b[g], linear2_a[g]);
      ReLU(linear2_a[g]);

      Linear_narrow(linear2_a[g], linear3_w[g], linear3_b[g], linear3_a[g]);

      #ifdef DEBUG
      if (g == 0) {
        size_t N3 = linear2_a[g]->shape[1]; // 512
        size_t M3 = linear3_w[g]->shape[0]; // 2
        printf("[Linear narrow] N3 = %lu, M3 = %lu\n", N3, M3);

        float *linear2_a_debug = (float *)malloc(512 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear2_a_debug, linear2_a[g]->buf + 512, 512 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear3_w_debug = (float *)malloc(2 * 512 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear3_w_debug, linear3_w[g]->buf, 2 * 512 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear3_b_debug = (float *)malloc(2 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear3_b_debug, linear3_b[g]->buf, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear3_a_debug = (float *)malloc(2 * sizeof(float));
        CHECK_CUDA(cudaMemcpy(linear3_a_debug, linear3_a[g]->buf + 2, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        float *linear3_a_ans = (float *)malloc(2 * sizeof(float));

        for (size_t i = 0; i < M3; i++) {
          float val = 0.f;
          for (size_t j = 0; j < N3; j++) {
            val += linear2_a_debug[j] * linear3_w_debug[i * N3 + j];
          }
          linear3_a_ans[i] = val + linear3_b_debug[i];
          printf("[i = %lu] ans = %f, cuda = %f\n", i, linear3_a_ans[i], linear3_a_debug[i]);
        }

        free(linear2_a_debug);
        free(linear3_w_debug);
        free(linear3_b_debug);
        free(linear3_a_debug);
        free(linear3_a_ans);
      }
      #endif

      // outputs = [num_sentences * N_CLASSES]
      CHECK_CUDA(cudaMemcpy(&outputs[(NUM_SENTENCES / NUM_GPUS) * 2 * g], linear3_a[g]->buf,
                            NUM_SENTENCES / NUM_GPUS * 2 * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
  }
}