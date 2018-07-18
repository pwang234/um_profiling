#include <stdio.h>
#include "nvToolsExt.h"

#define CUCHK(call) {                                    \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
              __FILE__, __LINE__, cudaGetErrorString( err) );              \
      fflush(stderr); \
      exit(EXIT_FAILURE);                                                  \
    } }

#define NSTREAM 2

__global__ void vec_inc1(float *a, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
    a[i] += 1;
  }
}

__global__ void vec_inc2(float *a, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
    a[i] += 2;
  }
}

int main(int argc, char *argv[])
{
  int n = 64*1024*1024;
  float *a, *b;
  CUCHK(cudaMallocManaged(&a, n*sizeof(float)));
  CUCHK(cudaMallocManaged(&b, n*sizeof(float)));
  cudaStream_t stream[NSTREAM];
  for (int i = 0; i < NSTREAM; i++) {
    CUCHK(cudaStreamCreate(&stream[i]));
  }

  // cudaProfilerStart();
  for (int i = 0; i < n; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  for (int i = 0; i < 10; i++) {
    vec_inc1<<<256, 128, 0, stream[0]>>>(a, n);
    vec_inc2<<<512, 128, 0, stream[1]>>>(b, n);
    CUCHK(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) {
      a[i] += 1;
      b[i] += 1;
    }
  }

  // cudaProfilerStop();

  for (int i = 0; i < NSTREAM; i++) {
    CUCHK(cudaStreamDestroy(stream[i]));
  }
  CUCHK(cudaFree(a));
  CUCHK(cudaFree(b));
}
