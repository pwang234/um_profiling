#include <stdio.h>

#define CUCHK(call) {                                    \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
              __FILE__, __LINE__, cudaGetErrorString( err) );              \
      fflush(stderr); \
      exit(EXIT_FAILURE);                                                  \
    } }

__global__ void vec_add(float *a, float *b, int n)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x*gridDim.x) {
    a[i] = a[i] + b[i];
  }
}

int main(int argc, char *argv[])
{
  int n = 64*1024*1024;
  float *a, *b;
  CUCHK(cudaMallocManaged(&a, n*sizeof(float)));
  CUCHK(cudaMallocManaged(&b, n*sizeof(float)));

  for (int i = 0; i < n; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  for (int iter = 0; iter < 2; iter++) {
    vec_add<<<n/128, 128>>>(a, b, n);
    CUCHK(cudaDeviceSynchronize());
    for (int i = 0; i < n; i++) {
      a[i] += 1;
    }
  }

  CUCHK(cudaFree(a));
  CUCHK(cudaFree(b));
}
