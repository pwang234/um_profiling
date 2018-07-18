# um_profiling

# Dependency analysis
nvprof ./a.out


 GPU activities:   53.56%  1.48266s        10  148.27ms  141.54ms  166.49ms  vec_inc1(float*, int)

                   46.44%  1.28559s        10  128.56ms  122.03ms  148.82ms  vec_inc2(float*, int)

      API calls:   77.03%  1.48237s        10  148.24ms  141.50ms  166.46ms  cudaDeviceSynchronize
                   19.67%  378.53ms         2  189.26ms  70.097us  378.46ms  cudaMallocManaged
                    2.96%  57.047ms         2  28.524ms  27.357ms  29.690ms  cudaFree
                    0.16%  3.0426ms       384  7.9230us     114ns  320.91us  cuDeviceGetAttribute


Questions:

1. Which kernels are important? (Since they overlap, vec_inc2 not really important)
2. Is cudaDeviceSynchronize a problem? (no)

nvprof --dependency-analysis  ./a.out

Critical path(%)  Critical path  Waiting time  Name
          71.31%      4.751991s           0ns  <Other>

          22.27%      1.484002s           0ns  vec_inc1(float*, int)

           5.54%   369.095123ms           0ns  cudaMallocManaged_v6000

           0.81%    53.900904ms           0ns  cudaFree

           0.00%            0ns     1.483397s  cudaDeviceSynchronize

           0.00%            0ns           0ns  vec_inc2(float*, int)

1. Shows vec_inc2 not really important
2. Shows cudaDeviceSynchronize not really important