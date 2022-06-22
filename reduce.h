#pragma once
#include <cstdio>
#include <sys/time.h>
#define SZ 1024
#define N 1000
#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))

//static void HandleError( cudaError_t err, const char *file, int line );

void reduceSum(float* d_input, float* d_output, const int num);
void reduceSum_v2(float* d_input, float* d_output, const int num);
void cpu_data_to_gpu(float* input, float* d_input, const int num);
void gpu_data_initial(float* input, float* output, float** d_input, float** d_output, const int num);
void gpu_data_to_cpu(float* output, float* d_output);


float sum_cpu(float* input, const int num);
void initial_input(float* input, const int num);
