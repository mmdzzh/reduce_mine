
#include<cstdio>
#include <sys/time.h>
#include<cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
// #include <immintrin.h>
// #include <avx2intrin.h>
#include "reduce.h"



static void HandleError( cudaError_t err, const char *file, int line ) {    
    if (err != cudaSuccess) {        
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );        
        exit( EXIT_FAILURE );    
    }
}


template <typename T>
__device__ T warpSum(T val){
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

template <typename T>
__device__ T blockSum(T val){
    __shared__ T s_data[SZ / 32];
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    
    val = warpSum(val);
    if(laneIdx == 0) s_data[warpIdx] = val;
    __syncthreads();
    val = (threadIdx.x < 32) ? s_data[laneIdx]:0;
    if(warpIdx == 0) return warpSum(val);
    
}

template <typename T>
__global__ void d_reduceSum(T* d_input,T* d_output, int n){
    T val = 0;
    for(int i = threadIdx.x;i < n;i += blockDim.x){
        val += d_input[i];
    }
    __syncthreads();
    val = blockSum(val);

    if(threadIdx.x == 0){
        
        d_output[0] = val;
    }
    
}

__global__ void printa(float* a, const int num){
    printf("%d: ", threadIdx.x);
    printf("%f\n", a[num -1 - threadIdx.x]);
}

template <typename T>
__global__ void d_reduceSum_v2(T* d_input, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T val = idx < n ? d_input[idx] : 0;
    // for(int i = threadIdx.x;i < n;i += blockDim.x){
    //     val += d_input[i];
    // }
    // __syncthreads();
    val = blockSum(val);

    if(threadIdx.x == 0){ 
        d_input[blockIdx.x] = val;
    }
    
}

void reduceSum(float* d_input, float* d_output, const int num){
    printa<<<1, 1>>>(d_output, 1);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    d_reduceSum<<<1, SZ>>>(d_input, d_output, num);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

}

void reduceSum_v2(float* d_input, float* d_output, const int num){
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int blockSize = (num + SZ - 1) / SZ;
    d_reduceSum_v2<<<blockSize, SZ>>>(d_input, num);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());
    d_reduceSum<<<1, SZ>>>(d_input, d_output, blockSize);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

    cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   float elapsedTime=0;
   cudaEventElapsedTime(&elapsedTime, start, stop);

   printf("gpu spent time: %f <ms>",  elapsedTime);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
    //int blockSize2 = (blockSize + SZ - 1) / SZ;
    //d_reduceSum_v2<<<blockSize2, SZ>>>(d_input, d_input, blockSize);
    //cudaDeviceSynchronize();
}

void gpu_data_initial(float* input, float* output, float** d_input, float** d_output, const int num){
    HANDLE_ERROR(cudaMalloc((void**)d_input, num * sizeof(float)));
    
    //HANDLE_ERROR(cudaMalloc((void**)&d_part, blockSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void**)d_output, 1 * sizeof(float)));
    
    HANDLE_ERROR(cudaGetLastError());
}

void cpu_data_to_gpu(float* input, float* d_input, const int num){
    HANDLE_ERROR(cudaMemcpy(d_input, input, num * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_data_to_cpu(float* output, float* d_output){
    HANDLE_ERROR(cudaMemcpy(output, d_output, 1 * sizeof(float), cudaMemcpyDeviceToHost));
}
