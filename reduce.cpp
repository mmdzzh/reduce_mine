#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <immintrin.h>
//#include <avx2intrin.h>
#include "reduce.h"
#include <stack>
#include <nmmintrin.h>

//#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))



long long GetCurrentTime()
{
    struct timeval time;
 
    gettimeofday(&time,NULL);
    
    return (time.tv_sec * 1000000 + time.tv_usec);
}
 
double CalcTime_inusec(long long startusec, long long endusec)
{
    return (double)(endusec - startusec)/1000.0;

}


void initial_input(float* input, const int num){
    
    srand(time(NULL));
    for (int i = 0; i < num; i++){
        input[i] = (rand() % 10);// / (float)N;
    }
    
}


float sum_cpu(float* input, const int num){
    float out[8] = {0};
    int num8 = num / 8 * 8;
    int k = num % 8;
    for (int i = 0; i < num8; i+=8){
        out[0] += input[i];
        out[1] += input[i + 1];
        out[2] += input[i + 2];
        out[3] += input[i + 3];
        out[4] += input[i + 4];
        out[5] += input[i + 5];
        out[6] += input[i + 6];
        out[7] += input[i + 7];
    }
    
    for(;k > 0;k--){
        out[k - 1] += input[num - k];
    }
    for(int i = 4;i > 0;i >>= 1){
        for(int j = 0;j < i;j++){
            out[j] += out[j + i];
                // printf("out is %f, %f:%x, %x\n", out[j+i], out[j], *(long *)&out[j], *(long *)&out[j + i]);
            // float eps = out[j] + out[j + i] - out[j] - out[j + i];
            // printf("%f\n", eps);
            //     out[j] = out[j] + out[j + i] - eps;
            //float sum = aa + bb;
            //out[j] = sum;
        }
    }
    printf("sum is %f\n", out[0]);
    return out[0];
}


float reduce_cpu_sse(float* input, float* tmp, const int num, int num2) {
	__m128 sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps(), sum3 = _mm_setzero_ps(), sum4 = _mm_setzero_ps();
	__m128 load1, load2, load3, load4;

	// for(int i = 0;i < num2;i++){
	// 	int j = i;
	// 	for(;j < num;j += num2){
	// 		tmp[i] += input[j];
	// 	}

	// }
	int block_num = 8;
	int block_size = 16 * block_num;
	for (int i = 0; i < num2; i += block_size) {
		int j = i;
		for (; j < num; j += num2) {
			if(j + 16 < num){
				#pragma unroll
				for(int k = 0;k < block_size;k += 16){
					load1 = _mm_load_ps(input + j + k);
					load2 = _mm_load_ps(input + j + 4 + k);
					load3 = _mm_load_ps(input + j + 8 + k);
					load4 = _mm_load_ps(input + j + 12 + k);

					sum1 = _mm_add_ps(sum1, load1);
					sum2 = _mm_add_ps(sum2, load2);
					sum3 = _mm_add_ps(sum3, load3);
					sum4 = _mm_add_ps(sum4, load4);		
				}

			}
			else{
				break;
			}
		}

		_mm_store_ps(tmp + i, sum1);
		_mm_store_ps(tmp + i + 4, sum2);
		_mm_store_ps(tmp + i + 8, sum3);
		_mm_store_ps(tmp + i + 12, sum4);	

		sum1 = _mm_setzero_ps();
		sum2 = _mm_setzero_ps();
		sum3 = _mm_setzero_ps();
		sum4 = _mm_setzero_ps();

		if(j < num){
			int remain = num - j;
			for(int k = 0;k < remain;k++){
				tmp[i + k] += input[j + k];
			}
		}
		// cout << i << endl;
	}

	int gap = num2 / 2;

	for (; gap > 8; gap /= 2) {
		for (int i = 0; i < gap; i += 16) {
			sum1 = _mm_load_ps(tmp + i);
			sum2 = _mm_load_ps(tmp + i + 4);
			sum3 = _mm_load_ps(tmp + i + 8);
			sum4 = _mm_load_ps(tmp + i + 12);

			load1 = _mm_load_ps(tmp + i + gap);
			load2 = _mm_load_ps(tmp + i + 4 + gap);
			load3 = _mm_load_ps(tmp + i + 8 + gap);
			load4 = _mm_load_ps(tmp + i + 12 + gap);

			sum1 = _mm_add_ps(sum1, load1);
			sum2 = _mm_add_ps(sum2, load2);
			sum3 = _mm_add_ps(sum3, load3);
			sum4 = _mm_add_ps(sum4, load4);

			// load1 = _mm_load_ps(tmp + i + 2 * gap);
			// load2 = _mm_load_ps(tmp + i + 4 + 2 * gap);
			// load3 = _mm_load_ps(tmp + i + 8 + 2 * gap);
			// load4 = _mm_load_ps(tmp + i + 12 + 2 * gap);

			// sum1 = _mm_add_ps(sum1, load1);
			// sum2 = _mm_add_ps(sum2, load2);
			// sum3 = _mm_add_ps(sum3, load3);
			// sum4 = _mm_add_ps(sum4, load4);

			// load1 = _mm_load_ps(tmp + i + 3 * gap);
			// load2 = _mm_load_ps(tmp + i + 4 + 3 * gap);
			// load3 = _mm_load_ps(tmp + i + 8 + 3 * gap);
			// load4 = _mm_load_ps(tmp + i + 12 + 3 * gap);

			// sum1 = _mm_add_ps(sum1, load1);
			// sum2 = _mm_add_ps(sum2, load2);
			// sum3 = _mm_add_ps(sum3, load3);
			// sum4 = _mm_add_ps(sum4, load4);
			/*	eps += eps_sum(tmp, i, i + gap);*/
			_mm_store_ps(tmp + i, sum1);
			_mm_store_ps(tmp + i + 4, sum2);
			_mm_store_ps(tmp + i + 8, sum3);
			_mm_store_ps(tmp + i + 12, sum4);
		}
		// float t;
		// if (eps != 0) {
		// 	for (int i = 0; i < gap; i++) {
		// 		t = tmp[i] - eps;
		// 		if (t - tmp[i] + eps == 0) {
		// 			tmp[i] = t;
		// 			break;
		// 		}
		// 	}
		// 	eps = 0;
		// }
	}
	// for(int i = 1;i < gap;i++){
	// 	tmp[0] += tmp[i];
	// }


	// sum256 = _mm256_hadd_ps(sum256, sum256);
	// sum256 = _mm256_hadd_ps(sum256, sum256);
	sum1 = _mm_add_ps(sum1, sum2);
	sum3 = _mm_add_ps(sum3, sum4);
	sum1 = _mm_add_ps(sum1, sum3);
	sum1 = _mm_hadd_ps(sum1, sum1);
	sum1 = _mm_hadd_ps(sum1, sum1);
	// //sum1 = _mm256_hadd_ps(sum1, sum1);

	_mm_store_ps(tmp, sum1);
	return tmp[0];// +tmp[2];

	// return 0;
}


// int sum_cpu_int(int)

//add-->if a+b-a-b != 0-->

float sum_cpu_v2(float* input, int num, std::stack<float>& sf){
    //rate is max number / min number
    float sum = 0;
    float eps, t;
    
    // float sum = 0;
    for(int i = 0;i < num;i++){
        if(!sf.empty()){
            t = sf.top();
            sf.pop();
        }
        else{
            t = 0;
        }
        t += input[i];
        eps = sum + t - sum - t;
        if(eps == 0){
            sum += t;
        }
        else{
            sf.push(t);
        }


    }

    return sum;//out[0];
}

int pow2(int num){
    num--;
    num |= (num>>1);
    num |= (num>>2);
    num |= (num>>4);
    num |= (num>>8);
    num |= (num>>16);
    return num+1;
}

float reduce_cpu(float* input, float* tmp, const int num, int num2) {
	// int num2 = pow2(num);
	printf("num2 is %d\n", num2);
	// num2 >>= 1;
	for (int i = 0; i < num2; i++) {
		for (int j = 0; j < 32; j++) {
			int iend = i + num2 * j;
			if (iend < num) {
				tmp[i] += input[iend];
			}
			else {
				//tmp[i] += 0;
				break;
			}
		}

	}
	
	// float t;
	float tmin, tmax;
	for (int gap = num2 / 2; gap > 0; gap >>= 1) {
		for (int i = 0; i < gap; i++) {
			//tmin = min(tmp[i], tmp[i + gap]);
			//tmax = max(tmp[i], tmp[i + gap]);
			////eps = tmax + tmin - tmax - tmin;
			////tmp[i] = tmin - eps + tmax;
			////t = tmp[i] + tmp[i+gap];
			////eps += t - tmp[i] - tmp[i+gap];
			//t = tmin + tmax;
			//eps += t - tmax - tmin;
			//tmp[i] = t;


			tmp[i] += tmp[i + gap];
			/*if (eps == 0) {
				tmp[i] = t;
			}
			else {
				eps = t
			}*/

		}
		// if (eps != 0) {
		// 	float mineps = (float)INT_MAX;
		// 	int minid = 0;
		// 	int i = 0;
		// 	for (; i < gap; i++) {
		// 		t = tmp[i] - eps;
		// 		float eps2 = t - tmp[i] + eps;
		// 		if (abs(eps2) < mineps) {
		// 			minid = i;
		// 			mineps = abs(eps2);
		// 		}
		// 		//if (t - tmp[i] + eps == 0) {
		// 		//	tmp[i] = t;
		// 		//	break;
		// 		//}
		// 	}
		// 	tmp[minid] -= eps;
		// 	//if (i == gap) {
		// 	//	tmp[i - 1] = tmp[i - 1] - eps;
		// 	//}
		// 	eps = 0;
		// }
		//tmp[0] -= eps;
		//eps = 0;
	}

	return tmp[0];
}

float eps_sum(float* tmp, int i, int j){
    float eps = 0.0;
    for(int k = 0;k < 8;k++){
        // float tmin = min(tmp[i], tmp[j]);
        // float tmax = max(tmp[i], tmp[j]);
        eps += (tmp[i] + tmp[j]) - tmp[i] - tmp[j];
    }
    return eps;
}


float reduce_cpu_avx(float* input, float* tmp, const int num, int num2) {
	__m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps(), sum4 = _mm256_setzero_ps();
	__m256 load1, load2, load3, load4;
	const int Block = 32;
	const int group = 1024;
	// int num_group = num2 / group;
	
	int BlockGroupSize = group * Block;
	int BlockNum = num / BlockGroupSize;

	for (int i = 0; i < num2; i += 64) {
		int j = i;
		for (; j < num; j += num2) {
			if(j + 64 < num){
				load1 = _mm256_loadu_ps(input + j);
				load2 = _mm256_loadu_ps(input + j + 8);
				load3 = _mm256_loadu_ps(input + j + 16);
				load4 = _mm256_loadu_ps(input + j + 24);

				sum1 = _mm256_add_ps(sum1, load1);
				sum2 = _mm256_add_ps(sum2, load2);
				sum3 = _mm256_add_ps(sum3, load3);
				sum4 = _mm256_add_ps(sum4, load4);		

				load1 = _mm256_loadu_ps(input + j + 32);
				load2 = _mm256_loadu_ps(input + j + 40);
				load3 = _mm256_loadu_ps(input + j + 48);
				load4 = _mm256_loadu_ps(input + j + 56);

				sum1 = _mm256_add_ps(sum1, load1);
				sum2 = _mm256_add_ps(sum2, load2);
				sum3 = _mm256_add_ps(sum3, load3);
				sum4 = _mm256_add_ps(sum4, load4);	
			}
			else if(j + 32 < num){
				load1 = _mm256_loadu_ps(input + j);
				load2 = _mm256_loadu_ps(input + j + 8);
				load3 = _mm256_loadu_ps(input + j + 16);
				load4 = _mm256_loadu_ps(input + j + 24);

				sum1 = _mm256_add_ps(sum1, load1);
				sum2 = _mm256_add_ps(sum2, load2);
				sum3 = _mm256_add_ps(sum3, load3);
				sum4 = _mm256_add_ps(sum4, load4);						
			}
			else{
				break;
			}
		}

		_mm256_storeu_ps(tmp + i, sum1);
		_mm256_storeu_ps(tmp + i + 8, sum2);
		_mm256_storeu_ps(tmp + i + 16, sum3);
		_mm256_storeu_ps(tmp + i + 24, sum4);	

		sum1 = _mm256_setzero_ps();
		sum2 = _mm256_setzero_ps();
		sum3 = _mm256_setzero_ps();
		sum4 = _mm256_setzero_ps();

		if(j < num){
			int remain = num - j;
			for(int k = 0;k < remain;k++){
				tmp[i + k] += input[j + k];
			}
		}
		// cout << i << endl;
	}



	for (int gap = num2 / 2; gap > 16; gap /= 2) {

		for (int i = 0; i < gap; i += 32) {
			sum1 = _mm256_loadu_ps(tmp + i);
			sum2 = _mm256_loadu_ps(tmp + i + 8);
			sum3 = _mm256_loadu_ps(tmp + i + 16);
			sum4 = _mm256_loadu_ps(tmp + i + 24);

			load1 = _mm256_loadu_ps(tmp + i + gap);
			load2 = _mm256_loadu_ps(tmp + i + 8 + gap);
			load3 = _mm256_loadu_ps(tmp + i + 16 + gap);
			load4 = _mm256_loadu_ps(tmp + i + 24 + gap);

			// load1 = _mm256_loadu_ps(tmp + i + gap);
			// load2 = _mm256_loadu_ps(tmp + i + 8 + gap);
			// load3 = _mm256_loadu_ps(tmp + i + 16 + gap);
			// load4 = _mm256_loadu_ps(tmp + i + 24 + gap);

			sum1 = _mm256_add_ps(sum1, load1);
			sum2 = _mm256_add_ps(sum2, load2);
			sum3 = _mm256_add_ps(sum3, load3);
			sum4 = _mm256_add_ps(sum4, load4);


			/*	eps += eps_sum(tmp, i, i + gap);*/
			_mm256_storeu_ps(tmp + i, sum1);
			_mm256_storeu_ps(tmp + i + 8, sum2);
			_mm256_storeu_ps(tmp + i + 16, sum3);
			_mm256_storeu_ps(tmp + i + 24, sum4);

		}

	}


	// sum256 = _mm256_hadd_ps(sum256, sum256);
	// sum256 = _mm256_hadd_ps(sum256, sum256);
	sum1 = _mm256_add_ps(sum1, sum2);
	sum3 = _mm256_add_ps(sum3, sum4);
	sum1 = _mm256_add_ps(sum1, sum3);
	sum1 = _mm256_hadd_ps(sum1, sum1);
	sum1 = _mm256_hadd_ps(sum1, sum1);
	// sum1 = _mm256_hadd_ps(sum1, sum1);

	_mm256_storeu_ps(tmp, sum1);
	return tmp[0] + tmp[4];// + tmp[4] + tmp[5];


	// float output;
	// _mm256_storeuu_ps(&output, sum256);
	//return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
}


int main(){
    const int num = 3e8;
    float* input = new float[num];

    float output;
    initial_input(input, num);
    float elapsedTimeC;
    long long cstart, cstop;

	// long long int* i_input = new long long int[num];
	// long long int isum = 0;
    int num2 = pow2(num);
    num2 /= 32;
	
	cstart = GetCurrentTime();
    float* tmp = new float[num2];
	cstop = GetCurrentTime();
    elapsedTimeC = CalcTime_inusec(cstart, cstop);
    printf("cpu spent time: %f <ms>\n", elapsedTimeC);

    cstart = GetCurrentTime();
    // for(int i = 0;i < num;i++){
    //     out[i] = input[i];
    // }
    // float c_output = sum_cpu_v2(input, num);
    float c_output = reduce_cpu(input, tmp, num, num2);
    cstop = GetCurrentTime();
    elapsedTimeC = CalcTime_inusec(cstart, cstop);
    printf("cpu spent time: %f <ms>, result is %f\n", elapsedTimeC, c_output);

    memset(tmp, 0, num2 * sizeof(float));
    // float aa = (3e8)+8;
    // printf("%f : %x\n", aa, *(long *)&aa);


    cstart = GetCurrentTime();
    // for(int i = 0;i < num;i++){
    //     out[i] = input[i];
    // }
    // float c_output = sum_cpu_v2(input, num);
 	c_output = reduce_cpu_sse(input, tmp, num, num2);
    cstop = GetCurrentTime();
    elapsedTimeC = CalcTime_inusec(cstart, cstop);
    // float aa = (3e8)+8;
    // printf("%f : %x\n", aa, *(long *)&aa);

    printf("sse cpu spent time: %f <ms>, result is %f\n", elapsedTimeC, c_output);

    memset(tmp, 0, num2 * sizeof(float));
    cstart = GetCurrentTime();
    // for(int i = 0;i < num;i++){
    //     out[i] = input[i];
    // }
    // float c_output = sum_cpu_v2(input, num);
    c_output = reduce_cpu_avx(input, tmp, num, num2);
    cstop = GetCurrentTime();
    elapsedTimeC = CalcTime_inusec(cstart, cstop);
    // float aa = (3e8)+8;
    // printf("%f : %x\n", aa, *(long *)&aa);

    printf("avx cpu spent time: %f <ms>, result is %f\n", elapsedTimeC, c_output);


    float* d_input, *d_output;//, *d_part;
    int blockSize = (num + SZ - 1) / SZ;

    gpu_data_initial(input, &output, &d_input, &d_output, num);
    cpu_data_to_gpu(input, d_input, num);
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);
//
//     reduceSum(d_input, d_output, num);
     reduceSum_v2(d_input, d_output, num);
    
//    
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     float elapsedTime=0;
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//
//     printf("gpu spent time: %f <ms>",  elapsedTime);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

    gpu_data_to_cpu(&output, d_output);
    
    if(abs((output - c_output) / output) >= 1e-3){
        printf("d_output and output is %f, %f\n", output, c_output);
    }
    else{
        printf("success\n");
        printf("d_output and output is %f, %f\n", output, c_output);
    }
    return 0;
}

