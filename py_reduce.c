#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>
#include <stdio.h>
#include <immintrin.h>
#include <nmmintrin.h>

float eps_sum_sse(float* tmp, int i, int j){
    float eps = 0.0;
    for(int k = 0;k < 4;k++){
        // float tmin = min(tmp[i], tmp[j]);
        // float tmax = max(tmp[i], tmp[j]);
        eps += (tmp[i] + tmp[j]) - tmp[i] - tmp[j];
    }
    return eps;
}

static PyObject* sum_zjh_cpu_sse(PyObject *self, PyObject *args){
    PyArrayObject* npObject;

    if(!PyArg_ParseTuple(args, "O", &npObject)){
        printf("failed\n");
        return NULL;
    }
    
    npy_intp i, j, iend, gap;
    npy_intp n = npObject->dimensions[0];
    

    // dimensions[1] = 1;
    npy_float32 *input = (npy_float32*)(npObject->data);
 
  
    npy_intp n2 = n / 32;

    n2--;
    n2 |= (n2>>1);
    n2 |= (n2>>2);
    n2 |= (n2>>4);
    n2 |= (n2>>8);
    n2 |= (n2>>16);
    n2++;

    npy_float32* tmp = (npy_float32*)malloc(n2 * sizeof(npy_float32));
    memset(tmp, 0, sizeof(npy_float32) * n2);

	__m128 sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps(), sum3 = _mm_setzero_ps(), sum4 = _mm_setzero_ps();
	__m128 load1, load2, load3, load4;

	npy_intp block_num = 8;
	npy_intp block_size = 16 * block_num;
    npy_intp k, remain;
	for (i = 0; i < n2; i += block_size) {
		j = i;
		for (; j < n; j += n2) {
			if(j + block_size < n){
				#pragma unroll
				for(k = 0;k < block_size;k += 16){
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

		if(j < n){
			remain = n - j;
			for(k = 0;k < remain;k++){
				tmp[i + k] += input[j + k];
			}
		}
		// cout << i << endl;
	}

    gap = n2 / 2;

	for (; gap > 8; gap /= 2) {
		for (i = 0; i < gap; i += 16) {
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
			// eps += eps_sum(tmp, i, i + gap);
			_mm_store_ps(tmp + i, sum1);
			_mm_store_ps(tmp + i + 4, sum2);
			_mm_store_ps(tmp + i + 8, sum3);
			_mm_store_ps(tmp + i + 12, sum4);
		}
        // eps += eps_sum(tmp, i, i + gap);
		// float t;
		// if (eps != 0) {
            
		// 	for (i = 0; i < gap; i++) {
		// 		t = tmp[i] - eps;
		// 		if (t - tmp[i] + eps == 0) {
		// 			tmp[i] = t;
		// 			break;
		// 		}
		// 	}
        //     if(i == gap){
        //         tmp[i-1] -= eps;
        //     }
		// 	eps = 0;
		// }
	}
	sum1 = _mm_add_ps(sum1, sum2);
	sum3 = _mm_add_ps(sum3, sum4);
	sum1 = _mm_add_ps(sum1, sum3);
	sum1 = _mm_hadd_ps(sum1, sum1);
	sum1 = _mm_hadd_ps(sum1, sum1);
	// //sum1 = _mm256_hadd_ps(sum1, sum1);

	_mm_store_ps(tmp, sum1);
    
    // PyArrayObject *out = (PyArrayObject*)malloc(sizeof(PyArrayObject));
    // printf("11111\n");
    // out->dimensions = (npy_intp*)malloc(sizeof(npy_intp));
    // printf("22222\n");
    // out->dimensions[0] = 1;
    // printf("dimension is %d\n", out->dimensions[0]);
    // out->data = (npy_float32*)malloc(sizeof(npy_float32));
    
    // printf("3333\n");
    // out->data[0] = tmp[0];

    
    // printf("out is %f\n", tmp[0]);
    // printf("out is %f\n", out->data[0]);
    npy_float32 out = tmp[0];
    // free(tmp);
   

    return Py_BuildValue("f", out);
    // PyObject* ret_test = (PyObject*)malloc(sizeof(PyObject));
    // return ret_test;//out;
}

static void sum_zjh_cpu_avx(char **args, npy_intp *dimensions, npy_intp *steps, void *data){
    npy_intp i, j, iend, gap;
    npy_intp n = dimensions[0];
    float *input = (float*)args[0];
    float *out = (float*)args[1];
    // npy_float32* input = (npy_float32*)in;
    // npy_float32* out = (npy_float32*)ou;
    npy_intp n2 = n / 32;

    n2--;
    n2 |= (n2>>1);
    n2 |= (n2>>2);
    n2 |= (n2>>4);
    n2 |= (n2>>8);
    n2 |= (n2>>16);
    n2++;

    float* tmp = (float*)malloc(n2 * sizeof(float));
    memset(tmp, 0, sizeof(float) * n2);

	__m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps(), sum3 = _mm256_setzero_ps(), sum4 = _mm256_setzero_ps();
	__m256 load1, load2, load3, load4;
	for (i = 0; i < n2; i += 32) {
		int j = i;
		for (; j < n; j += n2) {
			if(j + 32 < n){
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

		if(j < n){
			int remain = n - j;
			for(int k = 0;k < remain;k++){
				tmp[i + k] += input[j + k];
			}
		}
		// cout << i << endl;
	}

    
}

static npy_float32* sum_zjh_cpu(char **args, npy_intp *dimensions, npy_intp *steps, void *data){
    npy_intp i, j, iend, gap;
    npy_intp n = dimensions[0];
    float *input = (float*)args[0];
    float *out = (float*)args[1];
    // npy_float32* input = (npy_float32*)in;
    // npy_float32* out = (npy_float32*)ou;
    npy_intp n2 = n / 32;

    n2--;
    n2 |= (n2>>1);
    n2 |= (n2>>2);
    n2 |= (n2>>4);
    n2 |= (n2>>8);
    n2 |= (n2>>16);
    n2++;


    float* tmp = (float*)malloc(n2 * sizeof(float));
    memset(tmp, 0, sizeof(float) * n2);

    for(i = 0;i < n2;i++){
        for (j = 0; j < 32; j++) {
			iend = i + n2 * j;
			if (iend < n) {
				tmp[i] += *(input+iend);
			}
			else {
				//tmp[i] += 0;
				break;
			}
		}
    }   

    for(gap = n2/2;gap > 0; gap >>= 1){
        for (i = 0; i < gap; i++) {
            tmp[i] += tmp[i + gap];
        }
    }
    *out = tmp[0];
    // printf("output is %f, %f\n", tmp[0], out[0]);


    // // npy_float32 out = tmp[0];
    free(tmp);
    return out;
}

static PyMethodDef SumMethods[] = {
    {"sum", sum_zjh_cpu_sse, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};


PyUFuncGenericFunction funcs[1] = {&sum_zjh_cpu_sse};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_FLOAT32, NPY_FLOAT32};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "zjhnp",
    NULL,
    -1,
    SumMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyMODINIT_FUNC PyInit_zjhnp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    // import_array();
    // import_umath();
    // Py_DECREF(m);
    return m;
    // import_array();
    // import_umath();

    // sum = PyUFunc_FromFuncAndDataAndSignatureAndIdentity(funcs, data, types, 1, 1, 1,
    //                                 PyUFunc_One, "sum",
    //                                 "sum_docstring", 0, NULL, PyUFunc_One);

    // d = PyModule_GetDict(m);

    // PyDict_SetItemString(d, "sum", sum);
    // Py_DECREF(sum);

    // return m;
}
#else
PyMODINIT_FUNC initzjhnp(void)
{
    PyObject *m, *sum, *d;
    m = Py_InitModule("zjhnp", &SumMethods);
    if (!m) {
        return NULL;
    }
    return m;
    import_array();
    import_umath();

    sum = PyUFunc_FromFuncAndDataAndSignatureAndIdentity(funcs, data, types, 1, 1, 1,
                                    PyUFunc_One, "sum",
                                    "sum_docstring", 0, NULL);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "sum", sum);
    Py_DECREF(sum);


}
#endif
// #if defined(NPY_PY3K)
// PyMODINIT_FUNC PyInit_struct_ufunc_test(void)
// #else
// PyMODINIT_FUNC initstruct_ufunc_test(void)
// #endif
// {
//     npy_cfloat *m;
// #if defined(NPY_PY3K)
//     m = PyModule_Create(&moduledef);
// #else
//     m = Py_InitModule("struct_ufunc_test", StructUfuncTestMethods);
// #endif

// }
