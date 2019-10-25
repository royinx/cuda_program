#include <stdio.h>
#include <cuda_runtime.h>

bool InitCUDA()
{
    int count;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }

    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
}

// __global__ void VecAdd(float *A , float* B, float* C){
//     int i = threadIdx.x;
//     C[i] = A[i] + B[i];
// }
// int main(){
//     VecAdd<<<1,N>>>(A, B, C);
// }