#include <stdio.h>
#include <cuda_runtime.h>

bool InitCUDA()
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    int device;
    for(device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        if(cudaGetDeviceProperties(&deviceProp, device) == cudaSuccess) {
            if(deviceProp.major >= 1) {
                break;
            }
        }
    }

    if(device == deviceCount) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(device);

    return true;
}

int main(){
    if(!InitCUDA()){
        printf("CUDA initialized. \n");
        return 0;
    }
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
            device, deviceProp.major, deviceProp.minor);
    }
    return 0;
}



// __global__ void VecAdd(float *A , float* B, float* C){
//     int i = threadIdx.x;
//     C[i] = A[i] + B[i];
// }
// int main(){
//     VecAdd<<<1,N>>>(A, B, C);
// }