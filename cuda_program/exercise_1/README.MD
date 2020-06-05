# CUDA 
docker container:
``` bash
docker run -it \
            --rm \
            --runtime=nvidia \
            -v ~/Desktop/c++/:/cpp/ \
            -w /cpp/ \
            nvidia/cuda:10.0-cudnn7-devel bash
```
compiler:`nvcc`

```bash
nvcc src/.cu
# remark : nvcc check
nvcc -V

```