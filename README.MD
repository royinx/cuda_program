## CUDA docker 

`nvidia/cuda:10.0-cudnn7-devel`

```
docker run -it \
            --rm \
            --runtime=nvidia \
            -v ~/Desktop/c++/:/cpp/ \
            -w /cpp/ \
            nvidia/cuda:10.0-cudnn7-devel bash
```