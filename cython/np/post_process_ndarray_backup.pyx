# distutils: sources = Rectangle.cpp
# distutils: sources = argmax.cpp

# %%cython --force
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

# from numpy cimport ndarray as ar
from cpython.array cimport array
from cython.parallel import prange # parallel, threadid
cimport cython
import numpy as np 
import cv2
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()

ctypedef fused my_array:
    int
    long long

cdef argmax3D(float[:, :, ::1] array_1):

    cdef Py_ssize_t h_max = array_1.shape[1]
    cdef Py_ssize_t w_max = array_1.shape[2]

    result = np.zeros((h_max, w_max), dtype=np.uint8)
    cdef unsigned char[:, :] result_view = result

    cdef vector[int] ignore_class = [0,1,2,3,4,5,9,10,13,14,15,16,17,18]
    
    cdef int tmp

    cdef Py_ssize_t x, y, class_id
    for x in prange(h_max, nogil=True):
        for y in range(w_max):
            tmp = argmax1D(array_1[:,x,y])
            for class_id in ignore_class:
                if tmp == class_id:
                    tmp = 0
            result_view[x, y] = tmp
    return result

cdef int argmax1D(float[:] l) nogil:
    cdef int index = -1
    cdef float max_val = -100.
    cdef Py_ssize_t i
    for i in range(len(l)):
        if l[i] > max_val:
            index, max_val = i, l[i]
    return index


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef list postprocess_cython(cnp.ndarray masks, list pads, list resized_imgs): # inp:batch
    # threshold filter
    
    cdef int N = masks.shape[0]
    cdef int C = masks.shape[1]
    cdef int net_h = 360, net_w = 640

    cdef cnp.ndarray[cnp.float32_t, ndim=4] mask_batch_ = np.ndarray((N, C, net_h, net_w), dtype=np.float32)
    cdef cnp.ndarray mask_argmax_batch = np.ndarray((N, net_h, net_w), dtype=np.uint8)
    cdef cnp.ndarray mask = np.zeros((net_h, net_w), dtype=np.uint8)
    cdef list output_list=[]
    cdef int top_pad, left_pad, h, w
    
    masks[masks < 0.5] = -100  # can be zero / -ve value
    for N, img in enumerate(masks):
        for C, class_ in enumerate(img):
            mask_batch_[N][C] = cv2.resize(class_, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    cdef float[:,:,:,::1] mask_batch = mask_batch_
    
    # Argmax
    for i in range(N):
        mask_argmax_batch[i] = argmax3D(mask_batch[i]) # axis = 1 , If NCHW  (batch, 21, 12, 20)

    # del padding

    for idx, pad in enumerate(pads):
        top_pad, left_pad, = pad
        h, w, _ = resized_imgs[idx].shape
        if top_pad:
            output_list.append(mask_argmax_batch[idx][top_pad:top_pad + h, :]) # top pad
        else:
            output_list.append(mask_argmax_batch[idx][:, left_pad:left_pad + w]) # left pad
    return output_list





# %%
