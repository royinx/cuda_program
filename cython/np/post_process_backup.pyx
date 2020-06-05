# distutils: sources = Rectangle.cpp
# distutils: sources = argmax.cpp

# %%cython --force
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

# from numpy cimport ndarray as ar
from cython.parallel import prange # parallel, threadid
cimport cython
import numpy as np 
import cv2
cimport numpy as np
np.import_array()

# from trtis_client import TrtisClient


# def test2(np.ndarray[np.float32_t, ndim=2] arr):    
#     cdef int i,j    
#     for i in xrange(arr.shape[0]):
#         for j in xrange(arr.shape[1]):
#             arr[i,j] += 1  

# cdef extern from "Rectangle.h" namespace "shapes":
#     cdef cppclass Rectangle:
#         Rectangle() except +
#         Rectangle(int, int, int, int) except +
#         int x0, y0, x1, y1
#         int getArea()
#         void getSize(int* width, int* height)
#         void move(int, int)
# cdef extern from "argmax.cpp":
    # pass

# cdef extern from "argmax.h":
    # cdef cppclass Rectangle:
    # cdef np.ndarray channelArgMax(np.ndarray)

ctypedef fused my_array:
    int
    long long

@cython.boundscheck(False)
@cython.wraparound(False)
cdef argmax3D(int[:, :, ::1] array_1):
# cdef argmax3D(int[:, :, ::1] array_1) nogil:

    cdef Py_ssize_t h_max = array_1.shape[1]
    cdef Py_ssize_t w_max = array_1.shape[2]

    # cdef int result[h_max][w_max]
    result = np.zeros((h_max, w_max), dtype=np.intc)
    cdef int[:, :] result_view = result

    cdef Py_ssize_t x, y

    # We use prange here.
    # for x in range(h_max):
    for x in prange(h_max, num_threads= 10, nogil=True):
        for y in prange(w_max):
            # print(array_1.shape)
            # print(array_1[:,x,y].shape)
            result_view[x, y] = argmax1D(array_1[:,x,y])

    return result


# cdef my_array argmax(my_array[:, ::1] array_1, my_array):
#     result = np.zeros((x_max, y_max), dtype=dtype)

#     argmax1D

#     cdef my_array[:, ::1] result_view = result
#     int index, max_val = -1, -1
#         for i in range(len(l)):

#         if l[i] > max_val:
#             index, max_val = i, l[i]
#     print(index)

cdef int argmax1D(int[:] l) nogil:
    cdef int index = -1
    cdef int max_val = -1
    cdef Py_ssize_t i
    for i in range(len(l)):
        if l[i] > max_val:
            index, max_val = i, l[i]
    return index

cpdef list postprocess_cython(np.ndarray masks, list pads, list resized_imgs): # inp:batch
    # threshold filter
    
    cdef int N = masks.shape[0]
    cdef int C = masks.shape[1]
    cdef int net_h = 360, net_w = 640
    # cdef cnp.ndarray mask_batch = cnp.PyArray_SimpleNew(1, (N,C,net_h,net_w), cnp.NPY_FLOAT32)

    # np.ndarray[np.float32, ndim=4]
    cdef np.ndarray mask_batch = np.ndarray((N, C, net_h, net_w), dtype=np.intc)
    # cdef np.ndarray mask_batch = np.zeros((N, C, net_h, net_w), dtype=np.float32)
    cdef np.ndarray mask_argmax_batch = np.ndarray((N, net_h, net_w), dtype=np.uint8)
    # cdef np.ndarray mask_argmax_batch = np.zeros((N, net_h, net_w), dtype=np.uint8)
    cdef np.ndarray mask_ = np.zeros((net_h, net_w), dtype=np.uint8)
    cdef list output_list=[]
    cdef int top_pad, left_pad, h, w
    
    masks[masks < 0.5] = -100  # can be zero / -ve value

    # resize to N,C,H,W (N,3,360,640) for 21 class
    # N, C, _, _ = masks.shape
    # mask_batch = np.zeros((N, C, net_h, net_w), dtype=np.float32)
    for N, img in enumerate(masks):
        for C, class_ in enumerate(img):
            mask_batch[N][C] = cv2.resize(class_, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    # Argmax
    for i in range(N):
    # for i in prange(N,num_threads = 10,nogil = True):
        # def foo(self, A bar):
        
        # mask_ = channelArgMax(mask_batch[i]) # axis = 1 , If NCHW  (batch, 21, 12, 20)
        mask_ = argmax3D(mask_batch[i]) # axis = 1 , If NCHW  (batch, 21, 12, 20)
        # mask_ = np.argmax(mask_batch[i], axis=0) # axis = 1 , If NCHW  (batch, 21, 12, 20)
        mask_ = np.isin(mask_, [0,1,2,3,4,5,9,10,13,14,15,16,17,18])
        mask_argmax_batch[i] = mask_

    # mask_batch = np.argmax(mask_batch, axis=1).astype(np.uint8) # axis = 1 , If NCHW  (batch, 21, 12, 20)
    # mask_batch[np.isin(mask_batch, [0,1,2,3,4,5,9,10,13,14,15,16,17,18] )] = 0




    # del padding

    for idx, pad in enumerate(pads):
        top_pad, left_pad, = pad
        h, w, _ = resized_imgs[idx].shape
        # top_pad, left_pad, h, w, _ = *pad, *resized_imgs[idx].shape
        if top_pad:
            output_list.append(mask_argmax_batch[idx][top_pad:top_pad + h, :]) # top pad
        else:
            output_list.append(mask_argmax_batch[idx][:, left_pad:left_pad + w]) # left pad
    return output_list





# %%
