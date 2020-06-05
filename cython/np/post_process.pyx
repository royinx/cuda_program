# cython: language_level=3, boundscheck=False 

# from numpy cimport ndarray as ar
from cython cimport view
# from cython.view cimport array
from cython.parallel cimport prange, parallel
cimport cython
import numpy as np 
import cv2
cimport numpy as cnp
from libcpp.vector cimport vector

# cnp.import_array()

cdef void argmax3D(float[:, :, ::1] array_1, vector[int] ignore_class, unsigned char[:, :] out) nogil:
# cdef unsigned char[:, :] argmax3D(float[:, :, ::1] array_1, vector[int] ignore_class, unsigned char[:, :] out) nogil:

    cdef Py_ssize_t h_max = array_1.shape[1]
    cdef Py_ssize_t w_max = array_1.shape[2]

    # view.array(shape=(360, 640), itemsize=sizeof(unsigned char), format="i") 

    # result = np.zeros((h_max, w_max), dtype=np.uint8)
    # result = view.array(shape=(h_max, w_max), itemsize=sizeof(unsigned char), format="i") 
    # cdef unsigned char[:, :] result = out

    # cdef int[:] ignore_class = view.array("i",[0,1,2,3,4,5,9,10,13,14,15,16,17,18]) 
    
    cdef int tmp

    cdef Py_ssize_t x, y, class_id
    # for x in prange(h_max, nogil=True):
    for x in range(h_max):
        for y in range(w_max):
            tmp = argmax1D(array_1[:,x,y])
            for class_id in ignore_class:
                if tmp == class_id:
                    tmp = 0
            out[x, y] = tmp
    # return out

cdef int argmax1D(float[:] l) nogil:
    cdef int index = -1
    cdef float max_val = -100.
    cdef Py_ssize_t i
    for i in range(len(l)):
        if l[i] > max_val:
            index, max_val = i, l[i]
    return index


cdef void argmax4D(float[:, :, :, ::1] batch_img, vector[int] ignore_class, unsigned char[:, :, :] out) nogil:

    cdef Py_ssize_t N = batch_img.shape[0]
    cdef Py_ssize_t h_max = batch_img.shape[2]
    cdef Py_ssize_t w_max = batch_img.shape[3]

    # view.array(shape=(360, 640), itemsize=sizeof(unsigned char), format="i") 

    # result = np.zeros((h_max, w_max), dtype=np.uint8)
    # result = view.array(shape=(h_max, w_max), itemsize=sizeof(unsigned char), format="i") 
    # cdef unsigned char[:, :, :] result = out

    # cdef int[:] ignore_class = view.array("i",[0,1,2,3,4,5,9,10,13,14,15,16,17,18]) 
    
    cdef int tmp

    cdef Py_ssize_t n, x, y, class_id
    # for x in prange(h_max, nogil=True):
    for n in prange(N):
        for x in range(h_max):
            for y in range(w_max):
                tmp = argmax1D(batch_img[n,:,x,y])
                for class_id in ignore_class:
                    if tmp == class_id:
                        tmp = 0
                out[n, x, y] = tmp
    # return out


cpdef list postprocess_cython(cnp.ndarray masks, list pads, list resized_imgs): # inp:batch

    # init size
    cdef int N = masks.shape[0]
    cdef int C = masks.shape[1]
    cdef int N_, C_
    cdef int net_h = 360, net_w = 640

    # init memorywise array
    # my_array = view.array(shape=(, 2), itemsize=sizeof(float), format="i")
    # cdef int[:, :] my_slice = my_array

    cdef cnp.ndarray[cnp.float32_t, ndim=4] mask_batch_ = np.ndarray((N, C, net_h, net_w), dtype=np.float32)
    cdef cnp.ndarray mask_argmax_batch = np.ndarray((N, net_h, net_w), dtype=np.uint8)
    cdef unsigned char[:, :, :] mask_argmax_batch_view = mask_argmax_batch
    cdef cnp.ndarray mask = np.empty((net_h, net_w), dtype=np.uint8)
    cdef unsigned char[:, :] my_out = mask
    cdef list output_list=[]
    cdef int top_pad, left_pad, h, w
    
    masks[masks < 0.5] = -100  # can be zero / -ve value
    for N_, img in enumerate(masks):
        for C_, class_ in enumerate(img):
            mask_batch_[N_][C_] = cv2.resize(class_, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    cdef float[:,:,:,::1] mask_batch = mask_batch_
    
    cdef vector[int] ignore_class = [0,1,2,3,4,5,9,10,13,14,15,16,17,18] 

    cdef Py_ssize_t i

    # Argmax
    # with nogil, parallel(num_threads=20):
    #     for i in prange(N):
    #         argmax3D(mask_batch[i], ignore_class, mask_argmax_batch_view[i]) # axis = 1 , If NCHW  (batch, 21, 12, 20)

    with nogil:
        argmax4D(mask_batch, ignore_class, mask_argmax_batch_view) # axis = 1 , If NCHW  (batch, 21, 12, 20)


    # del padding
    for idx, pad in enumerate(pads):
        top_pad, left_pad, = pad
        h, w, _ = resized_imgs[idx].shape
        if top_pad:
            output_list.append(mask_argmax_batch[idx][top_pad:top_pad + h, :]) # top pad
        else:
            output_list.append(mask_argmax_batch[idx][:, left_pad:left_pad + w]) # left pad
    return output_list
