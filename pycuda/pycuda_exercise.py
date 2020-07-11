from __future__ import print_function

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda  import gpuarray
host_data = np.array([1,2,3,4,5,6],dtype=np.float32)
device_data = gpuarray.to_gpu(host_data)
device_data_x2 = 2 * device_data
host_data_x2 = device_data_x2.get()
print(host_data_x2)