# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from openvino.pyopenvino import TBlobFloat32
from openvino.pyopenvino import TBlobInt64
from openvino.pyopenvino import TBlobInt32
from openvino.pyopenvino import TBlobInt16
from openvino.pyopenvino import TBlobInt8
from openvino.pyopenvino import TBlobUint8


import numpy as np

# Patch for Blobs to dispatch types on Python side
class BlobPatch:
    def __new__(cls, tensor_desc, arr : np.ndarray = None):
        # TODO: create tensor_desc based on arr itself
        # if tenosr_desc is not given
        arr = np.array(arr) # Keeping array as numpy array
        size_arr = np.prod(arr.shape)
        if arr is not None:
            if np.isfortran(arr):
                arr = arr.ravel(order="F")
            else:
                arr = arr.ravel(order="C")
        # Return TBlob depends on numpy array dtype
        # TODO: add dispatching based on tensor_desc precision value
        if arr.dtype in [np.float32]:
            return TBlobFloat32(tensor_desc, arr, size_arr)
        # elif arr.dtype in [np.float64]:
        #     return TBlobFloat32(tensor_desc, arr.view(dtype=np.float32), size_arr)
        # elif arr.dtype in [np.int64]:
        #     return TBlobInt64(tensor_desc, arr, size)
        # elif arr.dtype in [np.int32]:
        #     return TBlobInt32(tensor_desc, arr, size)
        # elif arr.dtype in [np.int16]:
        #     return TBlobInt16(tensor_desc, arr, size)
        # elif arr.dtype in [np.int8]:
        #     return TBlobInt8(tensor_desc, arr, size)
        # elif arr.dtype in [np.uint8]:
        #     return TBlobUint8(tensor_desc, arr, size)
        else:
            # TODO: raise error
            return None
