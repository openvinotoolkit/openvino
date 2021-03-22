# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.pyopenvino import TBlobFloat32
from openvino.pyopenvino import TBlobFloat64
from openvino.pyopenvino import TBlobInt8
from openvino.pyopenvino import TBlobInt16
from openvino.pyopenvino import TBlobInt32
from openvino.pyopenvino import TBlobInt64
from openvino.pyopenvino import TBlobUint8
from openvino.pyopenvino import TBlobUint16
from openvino.pyopenvino import TBlobUint32
from openvino.pyopenvino import TBlobUint64

from openvino.pyopenvino import TensorDesc

import numpy as np


# Patch for Blobs to dispatch types on Python side
class BlobPatch:
    def __new__(cls, tensor_desc, arr : np.ndarray = None):
        # TODO: create tensor_desc based on arr itself
        # if tenosr_desc is not given
        if arr is not None:
            arr = np.array(arr) # Keeping array as numpy array
            size_arr = np.prod(arr.shape)
            if arr is not None:
                if np.isfortran(arr):
                    arr = arr.ravel(order="F")
                else:
                    arr = arr.ravel(order="C")
        # Return TBlob depends on numpy array dtype
        # TODO: add dispatching based on tensor_desc precision value
        if tensor_desc is not None and arr is None:
            precision = tensor_desc.precision
            if precision == "FP32":
                return TBlobFloat32(tensor_desc)
            else:
                raise ValueError("not supported precision")
        elif tensor_desc is not None and arr is not None:
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
            raise AttributeError(f'Unsupported precision '
                                 f'{tensor_desc.precision} for Blob')
