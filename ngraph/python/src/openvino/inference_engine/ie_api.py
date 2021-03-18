# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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

precision_map = {'FP32': np.float32,
                 'FP64': np.float64,
                 # 'FP16': np.int16,
                 # 'BF16': np.int16,
                 'I16': np.int16,
                 'I8': np.int8,
                 'BIN': np.int8,
                 'I32': np.int32,
                 'I64': np.int64,
                 'U8': np.uint8,
                 'BOOL': np.uint8,
                 'U16': np.uint16,
                 # 'Q78': np.uint16,
                 'U32': np.uint32,
                 'U64': np.uint64}


class BlobWrapper:
    """Dispatch Blob types on Python side."""

    def __new__(cls, tensor_desc: TensorDesc, arr: np.ndarray = None):
        """Create new Blob."""
        arr_size = 0
        tensor_desc_size = np.prod(tensor_desc.dims)

        if arr is not None:
            arr = np.array(arr)
            arr_size = int(np.prod(arr.shape))  # int conversion for SCALAR
            if np.isfortran(arr):
                arr = arr.ravel(order='F')
            else:
                arr = arr.ravel(order='C')
            if arr_size != tensor_desc_size:
                raise AttributeError(f'Number of elements in provided numpy array '
                                     f'{arr_size} and required by TensorDesc '
                                     f'{tensor_desc_size} are not equal')
            if arr.dtype != precision_map[tensor_desc.precision]:
                raise ValueError(f'Data type {arr.dtype} of provided numpy array '
                                 f'does not match TensorDesc precision '
                                 f'{tensor_desc.precision}')
        else:
            arr = np.empty(0, dtype=precision_map[tensor_desc.precision])

        # Dispatching based on tensor_desc precision value
        if tensor_desc.precision in ['FP32']:
            return TBlobFloat32(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['FP64']:
            return TBlobFloat64(tensor_desc, arr, arr_size)
        # elif tensor_desc.precision in ['FP16', 'BF16']:
        #     return TBlobInt16(tensor_desc,
        #                       arr.view(dtype=np.int16),
        #                       arr_size)
        elif tensor_desc.precision in ['I16']:
            return TBlobInt16(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['I8', 'BIN']:
            return TBlobInt8(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['I32']:
            return TBlobInt32(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['I64']:
            return TBlobInt64(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['U8', 'BOOL']:
            return TBlobUint8(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['U16']:  # , 'Q78'
            return TBlobUint16(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['U32']:
            return TBlobUint32(tensor_desc, arr, arr_size)
        elif tensor_desc.precision in ['U64']:
            return TBlobUint64(tensor_desc, arr, arr_size)
        else:
            raise AttributeError(f'Unsupported precision '
                                 f'{tensor_desc.precision} for Blob')
