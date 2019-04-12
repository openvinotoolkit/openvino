"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
from tensorflow.core.framework import types_pb2 as tf_types  # pylint: disable=no-name-in-module

# mapping between TF data type and numpy data type and function to extract data from TF tensor
_tf_np_mapping = [('DT_BOOL', np.bool, lambda pb: pb.bool_val, lambda x: bool_cast(x)),
                  ('DT_INT8', np.int8, lambda pb: pb.int_val, lambda x: np.int8(x)),
                  ('DT_INT16', np.int16, lambda pb: pb.int_val, lambda x: np.int16(x)),
                  ('DT_INT32', np.int32, lambda pb: pb.int_val, lambda x: np.int32(x)),
                  ('DT_INT64', np.int64, lambda pb: pb.int64_val, lambda x: np.int64(x)),
                  ('DT_UINT8', np.uint8, lambda pb: pb.uint8_val, lambda x: np.uint8(x)),
                  ('DT_UINT16', np.uint16, lambda pb: pb.int_val, lambda x: np.uint16(x)),
                  ('DT_UINT32', np.uint32, lambda pb: pb.uint32_val, lambda x: np.uint32(x)),
                  ('DT_UINT64', np.uint64, lambda pb: pb.uint64_val, lambda x: np.uint64(x)),
                  ('DT_FLOAT', np.float32, lambda pb: pb.float_val, lambda x: np.float32(x)),
                  ('DT_DOUBLE', np.double, lambda pb: pb.double_val, lambda x: np.double(x)),
                  ('DT_STRING', np.str, lambda pb: pb.string_val, lambda x: np.str(x)),
                  ]

tf_data_type_decode = {getattr(tf_types, tf_dt): (np_type, func) for tf_dt, np_type, func, cast in _tf_np_mapping if
                       hasattr(tf_types, tf_dt)}

tf_data_type_cast = {np_type: cast for tf_dt, np_type, func, cast in _tf_np_mapping if hasattr(tf_types, tf_dt)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return np.bool(x)
