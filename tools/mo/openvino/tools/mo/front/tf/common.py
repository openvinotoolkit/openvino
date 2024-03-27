# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from tensorflow.core.framework import types_pb2 as tf_types  # pylint: disable=no-name-in-module,import-error

# Suppress false positive pylint warning about function with too many arguments
# pylint: disable=E1121
# mapping between TF data type and numpy data type and function to extract data from TF tensor
_tf_np_mapping = [('DT_BOOL', bool, lambda pb: pb.bool_val, lambda x: bool_cast(x)),
                  ('DT_INT8', np.int8, lambda pb: pb.int_val, lambda x: np.int8(x)),
                  ('DT_INT16', np.int16, lambda pb: pb.int_val, lambda x: np.int16(x)),
                  ('DT_INT32', np.int32, lambda pb: pb.int_val, lambda x: np.int32(x)),
                  ('DT_INT64', np.int64, lambda pb: pb.int64_val, lambda x: np.int64(x)),
                  ('DT_UINT8', np.uint8, lambda pb: pb.uint8_val, lambda x: np.uint8(x)),
                  ('DT_UINT16', np.uint16, lambda pb: pb.int_val, lambda x: np.uint16(x)),
                  ('DT_UINT32', np.uint32, lambda pb: pb.uint32_val, lambda x: np.uint32(x)),
                  ('DT_UINT64', np.uint64, lambda pb: pb.uint64_val, lambda x: np.uint64(x)),
                  ('DT_HALF', np.float16, lambda pb: np.uint16(pb.half_val).view(np.float16), lambda x: np.float16(x)),
                  ('DT_FLOAT', np.float32, lambda pb: pb.float_val, lambda x: np.float32(x)),
                  ('DT_DOUBLE', np.double, lambda pb: pb.double_val, lambda x: np.double(x)),
                  ('DT_STRING', str, lambda pb: pb.string_val, lambda x: str(x)),
                  ]

tf_data_type_decode = {getattr(tf_types, tf_dt): (np_type, func) for tf_dt, np_type, func, _ in _tf_np_mapping if
                       hasattr(tf_types, tf_dt)}

tf_data_type_cast = {np_type: cast for tf_dt, np_type, _, cast in _tf_np_mapping if hasattr(tf_types, tf_dt)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return bool(x)
