# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

np_map_cast = {bool: lambda x: bool_cast(x),
               np.int8: lambda x: np.int8(x),
               np.int16: lambda x: np.int16(x),
               np.int32: lambda x: np.int32(x),
               np.int64: lambda x: np.int64(x),
               np.uint8: lambda x: np.uint8(x),
               np.uint16: lambda x: np.uint16(x),
               np.uint32: lambda x: np.uint32(x),
               np.uint64: lambda x: np.uint64(x),
               np.float16: lambda x: np.float16(x),
               np.float32: lambda x: np.float32(x),
               np.double: lambda x: np.double(x),
               str: lambda x: str(x)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return bool(x)
