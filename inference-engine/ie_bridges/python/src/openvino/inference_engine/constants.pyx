# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cimport ie_api_impl_defs as C

import numpy as np
from enum import Enum

supported_precisions = ['FP32', 'FP64', 'FP16', 'I64', 'U64', 'I32', 'U32',
                        'I16', 'I4', 'I8', 'U16', 'U4', 'U8', 'BOOL', 'BIN', 'BF16']

known_plugins = ['CPU', 'GPU', 'FPGA', 'MYRIAD', 'HETERO', 'HDDL', 'MULTI']

layout_int_to_str_map = {0: 'ANY', 1: 'NCHW', 2: 'NHWC', 3: 'NCDHW', 4: 'NDHWC', 64: 'OIHW', 95: 'SCALAR', 96: 'C',
                         128: 'CHW', 192: 'HW', 193: 'NC', 194: 'CN', 200: 'BLOCKED'}

format_map = {'FP32' : np.float32,
              'FP64' : np.float64,
              'FP16' : np.float16,
              'I64'  : np.int64,
              'U64'  : np.uint64,
              'I32'  : np.int32,
              'U32'  : np.uint32,
              'I16'  : np.int16,
              'U16'  : np.uint16,
              'I4'   : np.int8,
              'I8'   : np.int8,
              'U4'   : np.int8,
              'U8'   : np.uint8,
              'BOOL' : np.uint8,
              'BIN'  : np.int8,
              'BF16' : np.float16,
              }

layout_str_to_enum = {'ANY': C.Layout.ANY,
                      'NHWC': C.Layout.NHWC,
                      'NCHW': C.Layout.NCHW,
                      'NCDHW': C.Layout.NCDHW,
                      'NDHWC': C.Layout.NDHWC,
                      'OIHW': C.Layout.OIHW,
                      'GOIHW': C.Layout.GOIHW,
                      'OIDHW': C.Layout.OIDHW,
                      'GOIDHW': C.Layout.GOIDHW,
                      'SCALAR': C.Layout.SCALAR,
                      'C': C.Layout.C,
                      'CHW': C.Layout.CHW,
                      'HW': C.Layout.HW,
                      'NC': C.Layout.NC,
                      'CN': C.Layout.CN,
                      'BLOCKED': C.Layout.BLOCKED
                      }


class MeanVariant(Enum):
    MEAN_IMAGE = 0
    MEAN_VALUE = 1
    NONE = 2


class ResizeAlgorithm(Enum):
    NO_RESIZE = 0
    RESIZE_BILINEAR = 1
    RESIZE_AREA = 2


class ColorFormat(Enum):
    RAW = 0
    RGB = 1
    BGR = 2
    RGBX = 3
    BGRX = 4
    NV12 = 5
    I420 = 6


cpdef enum StatusCode:
    OK = 0
    GENERAL_ERROR = -1
    NOT_IMPLEMENTED = -2
    NETWORK_NOT_LOADED = -3
    PARAMETER_MISMATCH = -4
    NOT_FOUND = -5
    OUT_OF_BOUNDS = -6
    UNEXPECTED = -7
    REQUEST_BUSY = -8
    RESULT_NOT_READY = -9
    NOT_ALLOCATED = -10
    INFER_NOT_STARTED = -11
    NETWORK_NOT_READ = -12


cpdef enum WaitMode:
    RESULT_READY = -1
    STATUS_ONLY = 0
