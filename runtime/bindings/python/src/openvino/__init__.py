# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openvino module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

__path__ = __import__('pkgutil').extend_path(__path__, __name__)  # type: ignore  # mypy issue #1422

try:
    __version__ = get_distribution("openvino-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

from openvino import inference_engine

from openvino.ie_api import BlobWrapper
from openvino.ie_api import infer
from openvino.ie_api import async_infer
from openvino.ie_api import get_result
from openvino.ie_api import blob_from_file

from openvino.impl import Dimension
from openvino.impl import Function
from openvino.impl import Node
from openvino.impl import PartialShape
from openvino.frontend import FrontEnd
from openvino.frontend import FrontEndManager
from openvino.frontend import GeneralFailure
from openvino.frontend import NotImplementedFailure
from openvino.frontend import InitializationFailure
from openvino.frontend import InputModel
from openvino.frontend import OpConversionFailure
from openvino.frontend import OpValidationFailure
from openvino.frontend import Place
from openvino.opset8 import absolute
from openvino.opset8 import absolute as abs
from openvino.opset8 import acos
from openvino.opset8 import acosh
from openvino.opset8 import adaptive_avg_pool
from openvino.opset8 import adaptive_max_pool
from openvino.opset8 import add
from openvino.opset8 import asin
from openvino.opset8 import asinh
from openvino.opset8 import assign
from openvino.opset8 import atan
from openvino.opset8 import atanh
from openvino.opset8 import avg_pool
from openvino.opset8 import batch_norm_inference
from openvino.opset8 import batch_to_space
from openvino.opset8 import binary_convolution
from openvino.opset8 import broadcast
from openvino.opset8 import bucketize
from openvino.opset8 import ceiling
from openvino.opset8 import ceiling as ceil
from openvino.opset8 import clamp
from openvino.opset8 import concat
from openvino.opset8 import constant
from openvino.opset8 import convert
from openvino.opset8 import convert_like
from openvino.opset8 import convolution
from openvino.opset8 import convolution_backprop_data
from openvino.opset8 import cos
from openvino.opset8 import cosh
from openvino.opset8 import ctc_greedy_decoder
from openvino.opset8 import ctc_greedy_decoder_seq_len
from openvino.opset8 import ctc_loss
from openvino.opset8 import cum_sum
from openvino.opset8 import cum_sum as cumsum
from openvino.opset8 import deformable_convolution
from openvino.opset8 import deformable_psroi_pooling
from openvino.opset8 import depth_to_space
from openvino.opset8 import detection_output
from openvino.opset8 import dft
from openvino.opset8 import divide
from openvino.opset8 import einsum
from openvino.opset8 import elu
from openvino.opset8 import embedding_bag_offsets_sum
from openvino.opset8 import embedding_bag_packed_sum
from openvino.opset8 import embedding_segments_sum
from openvino.opset8 import extract_image_patches
from openvino.opset8 import equal
from openvino.opset8 import erf
from openvino.opset8 import exp
from openvino.opset8 import fake_quantize
from openvino.opset8 import floor
from openvino.opset8 import floor_mod
from openvino.opset8 import gather
from openvino.opset8 import gather_elements
from openvino.opset8 import gather_nd
from openvino.opset8 import gather_tree
from openvino.opset8 import gelu
from openvino.opset8 import greater
from openvino.opset8 import greater_equal
from openvino.opset8 import grn
from openvino.opset8 import group_convolution
from openvino.opset8 import group_convolution_backprop_data
from openvino.opset8 import gru_cell
from openvino.opset8 import gru_sequence
from openvino.opset8 import hard_sigmoid
from openvino.opset8 import hsigmoid
from openvino.opset8 import hswish
from openvino.opset8 import idft
from openvino.opset8 import interpolate
from openvino.opset8 import less
from openvino.opset8 import less_equal
from openvino.opset8 import log
from openvino.opset8 import logical_and
from openvino.opset8 import logical_not
from openvino.opset8 import logical_or
from openvino.opset8 import logical_xor
from openvino.opset8 import log_softmax
from openvino.opset8 import loop
from openvino.opset8 import lrn
from openvino.opset8 import lstm_cell
from openvino.opset8 import lstm_sequence
from openvino.opset8 import matmul
from openvino.opset8 import matrix_nms
from openvino.opset8 import max_pool
from openvino.opset8 import maximum
from openvino.opset8 import minimum
from openvino.opset8 import mish
from openvino.opset8 import mod
from openvino.opset8 import multiclass_nms
from openvino.opset8 import multiply
from openvino.opset8 import mvn
from openvino.opset8 import negative
from openvino.opset8 import non_max_suppression
from openvino.opset8 import non_zero
from openvino.opset8 import normalize_l2
from openvino.opset8 import not_equal
from openvino.opset8 import one_hot
from openvino.opset8 import pad
from openvino.opset8 import parameter
from openvino.opset8 import power
from openvino.opset8 import prelu
from openvino.opset8 import prior_box
from openvino.opset8 import prior_box_clustered
from openvino.opset8 import psroi_pooling
from openvino.opset8 import proposal
from openvino.opset8 import range
from openvino.opset8 import read_value
from openvino.opset8 import reduce_l1
from openvino.opset8 import reduce_l2
from openvino.opset8 import reduce_logical_and
from openvino.opset8 import reduce_logical_or
from openvino.opset8 import reduce_max
from openvino.opset8 import reduce_mean
from openvino.opset8 import reduce_min
from openvino.opset8 import reduce_prod
from openvino.opset8 import reduce_sum
from openvino.opset8 import region_yolo
from openvino.opset8 import reorg_yolo
from openvino.opset8 import relu
from openvino.opset8 import reshape
from openvino.opset8 import result
from openvino.opset8 import reverse_sequence
from openvino.opset8 import rnn_cell
from openvino.opset8 import rnn_sequence
from openvino.opset8 import roi_align
from openvino.opset8 import roi_pooling
from openvino.opset8 import roll
from openvino.opset8 import round
from openvino.opset8 import scatter_elements_update
from openvino.opset8 import scatter_update
from openvino.opset8 import select
from openvino.opset8 import selu
from openvino.opset8 import shape_of
from openvino.opset8 import shuffle_channels
from openvino.opset8 import sigmoid
from openvino.opset8 import sign
from openvino.opset8 import sin
from openvino.opset8 import sinh
from openvino.opset8 import softmax
from openvino.opset8 import softplus
from openvino.opset8 import space_to_batch
from openvino.opset8 import space_to_depth
from openvino.opset8 import split
from openvino.opset8 import sqrt
from openvino.opset8 import squared_difference
from openvino.opset8 import squeeze
from openvino.opset8 import strided_slice
from openvino.opset8 import subtract
from openvino.opset8 import swish
from openvino.opset8 import tan
from openvino.opset8 import tanh
from openvino.opset8 import tensor_iterator
from openvino.opset8 import tile
from openvino.opset8 import topk
from openvino.opset8 import transpose
from openvino.opset8 import unsqueeze
from openvino.opset8 import variadic_split

from openvino.pyopenvino import Core
from openvino.pyopenvino import IENetwork
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Version
from openvino.pyopenvino import Parameter
from openvino.pyopenvino import InputInfoPtr
from openvino.pyopenvino import InputInfoCPtr
from openvino.pyopenvino import DataPtr
from openvino.pyopenvino import TensorDesc
from openvino.pyopenvino import get_version
from openvino.pyopenvino import StatusCode
from openvino.pyopenvino import InferQueue
from openvino.pyopenvino import InferRequest  # TODO: move to ie_api?
from openvino.pyopenvino import Blob
from openvino.pyopenvino import PreProcessInfo
from openvino.pyopenvino import MeanVariant
from openvino.pyopenvino import ResizeAlgorithm
from openvino.pyopenvino import ColorFormat
from openvino.pyopenvino import PreProcessChannel


# Extend Node class to support binary operators
Node.__add__ = add
Node.__sub__ = subtract
Node.__mul__ = multiply
Node.__div__ = divide
Node.__truediv__ = divide
Node.__radd__ = lambda left, right: add(right, left)
Node.__rsub__ = lambda left, right: subtract(right, left)
Node.__rmul__ = lambda left, right: multiply(right, left)
Node.__rdiv__ = lambda left, right: divide(right, left)
Node.__rtruediv__ = lambda left, right: divide(right, left)
Node.__eq__ = equal
Node.__ne__ = not_equal
Node.__lt__ = less
Node.__le__ = less_equal
Node.__gt__ = greater
Node.__ge__ = greater_equal

# Patching for Blob class
# flake8: noqa: F811
Blob = BlobWrapper
# Patching ExecutableNetwork
ExecutableNetwork.infer = infer
# Patching InferRequest
InferRequest.infer = infer
InferRequest.async_infer = async_infer
InferRequest.get_result = get_result
# Patching InferQueue
InferQueue.async_infer = async_infer
