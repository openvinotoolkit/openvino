# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ngraph module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("ngraph-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"


from ngraph.impl import Dimension
from ngraph.impl import Function
from ngraph.impl import Node
from ngraph.impl import PartialShape
from ngraph.frontend import FrontEnd
from ngraph.frontend import FrontEndManager
from ngraph.frontend import GeneralFailure
from ngraph.frontend import NotImplementedFailure
from ngraph.frontend import InitializationFailure
from ngraph.frontend import InputModel
from ngraph.frontend import OpConversionFailure
from ngraph.frontend import OpValidationFailure
from ngraph.frontend import Place
from ngraph.helpers import function_from_cnn
from ngraph.helpers import function_to_cnn
from ngraph.opset8 import absolute
from ngraph.opset8 import absolute as abs
from ngraph.opset8 import acos
from ngraph.opset8 import acosh
from ngraph.opset8 import adaptive_avg_pool
from ngraph.opset8 import adaptive_max_pool
from ngraph.opset8 import add
from ngraph.opset8 import asin
from ngraph.opset8 import asinh
from ngraph.opset8 import assign
from ngraph.opset8 import atan
from ngraph.opset8 import atanh
from ngraph.opset8 import avg_pool
from ngraph.opset8 import batch_norm_inference
from ngraph.opset8 import batch_to_space
from ngraph.opset8 import binary_convolution
from ngraph.opset8 import broadcast
from ngraph.opset8 import bucketize
from ngraph.opset8 import ceiling
from ngraph.opset8 import ceiling as ceil
from ngraph.opset8 import clamp
from ngraph.opset8 import concat
from ngraph.opset8 import constant
from ngraph.opset8 import convert
from ngraph.opset8 import convert_like
from ngraph.opset8 import convolution
from ngraph.opset8 import convolution_backprop_data
from ngraph.opset8 import cos
from ngraph.opset8 import cosh
from ngraph.opset8 import ctc_greedy_decoder
from ngraph.opset8 import ctc_greedy_decoder_seq_len
from ngraph.opset8 import ctc_loss
from ngraph.opset8 import cum_sum
from ngraph.opset8 import cum_sum as cumsum
from ngraph.opset8 import deformable_convolution
from ngraph.opset8 import deformable_psroi_pooling
from ngraph.opset8 import depth_to_space
from ngraph.opset8 import detection_output
from ngraph.opset8 import dft
from ngraph.opset8 import divide
from ngraph.opset8 import einsum
from ngraph.opset8 import elu
from ngraph.opset8 import embedding_bag_offsets_sum
from ngraph.opset8 import embedding_bag_packed_sum
from ngraph.opset8 import embedding_segments_sum
from ngraph.opset8 import extract_image_patches
from ngraph.opset8 import equal
from ngraph.opset8 import erf
from ngraph.opset8 import exp
from ngraph.opset8 import fake_quantize
from ngraph.opset8 import floor
from ngraph.opset8 import floor_mod
from ngraph.opset8 import gather
from ngraph.opset8 import gather_elements
from ngraph.opset8 import gather_nd
from ngraph.opset8 import gather_tree
from ngraph.opset8 import gelu
from ngraph.opset8 import greater
from ngraph.opset8 import greater_equal
from ngraph.opset8 import grn
from ngraph.opset8 import group_convolution
from ngraph.opset8 import group_convolution_backprop_data
from ngraph.opset8 import gru_cell
from ngraph.opset8 import gru_sequence
from ngraph.opset8 import hard_sigmoid
from ngraph.opset8 import hsigmoid
from ngraph.opset8 import hswish
from ngraph.opset8 import idft
from ngraph.opset8 import interpolate
from ngraph.opset8 import less
from ngraph.opset8 import less_equal
from ngraph.opset8 import log
from ngraph.opset8 import logical_and
from ngraph.opset8 import logical_not
from ngraph.opset8 import logical_or
from ngraph.opset8 import logical_xor
from ngraph.opset8 import log_softmax
from ngraph.opset8 import loop
from ngraph.opset8 import lrn
from ngraph.opset8 import lstm_cell
from ngraph.opset8 import lstm_sequence
from ngraph.opset8 import matmul
from ngraph.opset8 import matrix_nms
from ngraph.opset8 import max_pool
from ngraph.opset8 import maximum
from ngraph.opset8 import minimum
from ngraph.opset8 import mish
from ngraph.opset8 import mod
from ngraph.opset8 import multiclass_nms
from ngraph.opset8 import multiply
from ngraph.opset8 import mvn
from ngraph.opset8 import negative
from ngraph.opset8 import non_max_suppression
from ngraph.opset8 import non_zero
from ngraph.opset8 import normalize_l2
from ngraph.opset8 import not_equal
from ngraph.opset8 import one_hot
from ngraph.opset8 import pad
from ngraph.opset8 import parameter
from ngraph.opset8 import power
from ngraph.opset8 import prelu
from ngraph.opset8 import prior_box
from ngraph.opset8 import prior_box_clustered
from ngraph.opset8 import psroi_pooling
from ngraph.opset8 import proposal
from ngraph.opset8 import range
from ngraph.opset8 import read_value
from ngraph.opset8 import reduce_l1
from ngraph.opset8 import reduce_l2
from ngraph.opset8 import reduce_logical_and
from ngraph.opset8 import reduce_logical_or
from ngraph.opset8 import reduce_max
from ngraph.opset8 import reduce_mean
from ngraph.opset8 import reduce_min
from ngraph.opset8 import reduce_prod
from ngraph.opset8 import reduce_sum
from ngraph.opset8 import region_yolo
from ngraph.opset8 import reorg_yolo
from ngraph.opset8 import relu
from ngraph.opset8 import reshape
from ngraph.opset8 import result
from ngraph.opset8 import reverse_sequence
from ngraph.opset8 import rnn_cell
from ngraph.opset8 import rnn_sequence
from ngraph.opset8 import roi_align
from ngraph.opset8 import roi_pooling
from ngraph.opset8 import roll
from ngraph.opset8 import round
from ngraph.opset8 import scatter_elements_update
from ngraph.opset8 import scatter_update
from ngraph.opset8 import select
from ngraph.opset8 import selu
from ngraph.opset8 import shape_of
from ngraph.opset8 import shuffle_channels
from ngraph.opset8 import sigmoid
from ngraph.opset8 import sign
from ngraph.opset8 import sin
from ngraph.opset8 import sinh
from ngraph.opset8 import softmax
from ngraph.opset8 import softplus
from ngraph.opset8 import space_to_batch
from ngraph.opset8 import space_to_depth
from ngraph.opset8 import split
from ngraph.opset8 import sqrt
from ngraph.opset8 import squared_difference
from ngraph.opset8 import squeeze
from ngraph.opset8 import strided_slice
from ngraph.opset8 import subtract
from ngraph.opset8 import swish
from ngraph.opset8 import tan
from ngraph.opset8 import tanh
from ngraph.opset8 import tensor_iterator
from ngraph.opset8 import tile
from ngraph.opset8 import topk
from ngraph.opset8 import transpose
from ngraph.opset8 import unsqueeze
from ngraph.opset8 import variadic_split


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
