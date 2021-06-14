# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ngraph module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("ngraph-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

from ngraph.impl import Node
from ngraph.impl import Function
from ngraph.helpers import function_from_cnn
from ngraph.helpers import function_to_cnn

from ngraph.opset7 import absolute
from ngraph.opset7 import absolute as abs
from ngraph.opset7 import acos
from ngraph.opset7 import acosh
from ngraph.opset7 import add
from ngraph.opset7 import asin
from ngraph.opset7 import asinh
from ngraph.opset7 import assign
from ngraph.opset7 import atan
from ngraph.opset7 import atanh
from ngraph.opset7 import avg_pool
from ngraph.opset7 import batch_norm_inference
from ngraph.opset7 import batch_to_space
from ngraph.opset7 import binary_convolution
from ngraph.opset7 import broadcast
from ngraph.opset7 import bucketize
from ngraph.opset7 import ceiling
from ngraph.opset7 import ceiling as ceil
from ngraph.opset7 import clamp
from ngraph.opset7 import concat
from ngraph.opset7 import constant
from ngraph.opset7 import convert
from ngraph.opset7 import convert_like
from ngraph.opset7 import convolution
from ngraph.opset7 import convolution_backprop_data
from ngraph.opset7 import cos
from ngraph.opset7 import cosh
from ngraph.opset7 import ctc_greedy_decoder
from ngraph.opset7 import ctc_greedy_decoder_seq_len
from ngraph.opset7 import ctc_loss
from ngraph.opset7 import cum_sum
from ngraph.opset7 import cum_sum as cumsum
from ngraph.opset7 import deformable_convolution
from ngraph.opset7 import deformable_psroi_pooling
from ngraph.opset7 import depth_to_space
from ngraph.opset7 import detection_output
from ngraph.opset7 import dft
from ngraph.opset7 import divide
from ngraph.opset7 import einsum
from ngraph.opset7 import elu
from ngraph.opset7 import embedding_bag_offsets_sum
from ngraph.opset7 import embedding_bag_packed_sum
from ngraph.opset7 import embedding_segments_sum
from ngraph.opset7 import extract_image_patches
from ngraph.opset7 import equal
from ngraph.opset7 import erf
from ngraph.opset7 import exp
from ngraph.opset7 import fake_quantize
from ngraph.opset7 import floor
from ngraph.opset7 import floor_mod
from ngraph.opset7 import gather
from ngraph.opset7 import gather_elements
from ngraph.opset7 import gather_nd
from ngraph.opset7 import gather_tree
from ngraph.opset7 import gelu
from ngraph.opset7 import greater
from ngraph.opset7 import greater_equal
from ngraph.opset7 import grn
from ngraph.opset7 import group_convolution
from ngraph.opset7 import group_convolution_backprop_data
from ngraph.opset7 import gru_cell
from ngraph.opset7 import gru_sequence
from ngraph.opset7 import hard_sigmoid
from ngraph.opset7 import hsigmoid
from ngraph.opset7 import hswish
from ngraph.opset7 import idft
from ngraph.opset7 import interpolate
from ngraph.opset7 import less
from ngraph.opset7 import less_equal
from ngraph.opset7 import log
from ngraph.opset7 import logical_and
from ngraph.opset7 import logical_not
from ngraph.opset7 import logical_or
from ngraph.opset7 import logical_xor
from ngraph.opset7 import log_softmax
from ngraph.opset7 import loop
from ngraph.opset7 import lrn
from ngraph.opset7 import lstm_cell
from ngraph.opset7 import lstm_sequence
from ngraph.opset7 import matmul
from ngraph.opset7 import max_pool
from ngraph.opset7 import maximum
from ngraph.opset7 import minimum
from ngraph.opset7 import mish
from ngraph.opset7 import mod
from ngraph.opset7 import multiply
from ngraph.opset7 import mvn
from ngraph.opset7 import negative
from ngraph.opset7 import non_max_suppression
from ngraph.opset7 import non_zero
from ngraph.opset7 import normalize_l2
from ngraph.opset7 import not_equal
from ngraph.opset7 import one_hot
from ngraph.opset7 import pad
from ngraph.opset7 import parameter
from ngraph.opset7 import power
from ngraph.opset7 import prelu
from ngraph.opset7 import prior_box
from ngraph.opset7 import prior_box_clustered
from ngraph.opset7 import psroi_pooling
from ngraph.opset7 import proposal
from ngraph.opset7 import range
from ngraph.opset7 import read_value
from ngraph.opset7 import reduce_l1
from ngraph.opset7 import reduce_l2
from ngraph.opset7 import reduce_logical_and
from ngraph.opset7 import reduce_logical_or
from ngraph.opset7 import reduce_max
from ngraph.opset7 import reduce_mean
from ngraph.opset7 import reduce_min
from ngraph.opset7 import reduce_prod
from ngraph.opset7 import reduce_sum
from ngraph.opset7 import region_yolo
from ngraph.opset7 import reorg_yolo
from ngraph.opset7 import relu
from ngraph.opset7 import reshape
from ngraph.opset7 import result
from ngraph.opset7 import reverse_sequence
from ngraph.opset7 import rnn_cell
from ngraph.opset7 import rnn_sequence
from ngraph.opset7 import roi_align
from ngraph.opset7 import roi_pooling
from ngraph.opset7 import roll
from ngraph.opset7 import round
from ngraph.opset7 import scatter_elements_update
from ngraph.opset7 import scatter_update
from ngraph.opset7 import select
from ngraph.opset7 import selu
from ngraph.opset7 import shape_of
from ngraph.opset7 import shuffle_channels
from ngraph.opset7 import sigmoid
from ngraph.opset7 import sign
from ngraph.opset7 import sin
from ngraph.opset7 import sinh
from ngraph.opset7 import softmax
from ngraph.opset7 import softplus
from ngraph.opset7 import space_to_batch
from ngraph.opset7 import space_to_depth
from ngraph.opset7 import split
from ngraph.opset7 import sqrt
from ngraph.opset7 import squared_difference
from ngraph.opset7 import squeeze
from ngraph.opset7 import strided_slice
from ngraph.opset7 import subtract
from ngraph.opset7 import swish
from ngraph.opset7 import tan
from ngraph.opset7 import tanh
from ngraph.opset7 import tensor_iterator
from ngraph.opset7 import tile
from ngraph.opset7 import topk
from ngraph.opset7 import transpose
from ngraph.opset7 import unsqueeze
from ngraph.opset7 import variadic_split


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
