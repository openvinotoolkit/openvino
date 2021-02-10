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

from ngraph.opset6 import absolute
from ngraph.opset6 import absolute as abs
from ngraph.opset6 import acos
from ngraph.opset6 import acosh
from ngraph.opset6 import add
from ngraph.opset6 import asin
from ngraph.opset6 import asinh
from ngraph.opset6 import assign
from ngraph.opset6 import atan
from ngraph.opset6 import atanh
from ngraph.opset6 import avg_pool
from ngraph.opset6 import batch_norm_inference
from ngraph.opset6 import batch_to_space
from ngraph.opset6 import binary_convolution
from ngraph.opset6 import broadcast
from ngraph.opset6 import bucketize
from ngraph.opset6 import ceiling
from ngraph.opset6 import ceiling as ceil
from ngraph.opset6 import clamp
from ngraph.opset6 import concat
from ngraph.opset6 import constant
from ngraph.opset6 import convert
from ngraph.opset6 import convert_like
from ngraph.opset6 import convolution
from ngraph.opset6 import convolution_backprop_data
from ngraph.opset6 import cos
from ngraph.opset6 import cosh
from ngraph.opset6 import ctc_greedy_decoder
from ngraph.opset6 import ctc_greedy_decoder_seq_len
from ngraph.opset6 import ctc_loss
from ngraph.opset6 import cum_sum
from ngraph.opset6 import cum_sum as cumsum
from ngraph.opset6 import deformable_convolution
from ngraph.opset6 import deformable_psroi_pooling
from ngraph.opset6 import depth_to_space
from ngraph.opset6 import detection_output
from ngraph.opset6 import divide
from ngraph.opset6 import elu
from ngraph.opset6 import embedding_bag_offsets_sum
from ngraph.opset6 import embedding_bag_packed_sum
from ngraph.opset6 import embedding_segments_sum
from ngraph.opset6 import extract_image_patches
from ngraph.opset6 import equal
from ngraph.opset6 import erf
from ngraph.opset6 import exp
from ngraph.opset6 import fake_quantize
from ngraph.opset6 import floor
from ngraph.opset6 import floor_mod
from ngraph.opset6 import gather
from ngraph.opset6 import gather_elements
from ngraph.opset6 import gather_nd
from ngraph.opset6 import gather_tree
from ngraph.opset6 import gelu
from ngraph.opset6 import greater
from ngraph.opset6 import greater_equal
from ngraph.opset6 import grn
from ngraph.opset6 import group_convolution
from ngraph.opset6 import group_convolution_backprop_data
from ngraph.opset6 import gru_cell
from ngraph.opset6 import gru_sequence
from ngraph.opset6 import hard_sigmoid
from ngraph.opset6 import hsigmoid
from ngraph.opset6 import hswish
from ngraph.opset6 import interpolate
from ngraph.opset6 import less
from ngraph.opset6 import less_equal
from ngraph.opset6 import log
from ngraph.opset6 import logical_and
from ngraph.opset6 import logical_not
from ngraph.opset6 import logical_or
from ngraph.opset6 import logical_xor
from ngraph.opset6 import log_softmax
from ngraph.opset6 import loop
from ngraph.opset6 import lrn
from ngraph.opset6 import lstm_cell
from ngraph.opset6 import lstm_sequence
from ngraph.opset6 import matmul
from ngraph.opset6 import max_pool
from ngraph.opset6 import maximum
from ngraph.opset6 import minimum
from ngraph.opset6 import mish
from ngraph.opset6 import mod
from ngraph.opset6 import multiply
from ngraph.opset6 import mvn
from ngraph.opset6 import negative
from ngraph.opset6 import non_max_suppression
from ngraph.opset6 import non_zero
from ngraph.opset6 import normalize_l2
from ngraph.opset6 import not_equal
from ngraph.opset6 import one_hot
from ngraph.opset6 import pad
from ngraph.opset6 import parameter
from ngraph.opset6 import power
from ngraph.opset6 import prelu
from ngraph.opset6 import prior_box
from ngraph.opset6 import prior_box_clustered
from ngraph.opset6 import psroi_pooling
from ngraph.opset6 import proposal
from ngraph.opset6 import range
from ngraph.opset6 import read_value
from ngraph.opset6 import reduce_l1
from ngraph.opset6 import reduce_l2
from ngraph.opset6 import reduce_logical_and
from ngraph.opset6 import reduce_logical_or
from ngraph.opset6 import reduce_max
from ngraph.opset6 import reduce_mean
from ngraph.opset6 import reduce_min
from ngraph.opset6 import reduce_prod
from ngraph.opset6 import reduce_sum
from ngraph.opset6 import region_yolo
from ngraph.opset6 import reorg_yolo
from ngraph.opset6 import relu
from ngraph.opset6 import reshape
from ngraph.opset6 import result
from ngraph.opset6 import reverse_sequence
from ngraph.opset6 import rnn_cell
from ngraph.opset6 import rnn_sequence
from ngraph.opset6 import roi_align
from ngraph.opset6 import roi_pooling
from ngraph.opset6 import round
from ngraph.opset6 import scatter_elements_update
from ngraph.opset6 import scatter_update
from ngraph.opset6 import select
from ngraph.opset6 import selu
from ngraph.opset6 import shape_of
from ngraph.opset6 import shuffle_channels
from ngraph.opset6 import sigmoid
from ngraph.opset6 import sign
from ngraph.opset6 import sin
from ngraph.opset6 import sinh
from ngraph.opset6 import softmax
from ngraph.opset6 import softplus
from ngraph.opset6 import space_to_batch
from ngraph.opset6 import space_to_depth
from ngraph.opset6 import split
from ngraph.opset6 import sqrt
from ngraph.opset6 import squared_difference
from ngraph.opset6 import squeeze
from ngraph.opset6 import strided_slice
from ngraph.opset6 import subtract
from ngraph.opset6 import swish
from ngraph.opset6 import tan
from ngraph.opset6 import tanh
from ngraph.opset6 import tensor_iterator
from ngraph.opset6 import tile
from ngraph.opset6 import topk
from ngraph.opset6 import transpose
from ngraph.opset6 import unsqueeze
from ngraph.opset6 import variadic_split


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
