# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
"""! ngraph module namespace, exposing factory functions for all ops and other classes."""
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

from ngraph.opset5 import absolute
from ngraph.opset5 import absolute as abs
from ngraph.opset5 import acos
from ngraph.opset5 import acosh
from ngraph.opset5 import add
from ngraph.opset5 import asin
from ngraph.opset5 import asinh
from ngraph.opset5 import assign
from ngraph.opset5 import atan
from ngraph.opset5 import atanh
from ngraph.opset5 import avg_pool
from ngraph.opset5 import batch_norm_inference
from ngraph.opset5 import batch_to_space
from ngraph.opset5 import binary_convolution
from ngraph.opset5 import broadcast
from ngraph.opset5 import bucketize
from ngraph.opset5 import ceiling
from ngraph.opset5 import ceiling as ceil
from ngraph.opset5 import clamp
from ngraph.opset5 import concat
from ngraph.opset5 import constant
from ngraph.opset5 import convert
from ngraph.opset5 import convert_like
from ngraph.opset5 import convolution
from ngraph.opset5 import convolution_backprop_data
from ngraph.opset5 import cos
from ngraph.opset5 import cosh
from ngraph.opset5 import ctc_greedy_decoder
from ngraph.opset5 import ctc_loss
from ngraph.opset5 import cum_sum
from ngraph.opset5 import cum_sum as cumsum
from ngraph.opset5 import deformable_convolution
from ngraph.opset5 import deformable_psroi_pooling
from ngraph.opset5 import depth_to_space
from ngraph.opset5 import detection_output
from ngraph.opset5 import divide
from ngraph.opset5 import elu
from ngraph.opset5 import embedding_bag_offsets_sum
from ngraph.opset5 import embedding_bag_packed_sum
from ngraph.opset5 import embedding_segments_sum
from ngraph.opset5 import extract_image_patches
from ngraph.opset5 import equal
from ngraph.opset5 import erf
from ngraph.opset5 import exp
from ngraph.opset5 import fake_quantize
from ngraph.opset5 import floor
from ngraph.opset5 import floor_mod
from ngraph.opset5 import gather
from ngraph.opset5 import gather_nd
from ngraph.opset5 import gather_tree
from ngraph.opset5 import gelu
from ngraph.opset5 import greater
from ngraph.opset5 import greater_equal
from ngraph.opset5 import grn
from ngraph.opset5 import group_convolution
from ngraph.opset5 import group_convolution_backprop_data
from ngraph.opset5 import gru_cell
from ngraph.opset5 import hard_sigmoid
from ngraph.opset5 import hsigmoid
from ngraph.opset5 import hswish
from ngraph.opset5 import interpolate
from ngraph.opset5 import less
from ngraph.opset5 import less_equal
from ngraph.opset5 import log
from ngraph.opset5 import logical_and
from ngraph.opset5 import logical_not
from ngraph.opset5 import logical_or
from ngraph.opset5 import logical_xor
from ngraph.opset5 import log_softmax
from ngraph.opset5 import lrn
from ngraph.opset5 import lstm_cell
from ngraph.opset5 import lstm_sequence
from ngraph.opset5 import matmul
from ngraph.opset5 import max_pool
from ngraph.opset5 import maximum
from ngraph.opset5 import minimum
from ngraph.opset5 import mish
from ngraph.opset5 import mod
from ngraph.opset5 import multiply
from ngraph.opset5 import mvn
from ngraph.opset5 import negative
from ngraph.opset5 import non_max_suppression
from ngraph.opset5 import non_zero
from ngraph.opset5 import normalize_l2
from ngraph.opset5 import not_equal
from ngraph.opset5 import one_hot
from ngraph.opset5 import pad
from ngraph.opset5 import parameter
from ngraph.opset5 import power
from ngraph.opset5 import prelu
from ngraph.opset5 import prior_box
from ngraph.opset5 import prior_box_clustered
from ngraph.opset5 import psroi_pooling
from ngraph.opset5 import proposal
from ngraph.opset5 import range
from ngraph.opset5 import read_value
from ngraph.opset5 import reduce_l1
from ngraph.opset5 import reduce_l2
from ngraph.opset5 import reduce_logical_and
from ngraph.opset5 import reduce_logical_or
from ngraph.opset5 import reduce_max
from ngraph.opset5 import reduce_mean
from ngraph.opset5 import reduce_min
from ngraph.opset5 import reduce_prod
from ngraph.opset5 import reduce_sum
from ngraph.opset5 import region_yolo
from ngraph.opset5 import reorg_yolo
from ngraph.opset5 import relu
from ngraph.opset5 import reshape
from ngraph.opset5 import result
from ngraph.opset5 import reverse_sequence
from ngraph.opset5 import rnn_cell
from ngraph.opset5 import roi_align
from ngraph.opset5 import roi_pooling
from ngraph.opset5 import round
from ngraph.opset5 import scatter_elements_update
from ngraph.opset5 import scatter_update
from ngraph.opset5 import select
from ngraph.opset5 import selu
from ngraph.opset5 import shape_of
from ngraph.opset5 import shuffle_channels
from ngraph.opset5 import sigmoid
from ngraph.opset5 import sign
from ngraph.opset5 import sin
from ngraph.opset5 import sinh
from ngraph.opset5 import softmax
from ngraph.opset5 import softplus
from ngraph.opset5 import space_to_batch
from ngraph.opset5 import space_to_depth
from ngraph.opset5 import split
from ngraph.opset5 import sqrt
from ngraph.opset5 import squared_difference
from ngraph.opset5 import squeeze
from ngraph.opset5 import strided_slice
from ngraph.opset5 import subtract
from ngraph.opset5 import swish
from ngraph.opset5 import tan
from ngraph.opset5 import tanh
from ngraph.opset5 import tensor_iterator
from ngraph.opset5 import tile
from ngraph.opset5 import topk
from ngraph.opset5 import transpose
from ngraph.opset5 import unsqueeze
from ngraph.opset5 import variadic_split


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
