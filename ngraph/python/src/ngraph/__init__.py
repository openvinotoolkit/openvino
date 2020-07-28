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
"""ngraph module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("ngraph-core").version
except DistributionNotFound:
    __version__ = "0.0.0.dev0"

from ngraph.impl import Node

from ngraph.opset4 import absolute
from ngraph.opset4 import absolute as abs
from ngraph.opset4 import acos
from ngraph.opset4 import add
from ngraph.opset4 import asin
from ngraph.opset4 import assign
from ngraph.opset4 import atan
from ngraph.opset4 import avg_pool
from ngraph.opset4 import batch_norm_inference
from ngraph.opset4 import batch_to_space
from ngraph.opset4 import binary_convolution
from ngraph.opset4 import broadcast
from ngraph.opset4 import bucketize
from ngraph.opset4 import ceiling
from ngraph.opset4 import ceiling as ceil
from ngraph.opset4 import clamp
from ngraph.opset4 import concat
from ngraph.opset4 import constant
from ngraph.opset4 import convert
from ngraph.opset4 import convert_like
from ngraph.opset4 import convolution
from ngraph.opset4 import convolution_backprop_data
from ngraph.opset4 import cos
from ngraph.opset4 import cosh
from ngraph.opset4 import ctc_greedy_decoder
from ngraph.opset4 import cum_sum
from ngraph.opset4 import cum_sum as cumsum
from ngraph.opset4 import deformable_convolution
from ngraph.opset4 import deformable_psroi_pooling
from ngraph.opset4 import depth_to_space
from ngraph.opset4 import detection_output
from ngraph.opset4 import divide
from ngraph.opset4 import elu
from ngraph.opset4 import embedding_bag_offsets_sum
from ngraph.opset4 import embedding_bag_packed_sum
from ngraph.opset4 import embedding_segments_sum
from ngraph.opset4 import extract_image_patches
from ngraph.opset4 import equal
from ngraph.opset4 import erf
from ngraph.opset4 import exp
from ngraph.opset4 import fake_quantize
from ngraph.opset4 import floor
from ngraph.opset4 import floor_mod
from ngraph.opset4 import gather
from ngraph.opset4 import gather_tree
from ngraph.opset4 import gelu
from ngraph.opset4 import greater
from ngraph.opset4 import greater_equal
from ngraph.opset4 import grn
from ngraph.opset4 import group_convolution
from ngraph.opset4 import group_convolution_backprop_data
from ngraph.opset4 import gru_cell
from ngraph.opset4 import hard_sigmoid
from ngraph.opset4 import interpolate
from ngraph.opset4 import less
from ngraph.opset4 import less_equal
from ngraph.opset4 import log
from ngraph.opset4 import logical_and
from ngraph.opset4 import logical_not
from ngraph.opset4 import logical_or
from ngraph.opset4 import logical_xor
from ngraph.opset4 import lrn
from ngraph.opset4 import lstm_cell
from ngraph.opset4 import lstm_sequence
from ngraph.opset4 import matmul
from ngraph.opset4 import max_pool
from ngraph.opset4 import maximum
from ngraph.opset4 import minimum
from ngraph.opset4 import mod
from ngraph.opset4 import multiply
from ngraph.opset4 import mvn
from ngraph.opset4 import negative
from ngraph.opset4 import non_max_suppression
from ngraph.opset4 import non_zero
from ngraph.opset4 import normalize_l2
from ngraph.opset4 import not_equal
from ngraph.opset4 import one_hot
from ngraph.opset4 import pad
from ngraph.opset4 import parameter
from ngraph.opset4 import power
from ngraph.opset4 import prelu
from ngraph.opset4 import prior_box
from ngraph.opset4 import prior_box_clustered
from ngraph.opset4 import psroi_pooling
from ngraph.opset4 import proposal
from ngraph.opset4 import range
from ngraph.opset4 import read_value
from ngraph.opset4 import reduce_logical_and
from ngraph.opset4 import reduce_logical_or
from ngraph.opset4 import reduce_max
from ngraph.opset4 import reduce_mean
from ngraph.opset4 import reduce_min
from ngraph.opset4 import reduce_prod
from ngraph.opset4 import reduce_sum
from ngraph.opset4 import region_yolo
from ngraph.opset4 import reorg_yolo
from ngraph.opset4 import relu
from ngraph.opset4 import reshape
from ngraph.opset4 import result
from ngraph.opset4 import reverse
from ngraph.opset4 import reverse_sequence
from ngraph.opset4 import rnn_cell
from ngraph.opset4 import roi_align
from ngraph.opset4 import roi_pooling
from ngraph.opset4 import scatter_elements_update
from ngraph.opset4 import scatter_update
from ngraph.opset4 import select
from ngraph.opset4 import selu
from ngraph.opset4 import shape_of
from ngraph.opset4 import shuffle_channels
from ngraph.opset4 import sigmoid
from ngraph.opset4 import sign
from ngraph.opset4 import sin
from ngraph.opset4 import sinh
from ngraph.opset4 import softmax
from ngraph.opset4 import space_to_batch
from ngraph.opset4 import space_to_depth
from ngraph.opset4 import split
from ngraph.opset4 import sqrt
from ngraph.opset4 import squared_difference
from ngraph.opset4 import squeeze
from ngraph.opset4 import strided_slice
from ngraph.opset4 import subtract
from ngraph.opset4 import tan
from ngraph.opset4 import tanh
from ngraph.opset4 import tensor_iterator
from ngraph.opset4 import tile
from ngraph.opset4 import topk
from ngraph.opset4 import transpose
from ngraph.opset4 import unsqueeze
from ngraph.opset4 import variadic_split


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
