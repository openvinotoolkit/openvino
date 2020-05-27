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
    __version__ = "0.0.0-dev"


from ngraph.ops import absolute
from ngraph.ops import absolute as abs
from ngraph.ops import acos
from ngraph.ops import add
from ngraph.ops import asin
from ngraph.ops import atan
from ngraph.ops import avg_pool
from ngraph.ops import batch_norm_inference
from ngraph.ops import batch_to_space
from ngraph.ops import binary_convolution
from ngraph.ops import broadcast
from ngraph.ops import bucketize
from ngraph.ops import ceiling
from ngraph.ops import ceiling as ceil
from ngraph.ops import clamp
from ngraph.ops import concat
from ngraph.ops import constant
from ngraph.ops import convert
from ngraph.ops import convert_like
from ngraph.ops import convolution
from ngraph.ops import convolution_backprop_data
from ngraph.ops import cos
from ngraph.ops import cosh
from ngraph.ops import ctc_greedy_decoder
from ngraph.ops import cum_sum
from ngraph.ops import cum_sum as cumsum
from ngraph.ops import deformable_convolution
from ngraph.ops import deformable_psroi_pooling
from ngraph.ops import depth_to_space
from ngraph.ops import detection_output
from ngraph.ops import divide
from ngraph.ops import elu
from ngraph.ops import embedding_bag_offsets_sum
from ngraph.ops import embedding_bag_packed_sum
from ngraph.ops import embedding_segments_sum
from ngraph.ops import equal
from ngraph.ops import erf
from ngraph.ops import exp
from ngraph.ops import fake_quantize
from ngraph.ops import floor
from ngraph.ops import floor_mod
from ngraph.ops import gather
from ngraph.ops import gather_tree
from ngraph.ops import gelu
from ngraph.ops import get_output_element
from ngraph.ops import greater
from ngraph.ops import greater_equal
from ngraph.ops import grn
from ngraph.ops import group_convolution
from ngraph.ops import group_convolution_backprop_data
from ngraph.ops import gru_cell
from ngraph.ops import hard_sigmoid
from ngraph.ops import interpolate
from ngraph.ops import less
from ngraph.ops import less_equal
from ngraph.ops import log
from ngraph.ops import logical_and
from ngraph.ops import logical_not
from ngraph.ops import logical_or
from ngraph.ops import logical_xor
from ngraph.ops import lrn
from ngraph.ops import lstm_cell
from ngraph.ops import lstm_sequence
from ngraph.ops import matmul
from ngraph.ops import max_pool
from ngraph.ops import maximum
from ngraph.ops import minimum
from ngraph.ops import mod
from ngraph.ops import multiply
from ngraph.ops import mvn
from ngraph.ops import negative
from ngraph.ops import non_max_suppression
from ngraph.ops import non_zero
from ngraph.ops import normalize_l2
from ngraph.ops import not_equal
from ngraph.ops import one_hot
from ngraph.ops import pad
from ngraph.ops import parameter
from ngraph.ops import power
from ngraph.ops import prelu
from ngraph.ops import prior_box
from ngraph.ops import prior_box_clustered
from ngraph.ops import psroi_pooling
from ngraph.ops import proposal
from ngraph.ops import reduce_logical_and
from ngraph.ops import reduce_logical_or
from ngraph.ops import reduce_max
from ngraph.ops import reduce_mean
from ngraph.ops import reduce_min
from ngraph.ops import reduce_prod
from ngraph.ops import reduce_sum
from ngraph.ops import region_yolo
from ngraph.ops import reorg_yolo
from ngraph.ops import relu
from ngraph.ops import reshape
from ngraph.ops import result
from ngraph.ops import reverse
from ngraph.ops import reverse_sequence
from ngraph.ops import rnn_cell
from ngraph.ops import roi_align
from ngraph.ops import roi_pooling
from ngraph.ops import scatter_elements_update
from ngraph.ops import scatter_nd_update
from ngraph.ops import scatter_update
from ngraph.ops import select
from ngraph.ops import selu
from ngraph.ops import shape_of
from ngraph.ops import shuffle_channels
from ngraph.ops import sigmoid
from ngraph.ops import sign
from ngraph.ops import sin
from ngraph.ops import sinh
from ngraph.ops import softmax
from ngraph.ops import space_to_batch
from ngraph.ops import space_to_depth
from ngraph.ops import split
from ngraph.ops import sqrt
from ngraph.ops import squared_difference
from ngraph.ops import squeeze
from ngraph.ops import strided_slice
from ngraph.ops import subtract
from ngraph.ops import tan
from ngraph.ops import tanh
from ngraph.ops import tile
from ngraph.ops import topk
from ngraph.ops import transpose
from ngraph.ops import unsqueeze
from ngraph.ops import variadic_split


from ngraph.runtime import runtime
