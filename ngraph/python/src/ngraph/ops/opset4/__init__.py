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


from ngraph.ops.opset1 import absolute
from ngraph.ops.opset1 import absolute as abs
from ngraph.ops.opset1 import acos
from ngraph.ops.opset1 import add
from ngraph.ops.opset1 import asin
from ngraph.ops.opset3 import assign
from ngraph.ops.opset1 import atan
from ngraph.ops.opset1 import avg_pool
from ngraph.ops.opset1 import batch_norm_inference
from ngraph.ops.opset2 import batch_to_space
from ngraph.ops.opset1 import binary_convolution
from ngraph.ops.opset3 import broadcast
from ngraph.ops.opset3 import bucketize
from ngraph.ops.opset1 import ceiling
from ngraph.ops.opset1 import ceiling as ceil
from ngraph.ops.opset1 import clamp
from ngraph.ops.opset1 import concat
from ngraph.ops.opset1 import constant
from ngraph.ops.opset1 import convert
from ngraph.ops.opset1 import convert_like
from ngraph.ops.opset1 import convolution
from ngraph.ops.opset1 import convolution_backprop_data
from ngraph.ops.opset1 import cos
from ngraph.ops.opset1 import cosh
from ngraph.ops.opset1 import ctc_greedy_decoder
from ngraph.ops.opset3 import cum_sum
from ngraph.ops.opset3 import cum_sum as cumsum
from ngraph.ops.opset1 import deformable_convolution
from ngraph.ops.opset1 import deformable_psroi_pooling
from ngraph.ops.opset1 import depth_to_space
from ngraph.ops.opset1 import detection_output
from ngraph.ops.opset1 import divide
from ngraph.ops.opset1 import elu
from ngraph.ops.opset3 import embedding_bag_offsets_sum
from ngraph.ops.opset3 import embedding_bag_packed_sum
from ngraph.ops.opset3 import embedding_segments_sum
from ngraph.ops.opset3 import extract_image_patches
from ngraph.ops.opset1 import equal
from ngraph.ops.opset1 import erf
from ngraph.ops.opset1 import exp
from ngraph.ops.opset1 import fake_quantize
from ngraph.ops.opset1 import floor
from ngraph.ops.opset1 import floor_mod
from ngraph.ops.opset1 import gather
from ngraph.ops.opset1 import gather_tree
from ngraph.ops.opset2 import gelu
from ngraph.ops.inner_ops import get_output_element
from ngraph.ops.opset1 import greater
from ngraph.ops.opset1 import greater_equal
from ngraph.ops.opset1 import grn
from ngraph.ops.opset1 import group_convolution
from ngraph.ops.opset1 import group_convolution_backprop_data
from ngraph.ops.opset3 import gru_cell
from ngraph.ops.opset1 import hard_sigmoid
from ngraph.ops.opset1 import interpolate
from ngraph.ops.opset1 import less
from ngraph.ops.opset1 import less_equal
from ngraph.ops.opset1 import log
from ngraph.ops.opset1 import logical_and
from ngraph.ops.opset1 import logical_not
from ngraph.ops.opset1 import logical_or
from ngraph.ops.opset1 import logical_xor
from ngraph.ops.opset1 import lrn
from ngraph.ops.opset1 import lstm_cell
from ngraph.ops.opset1 import lstm_sequence
from ngraph.ops.opset1 import matmul
from ngraph.ops.opset1 import max_pool
from ngraph.ops.opset1 import maximum
from ngraph.ops.opset1 import minimum
from ngraph.ops.opset1 import mod
from ngraph.ops.opset1 import multiply
from ngraph.ops.opset2 import mvn
from ngraph.ops.opset1 import negative
from ngraph.ops.opset4 import non_max_suppression
from ngraph.ops.opset3 import non_zero
from ngraph.ops.opset1 import normalize_l2
from ngraph.ops.opset1 import not_equal
from ngraph.ops.opset1 import one_hot
from ngraph.ops.opset1 import pad
from ngraph.ops.opset1 import parameter
from ngraph.ops.opset1 import power
from ngraph.ops.opset1 import prelu
from ngraph.ops.opset1 import prior_box
from ngraph.ops.opset1 import prior_box_clustered
from ngraph.ops.opset1 import psroi_pooling
from ngraph.ops.opset1 import proposal
from ngraph.ops.opset3 import read_value
from ngraph.ops.opset1 import reduce_logical_and
from ngraph.ops.opset1 import reduce_logical_or
from ngraph.ops.opset1 import reduce_max
from ngraph.ops.opset1 import reduce_mean
from ngraph.ops.opset1 import reduce_min
from ngraph.ops.opset1 import reduce_prod
from ngraph.ops.opset1 import reduce_sum
from ngraph.ops.opset1 import region_yolo
from ngraph.ops.opset2 import reorg_yolo
from ngraph.ops.opset1 import relu
from ngraph.ops.opset1 import reshape
from ngraph.ops.opset1 import result
from ngraph.ops.opset3 import reverse
from ngraph.ops.opset1 import reverse_sequence
from ngraph.ops.opset3 import rnn_cell
from ngraph.ops.opset3 import roi_align
from ngraph.ops.opset2 import roi_pooling
from ngraph.ops.opset3 import scatter_elements_update
from ngraph.ops.opset3 import scatter_update
from ngraph.ops.opset1 import select
from ngraph.ops.opset1 import selu
from ngraph.ops.opset3 import shape_of
from ngraph.ops.opset3 import shuffle_channels
from ngraph.ops.opset1 import sigmoid
from ngraph.ops.opset1 import sign
from ngraph.ops.opset1 import sin
from ngraph.ops.opset1 import sinh
from ngraph.ops.opset1 import softmax
from ngraph.ops.opset2 import space_to_batch
from ngraph.ops.opset1 import space_to_depth
from ngraph.ops.opset1 import split
from ngraph.ops.opset1 import sqrt
from ngraph.ops.opset1 import squared_difference
from ngraph.ops.opset1 import squeeze
from ngraph.ops.opset1 import strided_slice
from ngraph.ops.opset1 import subtract
from ngraph.ops.opset1 import tan
from ngraph.ops.opset1 import tanh
from ngraph.ops.opset1 import tensor_iterator
from ngraph.ops.opset1 import tile
from ngraph.ops.opset3 import topk
from ngraph.ops.opset1 import transpose
from ngraph.ops.opset1 import unsqueeze
from ngraph.ops.opset1 import variadic_split
