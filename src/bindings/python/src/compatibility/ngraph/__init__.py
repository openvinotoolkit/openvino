# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ngraph module namespace, exposing factory functions for all ops and other classes."""
# noqa: F401

try:
    from ngraph.impl import util

    __version__ = util.get_ngraph_version_string()
except ImportError:
    __version__ = "0.0.0.dev0"


from ngraph.impl import Dimension
from ngraph.impl import Function
from ngraph.impl import Node
from ngraph.impl import PartialShape
from ngraph.helpers import function_from_cnn
from ngraph.helpers import function_to_cnn
from ngraph.opset11 import absolute
from ngraph.opset11 import absolute as abs
from ngraph.opset11 import acos
from ngraph.opset11 import acosh
from ngraph.opset11 import adaptive_avg_pool
from ngraph.opset11 import adaptive_max_pool
from ngraph.opset11 import add
from ngraph.opset11 import asin
from ngraph.opset11 import asinh
from ngraph.opset11 import assign
from ngraph.opset11 import atan
from ngraph.opset11 import atanh
from ngraph.opset11 import avg_pool
from ngraph.opset11 import batch_norm_inference
from ngraph.opset11 import batch_to_space
from ngraph.opset11 import binary_convolution
from ngraph.opset11 import broadcast
from ngraph.opset11 import bucketize
from ngraph.opset11 import ceiling
from ngraph.opset11 import ceiling as ceil
from ngraph.opset11 import clamp
from ngraph.opset11 import concat
from ngraph.opset11 import constant
from ngraph.opset11 import convert
from ngraph.opset11 import convert_like
from ngraph.opset11 import convolution
from ngraph.opset11 import convolution_backprop_data
from ngraph.opset11 import cos
from ngraph.opset11 import cosh
from ngraph.opset11 import ctc_greedy_decoder
from ngraph.opset11 import ctc_greedy_decoder_seq_len
from ngraph.opset11 import ctc_loss
from ngraph.opset11 import cum_sum
from ngraph.opset11 import cum_sum as cumsum
from ngraph.opset11 import deformable_convolution
from ngraph.opset11 import deformable_psroi_pooling
from ngraph.opset11 import depth_to_space
from ngraph.opset11 import detection_output
from ngraph.opset11 import dft
from ngraph.opset11 import divide
from ngraph.opset11 import einsum
from ngraph.opset11 import elu
from ngraph.opset11 import embedding_bag_offsets_sum
from ngraph.opset11 import embedding_bag_packed_sum
from ngraph.opset11 import embedding_segments_sum
from ngraph.opset11 import extract_image_patches
from ngraph.opset11 import equal
from ngraph.opset11 import erf
from ngraph.opset11 import exp
from ngraph.opset11 import eye
from ngraph.opset11 import fake_quantize
from ngraph.opset11 import floor
from ngraph.opset11 import floor_mod
from ngraph.opset11 import gather
from ngraph.opset11 import gather_elements
from ngraph.opset11 import gather_nd
from ngraph.opset11 import gather_tree
from ngraph.opset11 import gelu
from ngraph.opset11 import generate_proposals
from ngraph.opset11 import greater
from ngraph.opset11 import greater_equal
from ngraph.opset11 import grid_sample
from ngraph.opset11 import grn
from ngraph.opset11 import group_convolution
from ngraph.opset11 import group_convolution_backprop_data
from ngraph.opset11 import gru_cell
from ngraph.opset11 import gru_sequence
from ngraph.opset11 import hard_sigmoid
from ngraph.opset11 import hsigmoid
from ngraph.opset11 import hswish
from ngraph.opset11 import idft
from ngraph.opset11 import if_op
from ngraph.opset11 import interpolate
from ngraph.opset11 import irdft
from ngraph.opset11 import is_finite
from ngraph.opset11 import is_inf
from ngraph.opset11 import is_nan
from ngraph.opset11 import i420_to_bgr
from ngraph.opset11 import i420_to_rgb
from ngraph.opset11 import less
from ngraph.opset11 import less_equal
from ngraph.opset11 import log
from ngraph.opset11 import logical_and
from ngraph.opset11 import logical_not
from ngraph.opset11 import logical_or
from ngraph.opset11 import logical_xor
from ngraph.opset11 import log_softmax
from ngraph.opset11 import loop
from ngraph.opset11 import lrn
from ngraph.opset11 import lstm_cell
from ngraph.opset11 import lstm_sequence
from ngraph.opset11 import matmul
from ngraph.opset11 import matrix_nms
from ngraph.opset11 import max_pool
from ngraph.opset11 import maximum
from ngraph.opset11 import minimum
from ngraph.opset11 import mish
from ngraph.opset11 import mod
from ngraph.opset11 import multiclass_nms
from ngraph.opset11 import multiply
from ngraph.opset11 import mvn
from ngraph.opset11 import negative
from ngraph.opset11 import non_max_suppression
from ngraph.opset11 import non_zero
from ngraph.opset11 import normalize_l2
from ngraph.opset11 import not_equal
from ngraph.opset11 import nv12_to_bgr
from ngraph.opset11 import nv12_to_rgb
from ngraph.opset11 import one_hot
from ngraph.opset11 import pad
from ngraph.opset11 import parameter
from ngraph.opset11 import power
from ngraph.opset11 import prelu
from ngraph.opset11 import prior_box
from ngraph.opset11 import prior_box_clustered
from ngraph.opset11 import psroi_pooling
from ngraph.opset11 import proposal
from ngraph.opset11 import random_uniform
from ngraph.opset11 import range
from ngraph.opset11 import rdft
from ngraph.opset11 import read_value
from ngraph.opset11 import reduce_l1
from ngraph.opset11 import reduce_l2
from ngraph.opset11 import reduce_logical_and
from ngraph.opset11 import reduce_logical_or
from ngraph.opset11 import reduce_max
from ngraph.opset11 import reduce_mean
from ngraph.opset11 import reduce_min
from ngraph.opset11 import reduce_prod
from ngraph.opset11 import reduce_sum
from ngraph.opset11 import region_yolo
from ngraph.opset11 import reorg_yolo
from ngraph.opset11 import relu
from ngraph.opset11 import reshape
from ngraph.opset11 import result
from ngraph.opset11 import reverse_sequence
from ngraph.opset11 import rnn_cell
from ngraph.opset11 import rnn_sequence
from ngraph.opset11 import roi_align
from ngraph.opset11 import roi_pooling
from ngraph.opset11 import roll
from ngraph.opset11 import round
from ngraph.opset11 import scatter_elements_update
from ngraph.opset11 import scatter_update
from ngraph.opset11 import select
from ngraph.opset11 import selu
from ngraph.opset11 import shape_of
from ngraph.opset11 import shuffle_channels
from ngraph.opset11 import sigmoid
from ngraph.opset11 import sign
from ngraph.opset11 import sin
from ngraph.opset11 import sinh
from ngraph.opset11 import slice
from ngraph.opset11 import softmax
from ngraph.opset11 import softplus
from ngraph.opset11 import softsign
from ngraph.opset11 import space_to_batch
from ngraph.opset11 import space_to_depth
from ngraph.opset11 import split
from ngraph.opset11 import sqrt
from ngraph.opset11 import squared_difference
from ngraph.opset11 import squeeze
from ngraph.opset11 import strided_slice
from ngraph.opset11 import subtract
from ngraph.opset11 import swish
from ngraph.opset11 import tan
from ngraph.opset11 import tanh
from ngraph.opset11 import tensor_iterator
from ngraph.opset11 import tile
from ngraph.opset11 import topk
from ngraph.opset11 import transpose
from ngraph.opset11 import unique
from ngraph.opset11 import unsqueeze
from ngraph.opset11 import variadic_split

import warnings

warnings.warn(
    message="OpenVINO nGraph Python API is deprecated and will be removed in 2024.0 release."
            "For instructions on transitioning to the new API, please refer to "
            "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html",
    category=FutureWarning,
    stacklevel=2,
)

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
