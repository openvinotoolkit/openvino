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
from ngraph.opset12 import absolute
from ngraph.opset12 import absolute as abs
from ngraph.opset12 import acos
from ngraph.opset12 import acosh
from ngraph.opset12 import adaptive_avg_pool
from ngraph.opset12 import adaptive_max_pool
from ngraph.opset12 import add
from ngraph.opset12 import asin
from ngraph.opset12 import asinh
from ngraph.opset12 import assign
from ngraph.opset12 import atan
from ngraph.opset12 import atanh
from ngraph.opset12 import avg_pool
from ngraph.opset12 import batch_norm_inference
from ngraph.opset12 import batch_to_space
from ngraph.opset12 import binary_convolution
from ngraph.opset12 import broadcast
from ngraph.opset12 import bucketize
from ngraph.opset12 import ceiling
from ngraph.opset12 import ceiling as ceil
from ngraph.opset12 import clamp
from ngraph.opset12 import concat
from ngraph.opset12 import constant
from ngraph.opset12 import convert
from ngraph.opset12 import convert_like
from ngraph.opset12 import convolution
from ngraph.opset12 import convolution_backprop_data
from ngraph.opset12 import cos
from ngraph.opset12 import cosh
from ngraph.opset12 import ctc_greedy_decoder
from ngraph.opset12 import ctc_greedy_decoder_seq_len
from ngraph.opset12 import ctc_loss
from ngraph.opset12 import cum_sum
from ngraph.opset12 import cum_sum as cumsum
from ngraph.opset12 import deformable_convolution
from ngraph.opset12 import deformable_psroi_pooling
from ngraph.opset12 import depth_to_space
from ngraph.opset12 import detection_output
from ngraph.opset12 import dft
from ngraph.opset12 import divide
from ngraph.opset12 import einsum
from ngraph.opset12 import elu
from ngraph.opset12 import embedding_bag_offsets_sum
from ngraph.opset12 import embedding_bag_packed_sum
from ngraph.opset12 import embedding_segments_sum
from ngraph.opset12 import extract_image_patches
from ngraph.opset12 import equal
from ngraph.opset12 import erf
from ngraph.opset12 import exp
from ngraph.opset12 import eye
from ngraph.opset12 import fake_quantize
from ngraph.opset12 import floor
from ngraph.opset12 import floor_mod
from ngraph.opset12 import gather
from ngraph.opset12 import gather_elements
from ngraph.opset12 import gather_nd
from ngraph.opset12 import gather_tree
from ngraph.opset12 import gelu
from ngraph.opset12 import generate_proposals
from ngraph.opset12 import greater
from ngraph.opset12 import greater_equal
from ngraph.opset12 import grid_sample
from ngraph.opset12 import grn
from ngraph.opset12 import group_convolution
from ngraph.opset12 import group_convolution_backprop_data
from ngraph.opset12 import gru_cell
from ngraph.opset12 import gru_sequence
from ngraph.opset12 import hard_sigmoid
from ngraph.opset12 import hsigmoid
from ngraph.opset12 import hswish
from ngraph.opset12 import idft
from ngraph.opset12 import if_op
from ngraph.opset12 import interpolate
from ngraph.opset12 import irdft
from ngraph.opset12 import is_finite
from ngraph.opset12 import is_inf
from ngraph.opset12 import is_nan
from ngraph.opset12 import i420_to_bgr
from ngraph.opset12 import i420_to_rgb
from ngraph.opset12 import less
from ngraph.opset12 import less_equal
from ngraph.opset12 import log
from ngraph.opset12 import logical_and
from ngraph.opset12 import logical_not
from ngraph.opset12 import logical_or
from ngraph.opset12 import logical_xor
from ngraph.opset12 import log_softmax
from ngraph.opset12 import loop
from ngraph.opset12 import lrn
from ngraph.opset12 import lstm_cell
from ngraph.opset12 import lstm_sequence
from ngraph.opset12 import matmul
from ngraph.opset12 import matrix_nms
from ngraph.opset12 import max_pool
from ngraph.opset12 import maximum
from ngraph.opset12 import minimum
from ngraph.opset12 import mish
from ngraph.opset12 import mod
from ngraph.opset12 import multiclass_nms
from ngraph.opset12 import multiply
from ngraph.opset12 import mvn
from ngraph.opset12 import negative
from ngraph.opset12 import non_max_suppression
from ngraph.opset12 import non_zero
from ngraph.opset12 import normalize_l2
from ngraph.opset12 import not_equal
from ngraph.opset12 import nv12_to_bgr
from ngraph.opset12 import nv12_to_rgb
from ngraph.opset12 import one_hot
from ngraph.opset12 import pad
from ngraph.opset12 import parameter
from ngraph.opset12 import power
from ngraph.opset12 import prelu
from ngraph.opset12 import prior_box
from ngraph.opset12 import prior_box_clustered
from ngraph.opset12 import psroi_pooling
from ngraph.opset12 import proposal
from ngraph.opset12 import random_uniform
from ngraph.opset12 import range
from ngraph.opset12 import rdft
from ngraph.opset12 import read_value
from ngraph.opset12 import reduce_l1
from ngraph.opset12 import reduce_l2
from ngraph.opset12 import reduce_logical_and
from ngraph.opset12 import reduce_logical_or
from ngraph.opset12 import reduce_max
from ngraph.opset12 import reduce_mean
from ngraph.opset12 import reduce_min
from ngraph.opset12 import reduce_prod
from ngraph.opset12 import reduce_sum
from ngraph.opset12 import region_yolo
from ngraph.opset12 import reorg_yolo
from ngraph.opset12 import relu
from ngraph.opset12 import reshape
from ngraph.opset12 import result
from ngraph.opset12 import reverse_sequence
from ngraph.opset12 import rnn_cell
from ngraph.opset12 import rnn_sequence
from ngraph.opset12 import roi_align
from ngraph.opset12 import roi_pooling
from ngraph.opset12 import roll
from ngraph.opset12 import round
from ngraph.opset12 import scatter_elements_update
from ngraph.opset12 import scatter_update
from ngraph.opset12 import select
from ngraph.opset12 import selu
from ngraph.opset12 import shape_of
from ngraph.opset12 import shuffle_channels
from ngraph.opset12 import sigmoid
from ngraph.opset12 import sign
from ngraph.opset12 import sin
from ngraph.opset12 import sinh
from ngraph.opset12 import slice
from ngraph.opset12 import softmax
from ngraph.opset12 import softplus
from ngraph.opset12 import softsign
from ngraph.opset12 import space_to_batch
from ngraph.opset12 import space_to_depth
from ngraph.opset12 import split
from ngraph.opset12 import sqrt
from ngraph.opset12 import squared_difference
from ngraph.opset12 import squeeze
from ngraph.opset12 import strided_slice
from ngraph.opset12 import subtract
from ngraph.opset12 import swish
from ngraph.opset12 import tan
from ngraph.opset12 import tanh
from ngraph.opset12 import tensor_iterator
from ngraph.opset12 import tile
from ngraph.opset12 import topk
from ngraph.opset12 import transpose
from ngraph.opset12 import unique
from ngraph.opset12 import unsqueeze
from ngraph.opset12 import variadic_split

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
