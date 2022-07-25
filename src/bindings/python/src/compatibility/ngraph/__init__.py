# Copyright (C) 2018-2022 Intel Corporation
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
from ngraph.opset9 import absolute
from ngraph.opset9 import absolute as abs
from ngraph.opset9 import acos
from ngraph.opset9 import acosh
from ngraph.opset9 import adaptive_avg_pool
from ngraph.opset9 import adaptive_max_pool
from ngraph.opset9 import add
from ngraph.opset9 import asin
from ngraph.opset9 import asinh
from ngraph.opset9 import assign
from ngraph.opset9 import atan
from ngraph.opset9 import atanh
from ngraph.opset9 import avg_pool
from ngraph.opset9 import batch_norm_inference
from ngraph.opset9 import batch_to_space
from ngraph.opset9 import binary_convolution
from ngraph.opset9 import broadcast
from ngraph.opset9 import bucketize
from ngraph.opset9 import ceiling
from ngraph.opset9 import ceiling as ceil
from ngraph.opset9 import clamp
from ngraph.opset9 import concat
from ngraph.opset9 import constant
from ngraph.opset9 import convert
from ngraph.opset9 import convert_like
from ngraph.opset9 import convolution
from ngraph.opset9 import convolution_backprop_data
from ngraph.opset9 import cos
from ngraph.opset9 import cosh
from ngraph.opset9 import ctc_greedy_decoder
from ngraph.opset9 import ctc_greedy_decoder_seq_len
from ngraph.opset9 import ctc_loss
from ngraph.opset9 import cum_sum
from ngraph.opset9 import cum_sum as cumsum
from ngraph.opset9 import deformable_convolution
from ngraph.opset9 import deformable_psroi_pooling
from ngraph.opset9 import depth_to_space
from ngraph.opset9 import detection_output
from ngraph.opset9 import dft
from ngraph.opset9 import divide
from ngraph.opset9 import einsum
from ngraph.opset9 import elu
from ngraph.opset9 import embedding_bag_offsets_sum
from ngraph.opset9 import embedding_bag_packed_sum
from ngraph.opset9 import embedding_segments_sum
from ngraph.opset9 import extract_image_patches
from ngraph.opset9 import equal
from ngraph.opset9 import erf
from ngraph.opset9 import exp
from ngraph.opset9 import eye
from ngraph.opset9 import fake_quantize
from ngraph.opset9 import floor
from ngraph.opset9 import floor_mod
from ngraph.opset9 import gather
from ngraph.opset9 import gather_elements
from ngraph.opset9 import gather_nd
from ngraph.opset9 import gather_tree
from ngraph.opset9 import gelu
from ngraph.opset9 import generate_proposals
from ngraph.opset9 import greater
from ngraph.opset9 import greater_equal
from ngraph.opset9 import grid_sample
from ngraph.opset9 import grn
from ngraph.opset9 import group_convolution
from ngraph.opset9 import group_convolution_backprop_data
from ngraph.opset9 import gru_cell
from ngraph.opset9 import gru_sequence
from ngraph.opset9 import hard_sigmoid
from ngraph.opset9 import hsigmoid
from ngraph.opset9 import hswish
from ngraph.opset9 import idft
from ngraph.opset9 import if_op
from ngraph.opset9 import interpolate
from ngraph.opset9 import irdft
from ngraph.opset9 import i420_to_bgr
from ngraph.opset9 import i420_to_rgb
from ngraph.opset9 import less
from ngraph.opset9 import less_equal
from ngraph.opset9 import log
from ngraph.opset9 import logical_and
from ngraph.opset9 import logical_not
from ngraph.opset9 import logical_or
from ngraph.opset9 import logical_xor
from ngraph.opset9 import log_softmax
from ngraph.opset9 import loop
from ngraph.opset9 import lrn
from ngraph.opset9 import lstm_cell
from ngraph.opset9 import lstm_sequence
from ngraph.opset9 import matmul
from ngraph.opset9 import matrix_nms
from ngraph.opset9 import max_pool
from ngraph.opset9 import maximum
from ngraph.opset9 import minimum
from ngraph.opset9 import mish
from ngraph.opset9 import mod
from ngraph.opset9 import multiclass_nms
from ngraph.opset9 import multiply
from ngraph.opset9 import mvn
from ngraph.opset9 import negative
from ngraph.opset9 import non_max_suppression
from ngraph.opset9 import non_zero
from ngraph.opset9 import normalize_l2
from ngraph.opset9 import not_equal
from ngraph.opset9 import nv12_to_bgr
from ngraph.opset9 import nv12_to_rgb
from ngraph.opset9 import one_hot
from ngraph.opset9 import pad
from ngraph.opset9 import parameter
from ngraph.opset9 import power
from ngraph.opset9 import prelu
from ngraph.opset9 import prior_box
from ngraph.opset9 import prior_box_clustered
from ngraph.opset9 import psroi_pooling
from ngraph.opset9 import proposal
from ngraph.opset9 import random_uniform
from ngraph.opset9 import range
from ngraph.opset9 import rdft
from ngraph.opset9 import read_value
from ngraph.opset9 import reduce_l1
from ngraph.opset9 import reduce_l2
from ngraph.opset9 import reduce_logical_and
from ngraph.opset9 import reduce_logical_or
from ngraph.opset9 import reduce_max
from ngraph.opset9 import reduce_mean
from ngraph.opset9 import reduce_min
from ngraph.opset9 import reduce_prod
from ngraph.opset9 import reduce_sum
from ngraph.opset9 import region_yolo
from ngraph.opset9 import reorg_yolo
from ngraph.opset9 import relu
from ngraph.opset9 import reshape
from ngraph.opset9 import result
from ngraph.opset9 import reverse_sequence
from ngraph.opset9 import rnn_cell
from ngraph.opset9 import rnn_sequence
from ngraph.opset9 import roi_align
from ngraph.opset9 import roi_pooling
from ngraph.opset9 import roll
from ngraph.opset9 import round
from ngraph.opset9 import scatter_elements_update
from ngraph.opset9 import scatter_update
from ngraph.opset9 import select
from ngraph.opset9 import selu
from ngraph.opset9 import shape_of
from ngraph.opset9 import shuffle_channels
from ngraph.opset9 import sigmoid
from ngraph.opset9 import sign
from ngraph.opset9 import sin
from ngraph.opset9 import sinh
from ngraph.opset9 import slice
from ngraph.opset9 import softmax
from ngraph.opset9 import softplus
from ngraph.opset9 import softsign
from ngraph.opset9 import space_to_batch
from ngraph.opset9 import space_to_depth
from ngraph.opset9 import split
from ngraph.opset9 import sqrt
from ngraph.opset9 import squared_difference
from ngraph.opset9 import squeeze
from ngraph.opset9 import strided_slice
from ngraph.opset9 import subtract
from ngraph.opset9 import swish
from ngraph.opset9 import tan
from ngraph.opset9 import tanh
from ngraph.opset9 import tensor_iterator
from ngraph.opset9 import tile
from ngraph.opset9 import topk
from ngraph.opset9 import transpose
from ngraph.opset9 import unsqueeze
from ngraph.opset9 import variadic_split


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
