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
from ngraph.opset10 import absolute
from ngraph.opset10 import absolute as abs
from ngraph.opset10 import acos
from ngraph.opset10 import acosh
from ngraph.opset10 import adaptive_avg_pool
from ngraph.opset10 import adaptive_max_pool
from ngraph.opset10 import add
from ngraph.opset10 import asin
from ngraph.opset10 import asinh
from ngraph.opset10 import assign
from ngraph.opset10 import atan
from ngraph.opset10 import atanh
from ngraph.opset10 import avg_pool
from ngraph.opset10 import batch_norm_inference
from ngraph.opset10 import batch_to_space
from ngraph.opset10 import binary_convolution
from ngraph.opset10 import broadcast
from ngraph.opset10 import bucketize
from ngraph.opset10 import ceiling
from ngraph.opset10 import ceiling as ceil
from ngraph.opset10 import clamp
from ngraph.opset10 import concat
from ngraph.opset10 import constant
from ngraph.opset10 import convert
from ngraph.opset10 import convert_like
from ngraph.opset10 import convolution
from ngraph.opset10 import convolution_backprop_data
from ngraph.opset10 import cos
from ngraph.opset10 import cosh
from ngraph.opset10 import ctc_greedy_decoder
from ngraph.opset10 import ctc_greedy_decoder_seq_len
from ngraph.opset10 import ctc_loss
from ngraph.opset10 import cum_sum
from ngraph.opset10 import cum_sum as cumsum
from ngraph.opset10 import deformable_convolution
from ngraph.opset10 import deformable_psroi_pooling
from ngraph.opset10 import depth_to_space
from ngraph.opset10 import detection_output
from ngraph.opset10 import dft
from ngraph.opset10 import divide
from ngraph.opset10 import einsum
from ngraph.opset10 import elu
from ngraph.opset10 import embedding_bag_offsets_sum
from ngraph.opset10 import embedding_bag_packed_sum
from ngraph.opset10 import embedding_segments_sum
from ngraph.opset10 import extract_image_patches
from ngraph.opset10 import equal
from ngraph.opset10 import erf
from ngraph.opset10 import exp
from ngraph.opset10 import eye
from ngraph.opset10 import fake_quantize
from ngraph.opset10 import floor
from ngraph.opset10 import floor_mod
from ngraph.opset10 import gather
from ngraph.opset10 import gather_elements
from ngraph.opset10 import gather_nd
from ngraph.opset10 import gather_tree
from ngraph.opset10 import gelu
from ngraph.opset10 import generate_proposals
from ngraph.opset10 import greater
from ngraph.opset10 import greater_equal
from ngraph.opset10 import grid_sample
from ngraph.opset10 import grn
from ngraph.opset10 import group_convolution
from ngraph.opset10 import group_convolution_backprop_data
from ngraph.opset10 import gru_cell
from ngraph.opset10 import gru_sequence
from ngraph.opset10 import hard_sigmoid
from ngraph.opset10 import hsigmoid
from ngraph.opset10 import hswish
from ngraph.opset10 import idft
from ngraph.opset10 import if_op
from ngraph.opset10 import interpolate
from ngraph.opset10 import irdft
from ngraph.opset10 import i420_to_bgr
from ngraph.opset10 import i420_to_rgb
from ngraph.opset10 import less
from ngraph.opset10 import less_equal
from ngraph.opset10 import log
from ngraph.opset10 import logical_and
from ngraph.opset10 import logical_not
from ngraph.opset10 import logical_or
from ngraph.opset10 import logical_xor
from ngraph.opset10 import log_softmax
from ngraph.opset10 import loop
from ngraph.opset10 import lrn
from ngraph.opset10 import lstm_cell
from ngraph.opset10 import lstm_sequence
from ngraph.opset10 import matmul
from ngraph.opset10 import matrix_nms
from ngraph.opset10 import max_pool
from ngraph.opset10 import maximum
from ngraph.opset10 import minimum
from ngraph.opset10 import mish
from ngraph.opset10 import mod
from ngraph.opset10 import multiclass_nms
from ngraph.opset10 import multiply
from ngraph.opset10 import mvn
from ngraph.opset10 import negative
from ngraph.opset10 import non_max_suppression
from ngraph.opset10 import non_zero
from ngraph.opset10 import normalize_l2
from ngraph.opset10 import not_equal
from ngraph.opset10 import nv12_to_bgr
from ngraph.opset10 import nv12_to_rgb
from ngraph.opset10 import one_hot
from ngraph.opset10 import pad
from ngraph.opset10 import parameter
from ngraph.opset10 import power
from ngraph.opset10 import prelu
from ngraph.opset10 import prior_box
from ngraph.opset10 import prior_box_clustered
from ngraph.opset10 import psroi_pooling
from ngraph.opset10 import proposal
from ngraph.opset10 import random_uniform
from ngraph.opset10 import range
from ngraph.opset10 import rdft
from ngraph.opset10 import read_value
from ngraph.opset10 import reduce_l1
from ngraph.opset10 import reduce_l2
from ngraph.opset10 import reduce_logical_and
from ngraph.opset10 import reduce_logical_or
from ngraph.opset10 import reduce_max
from ngraph.opset10 import reduce_mean
from ngraph.opset10 import reduce_min
from ngraph.opset10 import reduce_prod
from ngraph.opset10 import reduce_sum
from ngraph.opset10 import region_yolo
from ngraph.opset10 import reorg_yolo
from ngraph.opset10 import relu
from ngraph.opset10 import reshape
from ngraph.opset10 import result
from ngraph.opset10 import reverse_sequence
from ngraph.opset10 import rnn_cell
from ngraph.opset10 import rnn_sequence
from ngraph.opset10 import roi_align
from ngraph.opset10 import roi_pooling
from ngraph.opset10 import roll
from ngraph.opset10 import round
from ngraph.opset10 import scatter_elements_update
from ngraph.opset10 import scatter_update
from ngraph.opset10 import select
from ngraph.opset10 import selu
from ngraph.opset10 import shape_of
from ngraph.opset10 import shuffle_channels
from ngraph.opset10 import sigmoid
from ngraph.opset10 import sign
from ngraph.opset10 import sin
from ngraph.opset10 import sinh
from ngraph.opset10 import slice
from ngraph.opset10 import softmax
from ngraph.opset10 import softplus
from ngraph.opset10 import softsign
from ngraph.opset10 import space_to_batch
from ngraph.opset10 import space_to_depth
from ngraph.opset10 import split
from ngraph.opset10 import sqrt
from ngraph.opset10 import squared_difference
from ngraph.opset10 import squeeze
from ngraph.opset10 import strided_slice
from ngraph.opset10 import subtract
from ngraph.opset10 import swish
from ngraph.opset10 import tan
from ngraph.opset10 import tanh
from ngraph.opset10 import tensor_iterator
from ngraph.opset10 import tile
from ngraph.opset10 import topk
from ngraph.opset10 import transpose
from ngraph.opset10 import unsqueeze
from ngraph.opset10 import variadic_split


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
