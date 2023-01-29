// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "decoder_map.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;

#define OP_CONVERT_TYPE_RENAME(func, name)                                                                         \
    [](const ov::frontend::tensorflow_lite::NodeContext& node) -> OutputVector {                                   \
        auto decoder = make_shared<DecoderMap>(node.get_decoder(), std::map<std::string, ov::Any>{}, name, false); \
        auto inputs = node.get_inputs();                                                                           \
        auto context = frontend::tensorflow_lite::NodeContext(decoder, inputs);                                    \
        return func(context);                                                                                      \
    }

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"ABS", ov::frontend::tensorflow::op::translate_unary_op<opset8::Abs>},
        {"ADD", translate_binary_op_with_activation<opset10::Add, tflite::AddOptions>},
        {"ADD_N", ov::frontend::tensorflow::op::translate_add_n_op},
        {"ARG_MAX", arg_max},
        {"ARG_MIN", arg_min},
        // ASSIGN_VARIABLE
        // ATAN2
        {"AVERAGE_POOL_2D", avg_pool_2d},
        {"BATCH_MATMUL", batch_matmul},
        {"BATCH_TO_SPACE_ND",
         OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_batch_to_space_nd_op, "BatchToSpaceND")},
        // BIDIRECTIONAL_SEQUENCE_LSTM
        // BIDIRECTIONAL_SEQUENCE_RNN
        {"BROADCAST_ARGS",
         OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_broadcast_args_op, "BroadcastArgs")},
        {"BROADCAST_TO",
         OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_broadcast_to_op, "BroadcastTo")},
        // BUCKETIZE
        // CALL
        // CALL_ONCE
        {"CAST", cast},
        {"CEIL", ov::frontend::tensorflow::op::translate_unary_op<opset8::Ceiling>},
        // COMPLEX_ABS
        // CONCAT_EMBEDDINGS
        {"CONCATENATION", concatenation},
        {"CONV_2D", conv2d},
        // CONV_3D
        // CONV_3D_TRANSPOSE
        {"COS", ov::frontend::tensorflow::op::translate_unary_op<opset8::Cos>},
        // CUMSUM
        // CUSTOM
        // DELEGATE
        // DENSIFY
        {"DEPTH_TO_SPACE", depth_to_space},
        {"DEPTHWISE_CONV_2D", depthwise_conv2d},
        // DEQUANTIZE
        {"DIV", translate_binary_op_with_activation<opset10::Divide, tflite::DivOptions>},
        // DYNAMIC_UPDATE_SLICE
        {"ELU", ov::frontend::tensorflow::op::translate_elu_op},
        // EMBEDDING_LOOKUP
        // EMBEDDING_LOOKUP_SPARSE
        {"EQUAL", ov::frontend::tensorflow::op::translate_binary_op<opset8::Equal>},
        {"EXP", ov::frontend::tensorflow::op::translate_unary_op<opset8::Exp>},
        {"EXPAND_DIMS", ov::frontend::tensorflow::op::translate_expand_dims_op},
        // FAKE_QUANT
        {"FILL", ov::frontend::tensorflow::op::translate_fill_op},
        {"FLOOR", ov::frontend::tensorflow::op::translate_unary_op<opset8::Floor>},
        {"FLOOR_DIV", ov::frontend::tensorflow::op::translate_floor_div_op},
        {"FLOOR_MOD", ov::frontend::tensorflow::op::translate_binary_op<opset8::FloorMod>},
        {"FULLY_CONNECTED", fully_connected},
        {"GATHER", gather},
        {"GATHER_ND", ov::frontend::tensorflow::op::translate_gather_nd_op},
        // GELU
        {"GREATER", ov::frontend::tensorflow::op::translate_binary_op<opset8::Greater>},
        {"GREATER_EQUAL", ov::frontend::tensorflow::op::translate_binary_op<opset8::GreaterEqual>},
        {"HARD_SWISH", ov::frontend::tensorflow::op::translate_unary_op<opset8::HSwish>},
        // HASHTABLE
        // HASHTABLE_FIND
        // HASHTABLE_IMPORT
        // HASHTABLE_LOOKUP
        // HASHTABLE_SIZE
        // IF
        // IMAG
        {"L2_NORMALIZATION", l2_normalization},
        // L2_POOL_2D
        {"LEAKY_RELU", leaky_relu},
        {"LESS", ov::frontend::tensorflow::op::translate_binary_op<opset8::Less>},
        {"LESS_EQUAL", ov::frontend::tensorflow::op::translate_binary_op<opset8::LessEqual>},
        // LOCAL_RESPONSE_NORMALIZATION
        {"LOG", ov::frontend::tensorflow::op::translate_unary_op<opset8::Log>},
        {"LOG_SOFTMAX", ov::frontend::tensorflow::op::translate_log_softmax_op},
        {"LOGICAL_AND", ov::frontend::tensorflow::op::translate_binary_op<opset8::LogicalAnd>},
        {"LOGICAL_NOT", ov::frontend::tensorflow::op::translate_unary_op<opset8::LogicalNot>},
        {"LOGICAL_OR", ov::frontend::tensorflow::op::translate_binary_op<opset8::LogicalOr>},
        {"LOGISTIC", ov::frontend::tensorflow::op::translate_unary_op<opset10::Sigmoid>},
        // LSH_PROJECTION
        // LSTM
        {"MATRIX_DIAG", ov::frontend::tensorflow::op::translate_matrix_diag_op},
        // MATRIX_SET_DIAG
        {"MAX_POOL_2D", max_pool_2d},
        {"MAXIMUM", ov::frontend::tensorflow::op::translate_binary_op<opset8::Maximum>},
        {"MEAN", translate_reduce_op<opset8::ReduceMean>},
        {"MINIMUM", ov::frontend::tensorflow::op::translate_binary_op<opset8::Minimum>},
        {"MIRROR_PAD", mirror_pad},
        {"MUL", translate_binary_op_with_activation<opset10::Multiply, tflite::MulOptions>},
        // MULTINOMIAL
        {"NEG", ov::frontend::tensorflow::op::translate_unary_op<opset8::Negative>},
        // NON_MAX_SUPPRESSION_V4
        // NON_MAX_SUPPRESSION_V5
        {"NOT_EQUAL", ov::frontend::tensorflow::op::translate_binary_op<opset8::NotEqual>},
        {"ONE_HOT", one_hot},
        {"PACK", pack},
        {"PAD", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_pad_op, "Pad")},
        {"PADV2", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_padv2_op, "PadV2")},
        {"POW", ov::frontend::tensorflow::op::translate_binary_op<opset8::Power>},
        {"PRELU", ov::frontend::tensorflow::op::translate_binary_op<opset10::PRelu>},
        // QUANTIZE
        // RANDOM_STANDARD_NORMAL
        // RANDOM_UNIFORM
        {"RANGE", range},
        {"RANK", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_rank_op, "Rank")},
        // READ_VARIABLE
        // REAL
        {"REDUCE_ALL", translate_reduce_op<opset8::ReduceLogicalAnd>},
        {"REDUCE_ANY", translate_reduce_op<opset8::ReduceLogicalOr>},
        {"REDUCE_MAX", translate_reduce_op<opset8::ReduceMax>},
        {"REDUCE_MIN", translate_reduce_op<opset8::ReduceMin>},
        {"REDUCE_PROD", translate_reduce_op<opset8::ReduceProd>},
        {"RELU", ov::frontend::tensorflow::op::translate_unary_op<opset10::Relu>},
        // RELU_0_TO_1
        // RELU_N1_TO_1
        {"RELU6", ov::frontend::tensorflow::op::translate_relu_6_op},
        {"RESHAPE", reshape},
        {"RESIZE_BILINEAR", resize_bilinear},
        {"RESIZE_NEAREST_NEIGHBOR", resize_nearest_neightbor},
        {"REVERSE_SEQUENCE", reverse_sequence},
        {"REVERSE_V2", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_reverse_v2_op, "ReverseV2")},
        // RFFT2D
        // RNN
        {"ROUND", ov::frontend::tensorflow::op::translate_round_op},
        {"RSQRT", ov::frontend::tensorflow::op::translate_rsqrt_op},
        {"SCATTER_ND", ov::frontend::tensorflow::op::translate_scatter_nd_op},
        {"SEGMENT_SUM", ov::frontend::tensorflow::op::translate_segment_sum_op},
        {"SELECT", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_select_op, "Select")},
        {"SELECT_V2", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_select_v2_op, "SelectV2")},
        {"SHAPE", shape},
        {"SIGN", ov::frontend::tensorflow::op::translate_unary_op<opset8::Sign>},
        {"SIN", ov::frontend::tensorflow::op::translate_unary_op<opset8::Sin>},
        // SKIP_GRAM
        {"SLICE", ov::frontend::tensorflow::op::translate_slice_op},
        {"SOFTMAX", softmax},
        {"SPACE_TO_BATCH_ND",
         OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_space_to_batch_nd_op, "SpaceToBatchND")},
        {"SPACE_TO_DEPTH", space_to_depth},
        // SPARSE_TO_DENSE
        {"SPLIT", split},
        {"SPLIT_V", ov::frontend::tensorflow::op::translate_split_v_op},
        {"SQRT", ov::frontend::tensorflow::op::translate_sqrt_op},
        {"SQUARE", ov::frontend::tensorflow::op::translate_square_op},
        {"SQUARED_DIFFERENCE", ov::frontend::tensorflow::op::translate_binary_op<opset8::SquaredDifference>},
        {"SQUEEZE", squeeze},
        {"STRIDED_SLICE", strided_slice},
        {"SUB", translate_binary_op_with_activation<opset10::Subtract, tflite::SubOptions>},
        {"SUM", translate_reduce_op<opset8::ReduceSum>},
        // SVDF
        {"TANH", ov::frontend::tensorflow::op::translate_unary_op<opset8::Tanh>},
        {"TILE", ov::frontend::tensorflow::op::translate_tile_op},
        {"TOPK_V2", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_top_k_v2_op, "TopKV2")},
        {"TRANSPOSE", ov::frontend::tensorflow::op::translate_transpose_op},
        // TRANSPOSE_CONV
        // UNIDIRECTIONAL_SEQUENCE_LSTM
        // UNIDIRECTIONAL_SEQUENCE_RNN
        {"UNIQUE", unique},
        {"UNPACK", unpack},
        // UNSORTED_SEGMENT_MAX
        // UNSORTED_SEGMENT_MIN
        // UNSORTED_SEGMENT_PROD
        // UNSORTED_SEGMENT_SUM
        // VAR_HANDLE
        {"WHERE", OP_CONVERT_TYPE_RENAME(ov::frontend::tensorflow::op::translate_where_op, "Where")},
        // WHILE
        {"ZEROS_LIKE", ov::frontend::tensorflow::op::translate_zeros_like_op},
    };
}
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
