// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "decoder_map.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow::op;

#define DEQUANTIZE_INPUTS(func)                                                     \
    [](const ov::frontend::tensorflow_lite::NodeContext& node) -> OutputVector {    \
        auto decoder = node.get_decoder();                                          \
        auto inputs = node.get_inputs();                                            \
        ov::frontend::tensorflow_lite::dequantize_inputs(inputs);                   \
        auto context = ov::frontend::tensorflow_lite::NodeContext(decoder, inputs); \
        return func(context);                                                       \
    }

#define OP_CONVERT_TYPE_RENAME(func, name)                                                                         \
    [](const ov::frontend::tensorflow_lite::NodeContext& node) -> OutputVector {                                   \
        auto decoder = make_shared<DecoderMap>(node.get_decoder(), std::map<std::string, ov::Any>{}, name, false); \
        auto inputs = node.get_inputs();                                                                           \
        ov::frontend::tensorflow_lite::dequantize_inputs(inputs);                                                  \
        auto context = frontend::tensorflow_lite::NodeContext(decoder, inputs);                                    \
        return get_indexed_outputs(func(context));                                                                 \
    }

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"ABS", translate_unary<opset8::Abs>},
        {"ADD", translate_binary_op_with_activation<opset10::Add, tflite::AddOptions>},
        {"ADD_N", DEQUANTIZE_INPUTS(translate_add_n_op)},
        {"ARG_MAX", DEQUANTIZE_INPUTS(arg_max)},
        {"ARG_MIN", DEQUANTIZE_INPUTS(arg_min)},
        // ASSIGN_VARIABLE
        // ATAN2
        {"AVERAGE_POOL_2D", DEQUANTIZE_INPUTS(avg_pool_2d)},
        {"BATCH_MATMUL", DEQUANTIZE_INPUTS(batch_matmul)},
        {"BATCH_TO_SPACE_ND", OP_CONVERT_TYPE_RENAME(translate_batch_to_space_nd_op, "BatchToSpaceND")},
        // BIDIRECTIONAL_SEQUENCE_LSTM
        // BIDIRECTIONAL_SEQUENCE_RNN
        {"BROADCAST_ARGS", OP_CONVERT_TYPE_RENAME(translate_broadcast_args_op, "BroadcastArgs")},
        {"BROADCAST_TO", OP_CONVERT_TYPE_RENAME(translate_broadcast_to_op, "BroadcastTo")},
        // BUCKETIZE
        // CALL
        // CALL_ONCE
        {"CAST", DEQUANTIZE_INPUTS(cast)},
        {"CEIL", translate_unary<opset8::Ceiling>},
        {"COMPLEX_ABS", DEQUANTIZE_INPUTS(complex_abs)},
        // CONCAT_EMBEDDINGS
        {"CONCATENATION", DEQUANTIZE_INPUTS(concatenation)},
        {"CONV_2D", DEQUANTIZE_INPUTS(conv2d)},
        // CONV_3D
        // CONV_3D_TRANSPOSE
        {"COS", translate_unary<opset8::Cos>},
        // CUMSUM
        // CUSTOM
        // DELEGATE
        {"DENSIFY", translate_identity_op},
        {"DEPTH_TO_SPACE", DEQUANTIZE_INPUTS(depth_to_space)},
        {"DEPTHWISE_CONV_2D", DEQUANTIZE_INPUTS(depthwise_conv2d)},
        {"DEQUANTIZE", DEQUANTIZE_INPUTS(dequantize)},
        {"DIV", translate_binary_op_with_activation<opset10::Divide, tflite::DivOptions>},
        // DYNAMIC_UPDATE_SLICE
        {"ELU", DEQUANTIZE_INPUTS(translate_elu_op)},
        // EMBEDDING_LOOKUP
        // EMBEDDING_LOOKUP_SPARSE
        {"EQUAL", translate_binary<opset8::Equal>},
        {"EXP", translate_unary<opset8::Exp>},
        {"EXPAND_DIMS", OP_CONVERT_TYPE_RENAME(translate_expand_dims_op, "ExpandDims")},
        // FAKE_QUANT
        {"FILL", DEQUANTIZE_INPUTS(translate_fill_op)},
        {"FLOOR", translate_unary<opset8::Floor>},
        {"FLOOR_DIV", DEQUANTIZE_INPUTS(translate_floor_div_op)},
        {"FLOOR_MOD", translate_binary<opset8::FloorMod>},
        {"FULLY_CONNECTED", DEQUANTIZE_INPUTS(fully_connected)},
        {"GATHER", DEQUANTIZE_INPUTS(gather)},
        {"GATHER_ND", DEQUANTIZE_INPUTS(translate_gather_nd_op)},
        // GELU
        {"GREATER", translate_binary<opset8::Greater>},
        {"GREATER_EQUAL", translate_binary<opset8::GreaterEqual>},
        {"HARD_SWISH", translate_unary<opset8::HSwish>},
        // HASHTABLE
        // HASHTABLE_FIND
        // HASHTABLE_IMPORT
        // HASHTABLE_LOOKUP
        // HASHTABLE_SIZE
        // IF
        // IMAG
        {"L2_NORMALIZATION", DEQUANTIZE_INPUTS(l2_normalization)},
        // L2_POOL_2D
        {"LEAKY_RELU", DEQUANTIZE_INPUTS(leaky_relu)},
        {"LESS", translate_binary<opset8::Less>},
        {"LESS_EQUAL", translate_binary<opset8::LessEqual>},
        // LOCAL_RESPONSE_NORMALIZATION
        {"LOG", translate_unary<opset8::Log>},
        {"LOG_SOFTMAX", DEQUANTIZE_INPUTS(translate_log_softmax_op)},
        {"LOGICAL_AND", translate_binary<opset8::LogicalAnd>},
        {"LOGICAL_NOT", translate_unary<opset8::LogicalNot>},
        {"LOGICAL_OR", translate_binary<opset8::LogicalOr>},
        {"LOGISTIC", translate_unary<opset10::Sigmoid>},
        // LSH_PROJECTION
        // LSTM
        {"MATRIX_DIAG", DEQUANTIZE_INPUTS(translate_matrix_diag_op)},
        // MATRIX_SET_DIAG
        {"MAX_POOL_2D", DEQUANTIZE_INPUTS(max_pool_2d)},
        {"MAXIMUM", translate_binary<opset8::Maximum>},
        {"MEAN", translate_reduce_op<opset8::ReduceMean>},
        {"MINIMUM", translate_binary<opset8::Minimum>},
        {"MIRROR_PAD", DEQUANTIZE_INPUTS(mirror_pad)},
        {"MUL", translate_binary_op_with_activation<opset10::Multiply, tflite::MulOptions>},
        // MULTINOMIAL
        {"NEG", translate_unary<opset8::Negative>},
        // NON_MAX_SUPPRESSION_V4
        // NON_MAX_SUPPRESSION_V5
        {"NOT_EQUAL", translate_binary<opset8::NotEqual>},
        {"ONE_HOT", DEQUANTIZE_INPUTS(one_hot)},
        {"PACK", DEQUANTIZE_INPUTS(pack)},
        {"PAD", OP_CONVERT_TYPE_RENAME(translate_pad_op, "Pad")},
        {"PADV2", OP_CONVERT_TYPE_RENAME(translate_padv2_op, "PadV2")},
        {"POW", translate_binary<opset8::Power>},
        {"PRELU", translate_binary<opset10::PRelu>},
        {"QUANTIZE", quantize},
        // RANDOM_STANDARD_NORMAL
        // RANDOM_UNIFORM
        {"RANGE", DEQUANTIZE_INPUTS(range)},
        {"RANK", OP_CONVERT_TYPE_RENAME(translate_rank_op, "Rank")},
        // READ_VARIABLE
        // REAL
        {"REDUCE_ALL", translate_reduce_op<opset8::ReduceLogicalAnd>},
        {"REDUCE_ANY", translate_reduce_op<opset8::ReduceLogicalOr>},
        {"REDUCE_MAX", translate_reduce_op<opset8::ReduceMax>},
        {"REDUCE_MIN", translate_reduce_op<opset8::ReduceMin>},
        {"REDUCE_PROD", translate_reduce_op<opset8::ReduceProd>},
        {"RELU", translate_unary<opset10::Relu>},
        // RELU_0_TO_1
        // RELU_N1_TO_1
        {"RELU6", DEQUANTIZE_INPUTS(translate_relu_6_op)},
        {"RESHAPE", DEQUANTIZE_INPUTS(reshape)},
        {"RESIZE_BILINEAR", DEQUANTIZE_INPUTS(resize_bilinear)},
        {"RESIZE_NEAREST_NEIGHBOR", DEQUANTIZE_INPUTS(resize_nearest_neightbor)},
        {"REVERSE_SEQUENCE", DEQUANTIZE_INPUTS(reverse_sequence)},
        {"REVERSE_V2", OP_CONVERT_TYPE_RENAME(translate_reverse_v2_op, "ReverseV2")},
        {"RFFT2D", DEQUANTIZE_INPUTS(rfft2d)},
        // RNN
        {"ROUND", DEQUANTIZE_INPUTS(translate_round_op)},
        {"RSQRT", DEQUANTIZE_INPUTS(translate_rsqrt_op)},
        {"SCATTER_ND", DEQUANTIZE_INPUTS(translate_scatter_nd_op)},
        {"SEGMENT_SUM", OP_CONVERT_TYPE_RENAME(translate_segment_sum_op, "SegmentSum")},
        {"SELECT", OP_CONVERT_TYPE_RENAME(translate_select_op, "Select")},
        {"SELECT_V2", OP_CONVERT_TYPE_RENAME(translate_select_v2_op, "SelectV2")},
        {"SHAPE", shape},
        {"SIGN", translate_unary<opset8::Sign>},
        {"SIN", translate_unary<opset8::Sin>},
        // SKIP_GRAM
        {"SLICE", OP_CONVERT_TYPE_RENAME(translate_slice_op, "Slice")},
        {"SOFTMAX", DEQUANTIZE_INPUTS(softmax)},
        {"SPACE_TO_BATCH_ND", OP_CONVERT_TYPE_RENAME(translate_space_to_batch_nd_op, "SpaceToBatchND")},
        {"SPACE_TO_DEPTH", DEQUANTIZE_INPUTS(space_to_depth)},
        // SPARSE_TO_DENSE
        {"SPLIT", DEQUANTIZE_INPUTS(split)},
        {"SPLIT_V", DEQUANTIZE_INPUTS(translate_split_v_op)},
        {"SQRT", DEQUANTIZE_INPUTS(translate_sqrt_op)},
        {"SQUARE", DEQUANTIZE_INPUTS(translate_square_op)},
        {"SQUARED_DIFFERENCE", translate_binary<opset8::SquaredDifference>},
        {"SQUEEZE", DEQUANTIZE_INPUTS(squeeze)},
        {"STRIDED_SLICE", DEQUANTIZE_INPUTS(strided_slice)},
        {"SUB", translate_binary_op_with_activation<opset10::Subtract, tflite::SubOptions>},
        {"SUM", translate_reduce_op<opset8::ReduceSum>},
        // SVDF
        {"TANH", translate_unary<opset8::Tanh>},
        {"TILE", DEQUANTIZE_INPUTS(translate_tile_op)},
        {"TOPK_V2", OP_CONVERT_TYPE_RENAME(translate_top_k_v2_op, "TopKV2")},
        {"TRANSPOSE", DEQUANTIZE_INPUTS(translate_transpose_op)},
        {"TRANSPOSE_CONV", DEQUANTIZE_INPUTS(transpose_conv)},
        // UNIDIRECTIONAL_SEQUENCE_LSTM
        // UNIDIRECTIONAL_SEQUENCE_RNN
        {"UNIQUE", DEQUANTIZE_INPUTS(unique)},
        {"UNPACK", DEQUANTIZE_INPUTS(unpack)},
        // UNSORTED_SEGMENT_MAX
        // UNSORTED_SEGMENT_MIN
        // UNSORTED_SEGMENT_PROD
        // UNSORTED_SEGMENT_SUM
        // VAR_HANDLE
        {"WHERE", OP_CONVERT_TYPE_RENAME(translate_where_op, "Where")},
        {"WHILE", while_op},
        {"ZEROS_LIKE", DEQUANTIZE_INPUTS(translate_zeros_like_op)},
    };
}
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
