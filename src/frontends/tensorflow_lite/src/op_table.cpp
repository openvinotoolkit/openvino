// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset10.hpp"
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

#define DEQUANTIZE_INPUTS_WITH_NAMED_OUTPUTS(func)                               \
    [](const ov::frontend::tensorflow_lite::NodeContext& node) -> OutputVector { \
        auto decoder = node.get_decoder();                                       \
        auto inputs = node.get_inputs();                                         \
        ov::frontend::tensorflow_lite::dequantize_inputs(inputs);                \
        auto context = frontend::tensorflow_lite::NodeContext(decoder, inputs);  \
        return get_indexed_outputs(func(context));                               \
    }

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {
std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"ABS", translate_unary<opset10::Abs>},
        {"ADD", translate_binary_op_with_activation<opset10::Add>},
        {"ADD_N", DEQUANTIZE_INPUTS(translate_add_n_op)},
        {"ARG_MAX", DEQUANTIZE_INPUTS(translate_arg_max_op)},
        {"ARG_MIN", DEQUANTIZE_INPUTS(translate_arg_min_op)},
        // ASSIGN_VARIABLE
        // ATAN2
        {"AVERAGE_POOL_2D", DEQUANTIZE_INPUTS(avg_pool_2d)},
        {"BATCH_MATMUL", DEQUANTIZE_INPUTS(translate_batch_mat_mul_op)},
        {"BATCH_TO_SPACE_ND", DEQUANTIZE_INPUTS(translate_batch_to_space_nd_op)},
        // BIDIRECTIONAL_SEQUENCE_LSTM
        // BIDIRECTIONAL_SEQUENCE_RNN
        {"BROADCAST_ARGS", DEQUANTIZE_INPUTS(translate_broadcast_args_op)},
        {"BROADCAST_TO", DEQUANTIZE_INPUTS(translate_broadcast_to_op)},
        // BUCKETIZE
        // CALL
        // CALL_ONCE
        {"CAST", DEQUANTIZE_INPUTS(translate_cast_op)},
        {"CEIL", translate_unary<opset10::Ceiling>},
        {"COMPLEX_ABS", DEQUANTIZE_INPUTS(complex_abs)},
        // CONCAT_EMBEDDINGS
        {"CONCATENATION", DEQUANTIZE_INPUTS(concatenation)},
        {"CONV_2D", DEQUANTIZE_INPUTS(conv2d)},
        // CONV_3D
        // CONV_3D_TRANSPOSE
        {"COS", translate_unary<opset10::Cos>},
        {"CUMSUM", translate_cumsum_op},
        // CUSTOM
        // DELEGATE
        {"DENSIFY", translate_identity_op},
        {"DEPTH_TO_SPACE", DEQUANTIZE_INPUTS(translate_depth_to_space_op)},
        {"DEPTHWISE_CONV_2D", DEQUANTIZE_INPUTS(depthwise_conv2d)},
        {"DEQUANTIZE", DEQUANTIZE_INPUTS(dequantize)},
        {"DIV", translate_binary_op_with_activation<opset10::Divide>},
        // DYNAMIC_UPDATE_SLICE
        {"ELU", DEQUANTIZE_INPUTS(translate_elu_op)},
        // EMBEDDING_LOOKUP
        // EMBEDDING_LOOKUP_SPARSE
        {"EQUAL", translate_binary<opset10::Equal>},
        {"EXP", translate_unary<opset10::Exp>},
        {"EXPAND_DIMS", DEQUANTIZE_INPUTS(translate_expand_dims_op)},
        // FAKE_QUANT
        {"FILL", DEQUANTIZE_INPUTS(translate_fill_op)},
        {"FLOOR", translate_unary<opset10::Floor>},
        {"FLOOR_DIV", DEQUANTIZE_INPUTS(translate_floor_div_op)},
        {"FLOOR_MOD", translate_binary<opset10::FloorMod>},
        {"FULLY_CONNECTED", DEQUANTIZE_INPUTS(fully_connected)},
        {"GATHER", DEQUANTIZE_INPUTS(gather)},
        {"GATHER_ND", DEQUANTIZE_INPUTS(translate_gather_nd_op)},
        {"GELU", DEQUANTIZE_INPUTS(translate_gelu_op)},
        {"GREATER", translate_binary<opset10::Greater>},
        {"GREATER_EQUAL", translate_binary<opset10::GreaterEqual>},
        {"HARD_SWISH", translate_unary<opset10::HSwish>},
        // HASHTABLE
        // HASHTABLE_FIND
        // HASHTABLE_IMPORT
        // HASHTABLE_LOOKUP
        // HASHTABLE_SIZE
        // IF
        // IMAG
        {"L2_NORMALIZATION", DEQUANTIZE_INPUTS(l2_normalization)},
        // L2_POOL_2D
        {"LEAKY_RELU", DEQUANTIZE_INPUTS(translate_leaky_relu_op)},
        {"LESS", translate_binary<opset10::Less>},
        {"LESS_EQUAL", translate_binary<opset10::LessEqual>},
        // LOCAL_RESPONSE_NORMALIZATION
        {"LOG", translate_unary<opset10::Log>},
        {"LOG_SOFTMAX", DEQUANTIZE_INPUTS(translate_log_softmax_op)},
        {"LOGICAL_AND", translate_binary<opset10::LogicalAnd>},
        {"LOGICAL_NOT", translate_unary<opset10::LogicalNot>},
        {"LOGICAL_OR", translate_binary<opset10::LogicalOr>},
        {"LOGISTIC", translate_unary<opset10::Sigmoid>},
        // LSH_PROJECTION
        // LSTM
        {"MATRIX_DIAG", DEQUANTIZE_INPUTS(translate_matrix_diag_op)},
        // MATRIX_SET_DIAG
        {"MAX_POOL_2D", DEQUANTIZE_INPUTS(max_pool_2d)},
        {"MAXIMUM", translate_binary<opset10::Maximum>},
        {"MEAN", translate_reduce_op<opset10::ReduceMean>},
        {"MINIMUM", translate_binary<opset10::Minimum>},
        {"MIRROR_PAD", DEQUANTIZE_INPUTS(translate_mirror_pad_op)},
        {"MUL", translate_binary_op_with_activation<opset10::Multiply>},
        // MULTINOMIAL
        {"NEG", translate_unary<opset10::Negative>},
        // NON_MAX_SUPPRESSION_V4
        // NON_MAX_SUPPRESSION_V5
        {"NOT_EQUAL", translate_binary<opset10::NotEqual>},
        {"ONE_HOT", DEQUANTIZE_INPUTS(translate_one_hot_op)},
        {"PACK", DEQUANTIZE_INPUTS(translate_pack_op)},
        {"PAD", DEQUANTIZE_INPUTS(translate_pad_op)},
        {"PADV2", DEQUANTIZE_INPUTS(translate_padv2_op)},
        {"POW", translate_binary<opset10::Power>},
        {"PRELU", translate_binary<opset10::PRelu>},
        {"QUANTIZE", quantize},
        // RANDOM_STANDARD_NORMAL
        // RANDOM_UNIFORM
        {"RANGE", DEQUANTIZE_INPUTS(translate_range_op)},
        {"RANK", DEQUANTIZE_INPUTS(translate_rank_op)},
        // READ_VARIABLE
        // REAL
        {"REDUCE_ALL", translate_reduce_op<opset10::ReduceLogicalAnd>},
        {"REDUCE_ANY", translate_reduce_op<opset10::ReduceLogicalOr>},
        {"REDUCE_MAX", translate_reduce_op<opset10::ReduceMax>},
        {"REDUCE_MIN", translate_reduce_op<opset10::ReduceMin>},
        {"REDUCE_PROD", translate_reduce_op<opset10::ReduceProd>},
        {"RELU", translate_unary<opset10::Relu>},
        // RELU_0_TO_1
        // RELU_N1_TO_1
        {"RELU6", DEQUANTIZE_INPUTS(translate_relu_6_op)},
        {"RESHAPE", DEQUANTIZE_INPUTS(reshape)},
        {"RESIZE_BILINEAR", DEQUANTIZE_INPUTS(translate_interpolate_op)},
        {"RESIZE_NEAREST_NEIGHBOR", DEQUANTIZE_INPUTS(translate_interpolate_op)},
        {"REVERSE_SEQUENCE", DEQUANTIZE_INPUTS(translate_reverse_sequence_op)},
        {"REVERSE_V2", DEQUANTIZE_INPUTS(translate_reverse_v2_op)},
        {"RFFT2D", DEQUANTIZE_INPUTS(rfft2d)},
        // RNN
        {"ROUND", DEQUANTIZE_INPUTS(translate_round_op)},
        {"RSQRT", DEQUANTIZE_INPUTS(translate_rsqrt_op)},
        {"SCATTER_ND", DEQUANTIZE_INPUTS(translate_scatter_nd_op)},
        {"SEGMENT_SUM", DEQUANTIZE_INPUTS(translate_segment_sum_op)},
        {"SELECT", DEQUANTIZE_INPUTS(translate_select_op)},
        {"SELECT_V2", DEQUANTIZE_INPUTS(translate_select_v2_op)},
        {"SHAPE", translate_shape_op},
        {"SIGN", translate_unary<opset10::Sign>},
        {"SIN", translate_unary<opset10::Sin>},
        // SKIP_GRAM
        {"SLICE", DEQUANTIZE_INPUTS(translate_slice_op)},
        {"SOFTMAX", DEQUANTIZE_INPUTS(softmax)},
        {"SPACE_TO_BATCH_ND", DEQUANTIZE_INPUTS(translate_space_to_batch_nd_op)},
        {"SPACE_TO_DEPTH", DEQUANTIZE_INPUTS(translate_space_to_depth_op)},
        // SPARSE_TO_DENSE
        {"SPLIT", DEQUANTIZE_INPUTS(translate_split_op)},
        {"SPLIT_V", DEQUANTIZE_INPUTS(translate_split_v_op)},
        {"SQRT", DEQUANTIZE_INPUTS(translate_sqrt_op)},
        {"SQUARE", DEQUANTIZE_INPUTS(translate_square_op)},
        {"SQUARED_DIFFERENCE", translate_binary<opset10::SquaredDifference>},
        {"SQUEEZE", DEQUANTIZE_INPUTS(translate_squeeze_op)},
        {"STRIDED_SLICE", DEQUANTIZE_INPUTS(translate_strided_slice_op)},
        {"SUB", translate_binary_op_with_activation<opset10::Subtract>},
        {"SUM", translate_reduce_op<opset10::ReduceSum>},
        // SVDF
        {"TANH", translate_unary<opset10::Tanh>},
        {"TILE", DEQUANTIZE_INPUTS(translate_tile_op)},
        {"TOPK_V2", DEQUANTIZE_INPUTS_WITH_NAMED_OUTPUTS(translate_top_k_v2_op)},
        {"TRANSPOSE", DEQUANTIZE_INPUTS(translate_transpose_op)},
        {"TRANSPOSE_CONV", DEQUANTIZE_INPUTS(transpose_conv)},
        // UNIDIRECTIONAL_SEQUENCE_LSTM
        // UNIDIRECTIONAL_SEQUENCE_RNN
        {"UNIQUE", DEQUANTIZE_INPUTS(unique)},
        {"UNPACK", DEQUANTIZE_INPUTS(translate_unpack_op)},
        // UNSORTED_SEGMENT_MAX
        // UNSORTED_SEGMENT_MIN
        // UNSORTED_SEGMENT_PROD
        // UNSORTED_SEGMENT_SUM
        // VAR_HANDLE
        {"WHERE", DEQUANTIZE_INPUTS(translate_where_op)},
        {"WHILE", while_op},
        {"ZEROS_LIKE", DEQUANTIZE_INPUTS(translate_zeros_like_op)},
    };
}
}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
