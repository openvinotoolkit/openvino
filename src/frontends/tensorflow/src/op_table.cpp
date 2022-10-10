// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
#define OP_CONVERTER(op) OutputVector op(const NodeContext& node)
#define OP_T_CONVERTER(op) \
    template <class T>     \
    OutputVector op(const NodeContext& node)

OP_T_CONVERTER(translate_unary_op);
OP_T_CONVERTER(translate_binary_op);
OP_T_CONVERTER(translate_direct_reduce_op);

OP_CONVERTER(translate_add_n_op);
OP_CONVERTER(translate_arg_max_op);
OP_CONVERTER(translate_arg_min_op);
OP_CONVERTER(translate_avg_pool_op);
OP_CONVERTER(translate_batch_mat_mul_op);
OP_CONVERTER(translate_batch_nd_and_space_nd_op);
OP_CONVERTER(translate_bias_add_op);
OP_CONVERTER(translate_block_lstm_op);
OP_CONVERTER(translate_broadcast_args_op);
OP_CONVERTER(translate_broadcast_to_op);
OP_CONVERTER(translate_bucketize_op);
OP_CONVERTER(translate_cast_op);
OP_CONVERTER(translate_concat_op);
OP_CONVERTER(translate_const_op);
OP_CONVERTER(translate_conv_2d_op);
OP_CONVERTER(translate_conv_2d_backprop_input_op);
OP_CONVERTER(translate_conv_3d_op);
OP_CONVERTER(translate_conv_3d_backprop_input_v2_op);
OP_CONVERTER(translate_ctc_greedy_decoder_op);
OP_CONVERTER(translate_ctc_loss_op);
OP_CONVERTER(translate_cumsum_op);
OP_CONVERTER(translate_crop_and_resize_op);
OP_CONVERTER(translate_depth_to_space_op);
OP_CONVERTER(translate_depthwise_conv_2d_native_op);
OP_CONVERTER(translate_dynamic_partition_op);
OP_CONVERTER(translate_einsum_op);
OP_CONVERTER(translate_elu_op);
OP_CONVERTER(translate_expand_dims_op);
OP_CONVERTER(translate_extract_image_patches_op);
OP_CONVERTER(translate_fake_quant_op);
OP_CONVERTER(translate_fill_op);
OP_CONVERTER(translate_floor_div_op);
OP_CONVERTER(translate_fused_batch_norm_op);
OP_CONVERTER(translate_gather_op);
OP_CONVERTER(translate_gather_v2_op);
OP_CONVERTER(translate_gather_nd_op);
OP_CONVERTER(translate_gru_block_cell_op);
OP_CONVERTER(translate_identity_op);
OP_CONVERTER(translate_identity_n_op);
OP_CONVERTER(translate_interpolate_op);
OP_CONVERTER(translate_is_finite_op);
OP_CONVERTER(translate_l2_loss_op);
OP_CONVERTER(translate_linspace_op);
OP_CONVERTER(translate_list_diff_op);
OP_CONVERTER(translate_leaky_relu_op);
OP_CONVERTER(translate_log_softmax_op);
OP_CONVERTER(translate_log_1p_op);
OP_CONVERTER(translate_lrn_op);
OP_CONVERTER(translate_mat_mul_op);
OP_CONVERTER(translate_matrix_diag_op);
OP_CONVERTER(translate_max_pool_op);
OP_CONVERTER(translate_non_max_suppression_op);
OP_CONVERTER(translate_normalize_l2_op);
OP_CONVERTER(translate_pad_op);
OP_CONVERTER(translate_placeholder_op);
OP_CONVERTER(translate_placeholder_with_default_op);
OP_CONVERTER(translate_no_op);
OP_CONVERTER(translate_one_hot_op);
OP_CONVERTER(translate_pack_op);
OP_CONVERTER(translate_range_op);
OP_CONVERTER(translate_rank_op);
OP_CONVERTER(translate_random_uniform_op);
OP_CONVERTER(translate_random_uniform_int_op);
OP_CONVERTER(translate_relu_6_op);
OP_CONVERTER(translate_reciprocal_op);
OP_CONVERTER(translate_reshape_op);
OP_CONVERTER(translate_resource_gather_op);
OP_CONVERTER(translate_reverse_op);
OP_CONVERTER(translate_reverse_sequence_op);
OP_CONVERTER(translate_roll_op);
OP_CONVERTER(translate_round_op);
OP_CONVERTER(translate_rsqrt_op);
OP_CONVERTER(translate_scatter_nd_op);
OP_CONVERTER(translate_segment_sum_op);
OP_CONVERTER(translate_sparse_to_dense_op);
OP_CONVERTER(translate_select_op);
OP_CONVERTER(translate_shape_op);
OP_CONVERTER(translate_size_op);
OP_CONVERTER(translate_slice_op);
OP_CONVERTER(translate_softmax_op);
OP_CONVERTER(translate_space_to_depth_op);
OP_CONVERTER(translate_sparse_reshape_op);
OP_CONVERTER(translate_split_op);
OP_CONVERTER(translate_split_v_op);
OP_CONVERTER(translate_square_op);
OP_CONVERTER(translate_squeeze_op);
OP_CONVERTER(translate_strided_slice_op);
OP_CONVERTER(translate_sqrt_op);
OP_CONVERTER(translate_tile_op);
OP_CONVERTER(translate_top_k_op);
OP_CONVERTER(translate_top_k_v2_op);
OP_CONVERTER(translate_transpose_op);
OP_CONVERTER(translate_unpack_op);
OP_CONVERTER(translate_where_op);
OP_CONVERTER(translate_x_div_y_op);
OP_CONVERTER(translate_zeros_like_op);

// Translators for internal operations
OP_CONVERTER(translate_sparse_fill_empty_rows_op);
OP_CONVERTER(translate_sparse_segment_sum_op);
OP_CONVERTER(translate_unique_op);

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        // note: UnaryOp translator declaration for each op must to be added in unary_op.cpp file
        {"Abs", translate_unary_op<opset8::Abs>},
        {"Acos", translate_unary_op<opset8::Acos>},
        {"Acosh", translate_unary_op<opset8::Acosh>},
        {"Asin", translate_unary_op<opset8::Asin>},
        {"Asinh", translate_unary_op<opset8::Asinh>},
        {"Atan", translate_unary_op<opset8::Atan>},
        {"Atanh", translate_unary_op<opset8::Atanh>},
        {"Ceil", translate_unary_op<opset8::Ceiling>},
        {"Cos", translate_unary_op<opset8::Cos>},
        {"Cosh", translate_unary_op<opset8::Cosh>},
        {"Erf", translate_unary_op<opset8::Erf>},
        {"Exp", translate_unary_op<opset8::Exp>},
        {"Floor", translate_unary_op<opset8::Floor>},
        {"Log", translate_unary_op<opset8::Log>},
        {"LogicalNot", translate_unary_op<opset8::LogicalNot>},
        {"Mish", translate_unary_op<opset8::Mish>},
        {"Neg", translate_unary_op<opset8::Negative>},
        {"Relu", translate_unary_op<opset8::Relu>},
        {"Sigmoid", translate_unary_op<opset8::Sigmoid>},
        {"Sin", translate_unary_op<opset8::Sin>},
        {"Sinh", translate_unary_op<opset8::Sinh>},
        {"Sign", translate_unary_op<opset8::Sign>},
        {"Softplus", translate_unary_op<opset8::SoftPlus>},
        {"Softsign", translate_unary_op<opset9::SoftSign>},
        {"Tan", translate_unary_op<opset8::Tan>},
        {"Tanh", translate_unary_op<opset8::Tanh>},
        {"Swish", translate_unary_op<opset8::Swish>},

        // note: BinaryOp translator declaration for each op must to be added in binary_op.cpp file
        {"Add", translate_binary_op<opset8::Add>},
        {"AddV2", translate_binary_op<opset8::Add>},
        {"Equal", translate_binary_op<opset8::Equal>},
        {"FloorMod", translate_binary_op<opset8::FloorMod>},
        {"Greater", translate_binary_op<opset8::Greater>},
        {"GreaterEqual", translate_binary_op<opset8::GreaterEqual>},
        {"Less", translate_binary_op<opset8::Less>},
        {"LessEqual", translate_binary_op<opset8::LessEqual>},
        {"LogicalAnd", translate_binary_op<opset8::LogicalAnd>},
        {"LogicalOr", translate_binary_op<opset8::LogicalOr>},
        {"LogicalXor", translate_binary_op<opset8::LogicalXor>},
        {"Maximum", translate_binary_op<opset8::Maximum>},
        {"Minimum", translate_binary_op<opset8::Minimum>},
        {"Mul", translate_binary_op<opset8::Multiply>},
        {"Mod", translate_binary_op<opset8::Mod>},
        {"NotEqual", translate_binary_op<opset8::NotEqual>},
        {"Pow", translate_binary_op<opset8::Power>},
        {"RealDiv", translate_binary_op<opset8::Divide>},
        {"SquaredDifference", translate_binary_op<opset8::SquaredDifference>},
        {"Sub", translate_binary_op<opset8::Subtract>},

        // note: ReduceOp translator declaration for each op must to be added in reduce.cpp file
        {"Any", translate_direct_reduce_op<opset8::ReduceLogicalOr>},
        {"All", translate_direct_reduce_op<opset8::ReduceLogicalAnd>},
        {"EuclideanNorm", translate_direct_reduce_op<opset8::ReduceL2>},
        {"Max", translate_direct_reduce_op<opset8::ReduceMax>},
        {"Mean", translate_direct_reduce_op<opset8::ReduceMean>},
        {"Min", translate_direct_reduce_op<opset8::ReduceMin>},
        {"Prod", translate_direct_reduce_op<opset8::ReduceProd>},
        {"Sum", translate_direct_reduce_op<opset8::ReduceSum>},

        // Separate translators:
        {"AddN", translate_add_n_op},
        {"ArgMax", translate_arg_max_op},
        {"ArgMin", translate_arg_min_op},
        {"AvgPool", translate_avg_pool_op},
        {"AvgPool3D", translate_avg_pool_op},
        {"BatchMatMul", translate_batch_mat_mul_op},
        {"BatchMatMulV2", translate_batch_mat_mul_op},
        {"BatchToSpaceND", translate_batch_nd_and_space_nd_op},
        {"BroadcastArgs", translate_broadcast_args_op},
        {"BroadcastTo", translate_broadcast_to_op},
        {"Bucketize", translate_bucketize_op},
        {"BiasAdd", translate_bias_add_op},
        {"Cast", translate_cast_op},
        {"Concat", translate_concat_op},
        {"ConcatV2", translate_concat_op},
        {"Const", translate_const_op},
        {"Conv2D", translate_conv_2d_op},
        {"Conv2DBackpropInput", translate_conv_2d_backprop_input_op},
        {"Conv3D", translate_conv_3d_op},
        {"Conv3DBackpropInputV2", translate_conv_3d_backprop_input_v2_op},
        {"CropAndResize", translate_crop_and_resize_op},
        {"CTCGreedyDecoder", translate_ctc_greedy_decoder_op},
        {"CTCLoss", translate_ctc_loss_op},
        {"Cumsum", translate_cumsum_op},
        {"DepthToSpace", translate_depth_to_space_op},
        {"DepthwiseConv2dNative", translate_depthwise_conv_2d_native_op},
        {"DynamicPartition", translate_dynamic_partition_op},
        {"Einsum", translate_einsum_op},
        {"Elu", translate_elu_op},
        {"ExpandDims", translate_expand_dims_op},
        {"ExtractImagePatches", translate_extract_image_patches_op},
        {"FakeQuantWithMinMaxVars", translate_fake_quant_op},
        {"FakeQuantWithMinMaxVarsPerChannel", translate_fake_quant_op},
        {"Fill", translate_fill_op},
        {"FloorDiv", translate_floor_div_op},
        {"FusedBatchNorm", translate_fused_batch_norm_op},
        {"FusedBatchNormV2", translate_fused_batch_norm_op},
        {"FusedBatchNormV3", translate_fused_batch_norm_op},
        {"Gather", translate_gather_op},
        {"GatherV2", translate_gather_v2_op},
        {"GatherNd", translate_gather_nd_op},
        {"Identity", translate_identity_op},
        {"IdentityN", translate_identity_n_op},
        {"IsFinite", translate_is_finite_op},
        {"L2Loss", translate_l2_loss_op},
        {"LeakyRelu", translate_leaky_relu_op},
        {"LinSpace", translate_linspace_op},
        {"ListDiff", translate_list_diff_op},
        {"LogSoftmax", translate_log_softmax_op},
        {"Log1p", translate_log_1p_op},
        {"LRN", translate_lrn_op},
        {"MatMul", translate_mat_mul_op},
        {"MatrixDiag", translate_matrix_diag_op},
        {"MaxPool", translate_max_pool_op},
        {"MaxPoolV2", translate_max_pool_op},
        {"MaxPool3D", translate_max_pool_op},
        {"MirrorPad", translate_pad_op},
        {"NonMaxSuppression", translate_non_max_suppression_op},
        {"NonMaxSuppressionV2", translate_non_max_suppression_op},
        {"NonMaxSuppressionV3", translate_non_max_suppression_op},
        {"NonMaxSuppressionV4", translate_non_max_suppression_op},
        {"NonMaxSuppressionV5", translate_non_max_suppression_op},
        {"NoOp", translate_no_op},  // do nothing
        {"NormalizeL2", translate_normalize_l2_op},
        {"OneHot", translate_one_hot_op},
        {"Pack", translate_pack_op},
        {"Pad", translate_pad_op},
        {"PadV2", translate_pad_op},
        {"Placeholder", translate_placeholder_op},
        {"PlaceholderWithDefault", translate_placeholder_with_default_op},
        {"PreventGradient", translate_identity_op},
        {"Range", translate_range_op},
        {"Rank", translate_rank_op},
        {"RandomUniform", translate_random_uniform_op},
        {"RandomUniformInt", translate_random_uniform_int_op},
        {"Reciprocal", translate_reciprocal_op},
        {"Relu6", translate_relu_6_op},
        {"Reshape", translate_reshape_op},
        {"Reverse", translate_reverse_op},
        {"ReverseSequence", translate_reverse_sequence_op},
        {"ReverseV2", translate_reverse_op},
        {"ResizeBilinear", translate_interpolate_op},
        {"ResizeNearestNeighbor", translate_interpolate_op},
        {"ResourceGather", translate_resource_gather_op},
        {"Roll", translate_roll_op},
        {"Round", translate_round_op},
        {"Rsqrt", translate_rsqrt_op},
        {"ScatterNd", translate_scatter_nd_op},
        {"SegmentSum", translate_segment_sum_op},
        {"SparseToDense", translate_sparse_to_dense_op},
        {"Select", translate_select_op},
        {"SelectV2", translate_select_op},
        {"Shape", translate_shape_op},
        {"Size", translate_size_op},
        {"Slice", translate_slice_op},
        {"Snapshot", translate_identity_op},
        {"Softmax", translate_softmax_op},
        {"SpaceToDepth", translate_space_to_depth_op},
        {"SparseReshape", translate_sparse_reshape_op},
        {"Split", translate_split_op},
        {"SplitV", translate_split_v_op},
        {"StopGradient", translate_identity_op},
        {"Sqrt", translate_sqrt_op},
        {"Square", translate_square_op},
        {"Squeeze", translate_squeeze_op},
        {"SpaceToBatchND", translate_batch_nd_and_space_nd_op},
        {"StridedSlice", translate_strided_slice_op},
        {"Tile", translate_tile_op},
        {"TopK", translate_top_k_op},
        {"TopKV2", translate_top_k_v2_op},
        {"Transpose", translate_transpose_op},
        {"Unpack", translate_unpack_op},
        {"Where", translate_where_op},
        {"Xdivy", translate_x_div_y_op},
        {"ZerosLike", translate_zeros_like_op},

        // Translators for internal operations
        {"BlockLSTM", translate_block_lstm_op},
        {"GRUBlockCell", translate_gru_block_cell_op},
        {"SparseFillEmptyRows", translate_sparse_fill_empty_rows_op},
        {"SparseSegmentSum", translate_sparse_segment_sum_op},
        {"Unique", translate_unique_op},
    };
};
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov