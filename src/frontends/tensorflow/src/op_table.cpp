// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/op/str_ops.hpp"

#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend::tensorflow;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

#define TF_OP_CONVERTER(op) OutputVector op(const ov::frontend::tensorflow::NodeContext& node)

using std::make_shared;

TF_OP_CONVERTER(translate_if_op);
TF_OP_CONVERTER(translate_block_lstm_op);
TF_OP_CONVERTER(translate_gru_block_cell_op);
TF_OP_CONVERTER(translate_partitioned_call_op);
TF_OP_CONVERTER(translate_sparse_fill_empty_rows_op);
TF_OP_CONVERTER(translate_sparse_reshape_op);
TF_OP_CONVERTER(translate_sparse_segment_sum_op);
TF_OP_CONVERTER(translate_while_op);

// Experimental translator for String/Tokenization/Structural Types
TF_OP_CONVERTER(translate_case_fold_utf8_op);
TF_OP_CONVERTER(translate_normalize_utf8_op);

// Save files, put implementations here


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
        {"IsFinite", translate_unary_op<opset10::IsFinite>},
        {"IsInf", translate_unary_op<opset10::IsInf>},
        {"IsNan", translate_unary_op<opset10::IsNaN>},
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
        {"Assert", translate_assert_op},
        {"AvgPool", translate_avg_pool_op},
        {"AvgPool3D", translate_avg_pool_op},
        {"BatchMatMul", translate_batch_mat_mul_op},
        {"BatchMatMulV2", translate_batch_mat_mul_op},
        {"BatchToSpaceND", translate_batch_to_space_nd_op},
        {"BroadcastArgs", translate_broadcast_args_op},
        {"BroadcastTo", translate_broadcast_to_op},
        {"Bucketize", translate_bucketize_op},
        {"BiasAdd", translate_bias_add_op},
        {"Cast", translate_cast_op},
        {"ClipByValue", translate_clip_by_value_op},
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
        {"If", translate_if_op},
        {"input_arg", translate_input_arg_op},
        {"output_arg", translate_output_arg_op},
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
        {"MirrorPad", translate_mirror_pad_op},
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
        {"PadV2", translate_padv2_op},
        {"DynamicStitch", translate_parallel_dynamic_stitch_op},
        {"ParallelDynamicStitch", translate_parallel_dynamic_stitch_op},
        {"PartitionedCall", translate_partitioned_call_op},
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
        {"ReverseV2", translate_reverse_v2_op},
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
        {"SelectV2", translate_select_v2_op},
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
        {"SpaceToBatchND", translate_space_to_batch_nd_op},
        {"StatefulPartitionedCall", translate_partitioned_call_op},
        {"StatelessIf", translate_if_op},
        {"StatelessWhile", translate_while_op},
        {"StridedSlice", translate_strided_slice_op},
        {"Tile", translate_tile_op},
        {"TopK", translate_top_k_op},
        {"TopKV2", translate_top_k_v2_op},
        {"Transpose", translate_transpose_op},
        {"Unpack", translate_unpack_op},
        {"While", translate_while_op},
        {"Where", translate_where_op},
        {"Xdivy", translate_x_div_y_op},
        {"ZerosLike", translate_zeros_like_op},

        // Translators for internal operations
        {"BlockLSTM", translate_block_lstm_op},
        {"GRUBlockCell", translate_gru_block_cell_op},
        {"SparseFillEmptyRows", translate_sparse_fill_empty_rows_op},
        {"SparseSegmentSum", translate_sparse_segment_sum_op},
        {"Unique", translate_unique_op},

        // Experimental translator for String/Tokenization/Structural Types
        {"CaseFoldUTF8", translate_case_fold_utf8_op},
        {"NormalizeUTF8", translate_normalize_utf8_op},

        {"WordpieceTokenizeWithOffsets", [](const NodeContext& node) -> OutputVector {
            return std::make_shared<WordpieceTokenizeWithOffsets>(
                OutputVector{node.get_input(0), node.get_input(1)}
            )->outputs(); }
        },

        {"LookupTableFindV2", [](const NodeContext& node) -> OutputVector {
            return std::make_shared<LookupTableFindV2>(
                OutputVector{node.get_input(0), node.get_input(1), node.get_input(2)}
            )->outputs(); }
        },

        {"StaticRegexReplace", [](const NodeContext& node) -> OutputVector {
            return std::make_shared<StaticRegexReplace>(
                OutputVector{node.get_input(0)},
                node.get_attribute<std::string>("pattern"),
                node.get_attribute<std::string>("rewrite")
            )->outputs(); }},

        {"RegexSplitWithOffsets", [](const NodeContext& node) -> OutputVector {
            return std::make_shared<RegexSplitWithOffsets>(
                OutputVector{node.get_input(0), node.get_input(1), node.get_input(2)}
            )->outputs(); }},

        {"TensorListReserve", [](const NodeContext& node) -> OutputVector {
            // Limitation: known rank of elements
            // Representation consists of 4 tensors: concatenated shapes, element begin indices, element end indices, elements

            auto element_shape = node.get_input(0);
            auto num_elements = std::make_shared<opset10::Reshape>(node.get_input(1), const_value(1, 1), false);

            // known rank of elements implies element_shape has static shape
            TENSORFLOW_OP_VALIDATION(node, element_shape.get_partial_shape().is_static(), "element_shape is not static");
            TENSORFLOW_OP_VALIDATION(node, element_shape.get_shape().size() == 1, "element_shape is not 1D tensor");
            auto element_rank = element_shape.get_shape()[0];
            std::cerr << "[ TF FE INFO ] Element rank = " << element_rank << "\n";

            auto element_type = node.get_attribute<element::Type>("element_dtype");
            auto shape_type = node.get_attribute<element::Type>("shape_type");

            // Form concatenated shapes tensor as zeros of [num_elements, element_rank] shape

            auto shape_shape = std::make_shared<opset10::Concat>(
                OutputVector{num_elements, std::make_shared<opset10::ShapeOf>(element_shape, shape_type)}, 0);

            auto shapes = std::make_shared<opset10::Tile>(const_value(0, 2, shape_type), shape_shape);

            // Use one tensor with zeros for both begins and ends as there are no real element in tensors
            auto indices = std::make_shared<opset10::Tile>(const_value(0, 1, shape_type), num_elements);

            // An empty tensor
            // FIXME: This should be an empty tensor but it breaks transformation flow which improperly over-optimize loop bodies
            // FIXME: That's why a padding in one element is used to keep it not empty. In all other operations this element is ignored
            // FIXME: due to nature of index operations. The only exception is in the operation which turns a list to a tensor,
            // FIXME: there will be an extra StridedSlice to cut off this padding.
            auto elements = opset10::Constant::create(element_type, {1}, {0});

            return make_shared<StructPack>(
                OutputVector{shapes, indices, indices, elements},
                element::StructuralType::TensorListWithRank(element_type, element_rank),
                PartialShape::dynamic())->outputs();
        }},
        {"TensorListFromTensor", [](const NodeContext& node) -> OutputVector {
            using namespace opset10;

            auto tensor = node.get_input(0);
            auto element_shape = node.get_input(1);

            // known rank of elements implies element_shape has static shape
            TENSORFLOW_OP_VALIDATION(node, element_shape.get_partial_shape().is_static(), "element_shape is not static");
            TENSORFLOW_OP_VALIDATION(node, element_shape.get_shape().size() == 1, "element_shape is not 1D tensor");
            auto element_rank = element_shape.get_shape()[0];
            std::cerr << "[ TF FE INFO ] Element rank = " << element_rank << "\n";

            auto element_type = node.get_attribute<element::Type>("element_dtype");
            auto shape_type = node.get_attribute<element::Type>("shape_type");

            auto tensor_shape = make_shared<ShapeOf>(tensor, shape_type);
            //zero_1d = const_value(0, 1, shape_type);
            auto one_1d = const_value(1, 1, shape_type);
            typedef std::vector<int64_t> V;
            auto num_elements = make_shared<StridedSlice>(tensor_shape, one_1d, one_1d, V{1}, V{0});
            auto real_element_shape = make_shared<StridedSlice>(tensor_shape, one_1d, one_1d, V{0}, V{1});

            auto shapes = make_shared<opset10::Tile>(
                real_element_shape, make_shared<Concat>(
                    OutputVector{num_elements, const_value(1, 1, shape_type)}, 0));

            auto total_element_size = make_shared<ReduceProd>(real_element_shape, const_value(0));
            auto num_elements_scalar = make_shared<Squeeze>(num_elements);

            // auto begins = make_shared<SpyOp>(OutputVector{make_shared<Range>(
            //     const_value(0),
            //     make_shared<Multiply>(num_elements_scalar, total_element_size),
            //     total_element_size,
            //     shape_type)});

            // auto ends = make_shared<SpyOp>(OutputVector{make_shared<Range>(
            //     total_element_size,
            //     make_shared<Multiply>(
            //         make_shared<Add>(num_elements_scalar, const_value(1, 0, shape_type)),
            //         total_element_size),
            //     total_element_size,
            //     shape_type)});

            auto begins = make_shared<Range>(
                const_value(0),
                make_shared<Multiply>(num_elements_scalar, total_element_size),
                total_element_size,
                shape_type);

            auto ends = make_shared<Range>(
                total_element_size,
                make_shared<Multiply>(
                    make_shared<Add>(num_elements_scalar, const_value(1, 0, shape_type)),
                    total_element_size),
                total_element_size,
                shape_type);

            auto elements = make_shared<Reshape>(tensor, const_value(-1, 1), true);

            return make_shared<StructPack>(
                OutputVector{shapes, begins, ends, elements},
                element::StructuralType::TensorListWithRank(element_type, element_rank),
                PartialShape::dynamic())->outputs();
        }},
    };
};
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov