// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moc_transformations.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/adaptive_pool_to_reduce.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/align_eltwise_input_ranks.hpp"
#include "transformations/common_optimizations/batch_to_space_fusion.hpp"
#include "transformations/common_optimizations/binarize_weights.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/broadcast_transition.hpp"
#include "transformations/common_optimizations/clamp_fusion.hpp"
#include "transformations/common_optimizations/concat_to_broadcast.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/conv_to_binary_conv.hpp"
#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/convert_u4_weights_zero_point_to_scalar.hpp"
#include "transformations/common_optimizations/convolution_to_group_convolution_fusion.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/common_optimizations/dilated_convolution_converter.hpp"
#include "transformations/common_optimizations/disable_random_uniform_constant_folding.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"
#include "transformations/common_optimizations/divide_fusion.hpp"
#include "transformations/common_optimizations/eliminate_duplicate_ti_inputs.hpp"
#include "transformations/common_optimizations/eliminate_loop_inputs_outputs.hpp"
#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"
#include "transformations/common_optimizations/fold_subgraph_empty_inputs.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/gelu_fusion.hpp"
#include "transformations/common_optimizations/gru_cell_fusion.hpp"
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/common_optimizations/leaky_relu_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/lstm_cell_fusion.hpp"
#include "transformations/common_optimizations/matmul_const_transposes_extraction.hpp"
#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp"
#include "transformations/common_optimizations/nonzero_horizontal_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/normalize_l2_fusion.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/common_optimizations/prelu_fusion.hpp"
#include "transformations/common_optimizations/pull_through_reduce.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"
#include "transformations/common_optimizations/random_uniform_fusion.hpp"
#include "transformations/common_optimizations/reduce_reshape_fusion.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/common_optimizations/ric_fusion.hpp"
#include "transformations/common_optimizations/sdpa_fusion.hpp"
#include "transformations/common_optimizations/select_with_one_value_condition.hpp"
#include "transformations/common_optimizations/sequence_fusion.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/common_optimizations/softplus_to_mish_fusion.hpp"
#include "transformations/common_optimizations/space_to_batch_fusion.hpp"
#include "transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp"
#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"
#include "transformations/common_optimizations/subtract_fusion.hpp"
#include "transformations/common_optimizations/swish_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/low_precision/mark_dequantization_subgraph.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/convert_convertpromotetypes.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_negative.hpp"
#include "transformations/op_conversions/convert_scatter_elements_to_scatter.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/convert_ti_to_sequences.hpp"
#include "transformations/resolve_names_collisions.hpp"
#include "transformations/smart_reshape/lstm_states_broadcast.hpp"
#include "transformations/smart_reshape/matmul_sr.hpp"
#include "transformations/smart_reshape/reshape_sinking.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"

static ov::PartialShape prepare_dynamic_shape(const ov::PartialShape& shape) {
    auto new_shape = ov::PartialShape::dynamic(shape.rank());
    if (shape.rank().is_static())
        for (size_t i = 0; i < shape.size(); ++i)
            new_shape[i].set_symbol(shape[i].get_symbol());
    return new_shape;
}

bool ov::pass::MOCTransformations::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MOCTransformations);
    // To avoid issues with dynamism we make ov::Model dynamic and after we apply all
    // transformations we restore original shapes to the ov::Model back
    std::unordered_map<ov::op::v0::Parameter*, PartialShape> input_shapes;
    std::unordered_map<ov::op::util::Variable*, PartialShape> variable_shapes;
    if (!m_use_shapes) {
        for (auto&& param : f->get_parameters()) {
            input_shapes[param.get()] = param->get_partial_shape();
            param->set_partial_shape(prepare_dynamic_shape(param->get_partial_shape()));
        }
        // After setting dynamic ranks into Parameters, the initializing subgraph of ReadValue operation might
        // also have a dynamic rank. The shape consistency check between this subgraph and Variable might fail.
        // We have to set dynamic rank to Variables to keep the ov::Model consistent.
        for (const auto& variable : f->get_variables()) {
            const auto& var_info = variable->get_info();
            variable_shapes[variable.get()] = var_info.data_shape;
            variable->update_data_shape(prepare_dynamic_shape(var_info.data_shape));
        }
        f->validate_nodes_and_infer_types();
    }

    ov::pass::Manager manager(get_pass_config(), "MOC");
    manager.set_per_pass_validation(false);
    using namespace ov::pass;
    REGISTER_PASS(manager, InitNodeInfo)
    if (m_low_precision_enabled) {
        manager.register_pass<ov::pass::MarkDequantization>(element::TypeVector{ov::element::i8,
                                                                                ov::element::u8,
                                                                                ov::element::i4,
                                                                                ov::element::u4,
                                                                                ov::element::nf4,
                                                                                ov::element::f4e2m1,
                                                                                ov::element::f8e4m3,
                                                                                ov::element::f8e5m2,
                                                                                ov::element::f8e8m0});
    }
    if (!m_use_shapes) {
        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
    }
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParamsResults
    // should be performed before first ConstantFolding call.
    // The passes can deteach graph branches where zero dimesion is calculated.
    // Zero dimensions in shape causes creation empty tensors, which are incorrect during CF.
    // In particular, if zero dim tensor is consumed in body of MultiSubGraphOp
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParamsResults should be called together.
    using namespace ov::pass;
    REGISTER_PASS(manager, EliminateConvert)
    REGISTER_PASS(manager, EliminateScatterUpdate)
    REGISTER_PASS(manager, RemoveConcatZeroDimInput)
    REGISTER_PASS(manager, EliminateLoopInputsOutputs);
    REGISTER_PASS(manager, Validate)
    // todo: ticket 96960
    // the order EliminateDuplicateTIInputs and RemoveMultiSubGraphOpDanglingParamsResults is important
    // it looks like we need to combine these transformations into one.
    REGISTER_PASS(manager, EliminateDuplicateTIInputs);
    REGISTER_PASS(manager, RemoveMultiSubGraphOpDanglingParamsResults)
    REGISTER_PASS(manager, FoldSubgraphEmptyInputs)
    REGISTER_PASS(manager, DisableRandomUniformConstantFolding)
    REGISTER_PASS(manager, PushConstantToSubgraph)
    REGISTER_PASS(manager, ConstantFolding)
    REGISTER_PASS(manager, Validate)

    // FusedFilteringBoxesBySize transformation has the complex pattern
    // which can be affected by further transformations. So we have to
    // execute it at the beginning of the pipeline. Also, this pass resolves
    // dynamism, so we have to execute type/shape propagation after.
    REGISTER_PASS(manager, FuseFilteringBoxesBySize)
    REGISTER_PASS(manager, Validate)

    if (!m_use_shapes) {  // Approved Smart Reshape
        REGISTER_PASS(manager, LSTMStatesBroadcast)
        REGISTER_PASS(manager, Validate)
        REGISTER_PASS(manager, ReshapeSinkingMatMul)
        REGISTER_PASS(manager, Validate)
    }
    REGISTER_PASS(manager, ConvertQuantizeDequantize)
    REGISTER_PASS(manager, SimplifyShapeOfSubGraph, m_use_shapes)

    if (!m_use_shapes) {
        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
    }
    // workaround until dynamism in NMS is not supported
    REGISTER_PASS(manager, ConvertNmsGatherPathToUnsigned)
    REGISTER_PASS(manager, StridedSliceOptimization, m_use_shapes)
    REGISTER_PASS(manager, BroadcastElementwiseFusion)
    REGISTER_PASS(manager, BroadcastTransition)
    REGISTER_PASS(manager, PullThroughReduce)

    // GRUCellFusion and SequenceFusion should be before NopElimination
    REGISTER_PASS(manager, LSTMCellFusion)
    REGISTER_PASS(manager, GRUCellFusion)
    REGISTER_PASS(manager, SequenceFusion)

    REGISTER_PASS(manager, ConcatToBroadcast);

    auto transpose_sinking = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(transpose_sinking, TransposeSinking)
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    ADD_MATCHER(transpose_sinking, SplitSqueezeConcatFusion, m_use_shapes)

    REGISTER_PASS(manager, TransposeMatMul)

    auto eliminations = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(eliminations, EliminateUnsqueezeGather)
    ADD_MATCHER(eliminations, NopElimination, m_use_shapes)
    ADD_MATCHER(eliminations, SelectWithOneValueCondition)
    eliminations->set_name("ov::pass::CommonEliminations");

    manager.register_pass<ov::pass::ConstantFolding>();

    auto common_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(common_fusions, ConvertScatterElementsToScatter)
    ADD_MATCHER(common_fusions, SoftPlusFusion)
    ADD_MATCHER(common_fusions, SoftPlusToMishFusion)
    ADD_MATCHER(common_fusions, SwishFusion)
    ADD_MATCHER(common_fusions, HSwishFusion)
    ADD_MATCHER(common_fusions, HSigmoidFusion)
    ADD_MATCHER(common_fusions, NormalizeL2Fusion)
    ADD_MATCHER(common_fusions, ClampFusion)
    ADD_MATCHER(common_fusions, PadFusion)
    ADD_MATCHER(common_fusions, SoftmaxFusion)
    ADD_MATCHER(common_fusions, ReduceReshapeFusion)
    ADD_MATCHER(common_fusions, MVNFusion)
    ADD_MATCHER(common_fusions, DilatedConvolutionConverter)
    ADD_MATCHER(common_fusions, GeluFusion)
    ADD_MATCHER(common_fusions, LeakyReluFusion)
    ADD_MATCHER(common_fusions, RandomUniformFusion)
    ADD_MATCHER(common_fusions, ConvertTensorIteratorToSequence)
    ADD_MATCHER(common_fusions, SplitConcatPairToInterpolateFusion, m_use_shapes)
    ADD_MATCHER(common_fusions, ConvolutionToGroupConvolutionFusion)
    ADD_MATCHER(common_fusions, SDPAFusion)
    if (m_use_shapes) {
        ADD_MATCHER(common_fusions, NearestNeighborUpsamplingFusion)
    }

    ADD_MATCHER(common_fusions, DivideFusion)
    ADD_MATCHER(common_fusions, SubtractFusion)
    ADD_MATCHER(common_fusions, TransposeToReshape)
    ADD_MATCHER(common_fusions, ReshapeSequenceFusion, m_use_shapes)
    ADD_MATCHER(common_fusions, MatMulConstTransposesExtraction)
    ADD_MATCHER(common_fusions, PReluFusion)
    ADD_MATCHER(common_fusions, DepthToSpaceFusion)
    ADD_MATCHER(common_fusions, ShuffleChannelsFusion, !m_use_shapes)
    ADD_MATCHER(common_fusions, NonZeroHorizontalFusion)
    ADD_MATCHER(common_fusions, AdaptivePoolToReduce)
    ADD_MATCHER(common_fusions, ConvertU4WeightsZeroPointToScalar)
    common_fusions->set_name("ov::pass::CommonFusions");

    REGISTER_PASS(manager, BinarizeWeights)
    REGISTER_PASS(manager, ConvToBinaryConv)

    auto decomp = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(decomp, BatchNormDecomposition)
    ADD_MATCHER(decomp, ConvertDivideWithConstant)
    ADD_MATCHER(decomp, ConvertSubtractWithConstant)
    ADD_MATCHER(decomp, ConvertNegative)
    ADD_MATCHER(decomp, ConvertConvertPromoteTypes)
    manager.register_pass<ov::pass::LinOpSequenceFusion>();

    auto multiply_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(multiply_fusions, ConvolutionMultiplyFusion)
    ADD_MATCHER(multiply_fusions, GroupConvolutionMultiplyFusion)
    ADD_MATCHER(multiply_fusions, ConvolutionBackpropDataMultiplyFusion)
    ADD_MATCHER(multiply_fusions, GroupConvolutionBackpropDataMultiplyFusion)
    ADD_MATCHER(multiply_fusions, MultiplyConvolutionFusion)
    ADD_MATCHER(multiply_fusions, MultiplyGroupConvolutionFusion)
    ADD_MATCHER(multiply_fusions, MultiplyConvolutionBackpropDataFusion)
    ADD_MATCHER(multiply_fusions, MultiplyGroupConvolutionBackpropDataFusion)
    ADD_MATCHER(multiply_fusions, MatMulMultiplyFusion)
    multiply_fusions->set_name("ov::pass::MultiplyFusions");
    REGISTER_PASS(manager, ConstantFolding)

    auto fq_fusions = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(fq_fusions, FakeQuantizeMulFusion)
    ADD_MATCHER(fq_fusions, FakeQuantizeReshapeFusion)
    ADD_MATCHER(fq_fusions, PullTransposeThroughFQUp)
    ADD_MATCHER(fq_fusions, ReluFakeQuantizeFusion)
    ADD_MATCHER(fq_fusions, AddFakeQuantizeFusion)
    ADD_MATCHER(fq_fusions, MulFakeQuantizeFusion)
    fq_fusions->set_name("ov::pass::FakeQuantizeFusions");
    REGISTER_PASS(manager, ReverseInputChannelsFusion)
    REGISTER_PASS(manager, AlignEltwiseInputRanks)
    REGISTER_PASS(manager, SharedOpOptimization)
    REGISTER_PASS(manager, ConstantFolding)
    REGISTER_PASS(manager, SymbolicOptimizations)
    REGISTER_PASS(manager, ResolveNameCollisions, true);
    manager.run_passes(f);

    if (!m_use_shapes) {
        // Restore original shapes to the ov::Model
        for (auto&& param : f->get_parameters()) {
            param->set_partial_shape(input_shapes.at(param.get()));
        }

        for (const auto& variable : f->get_variables()) {
            variable->update_data_shape(variable_shapes.at(variable.get()));
        }
    }
    f->validate_nodes_and_infer_types();

    return false;
}
