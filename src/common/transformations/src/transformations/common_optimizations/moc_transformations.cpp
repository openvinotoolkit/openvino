// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/add_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/align_eltwise_input_ranks.hpp>
#include <transformations/common_optimizations/batch_to_space_fusion.hpp>
#include <transformations/common_optimizations/binarize_weights.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/common_optimizations/conv_mul_fusion.hpp>
#include <transformations/common_optimizations/conv_to_binary_conv.hpp>
#include <transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp>
#include <transformations/common_optimizations/convert_quantize_dequantize.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <transformations/common_optimizations/dilated_convolution_converter.hpp>
#include <transformations/common_optimizations/disable_random_uniform_constant_folding.hpp>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>
#include <transformations/common_optimizations/divide_fusion.hpp>
#include <transformations/common_optimizations/eliminate_unsqueeze_gather.hpp>
#include <transformations/common_optimizations/fold_subgraph_empty_inputs.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>
#include <transformations/common_optimizations/fq_reshape_fusion.hpp>
#include <transformations/common_optimizations/gelu_fusion.hpp>
#include <transformations/common_optimizations/hsigmoid_fusion.hpp>
#include <transformations/common_optimizations/hswish_fusion.hpp>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/common_optimizations/matmul_const_transposes_extraction.hpp>
#include <transformations/common_optimizations/matmul_multiply_fusion.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/mul_conv_fusion.hpp>
#include <transformations/common_optimizations/mul_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/common_optimizations/nearest_neighbor_upsampling_fusion.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/common_optimizations/optimize_strided_slice.hpp>
#include <transformations/common_optimizations/pad_fusion.hpp>
#include <transformations/common_optimizations/prelu_fusion.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <transformations/common_optimizations/random_uniform_fusion.hpp>
#include <transformations/common_optimizations/relu_fake_quantize_fusion.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/common_optimizations/remove_filtering_boxes_by_size.hpp>
#include <transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp>
#include <transformations/common_optimizations/reshape_sequence_fusion.hpp>
#include <transformations/common_optimizations/ric_fusion.hpp>
#include <transformations/common_optimizations/shuffle_channels_fusion.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/common_optimizations/softmax_fusion.hpp>
#include <transformations/common_optimizations/softplus_fusion.hpp>
#include <transformations/common_optimizations/softplus_to_mish_fusion.hpp>
#include <transformations/common_optimizations/space_to_batch_fusion.hpp>
#include <transformations/common_optimizations/split_concat_pair_to_interpolate_fusion.hpp>
#include <transformations/common_optimizations/split_squeeze_concat_fusion.hpp>
#include <transformations/common_optimizations/subtract_fusion.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/common_optimizations/transpose_to_reshape.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/disable_convert_constant_folding_on_const_path.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/op_conversions/convert_negative.hpp>
#include <transformations/op_conversions/convert_scatter_elements_to_scatter.hpp>
#include <transformations/smart_reshape/lstm_states_broadcast.hpp>
#include <transformations/smart_reshape/reshape_sinking.hpp>

#include "itt.hpp"

bool ngraph::pass::MOCTransformations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MOCTransformations);
    // To avoid issues with dynamism we make nGraph Function dynamic and after we apply all
    // transformations we restore original shapes to the nGraph Function back
    std::unordered_map<ngraph::op::Parameter*, PartialShape> input_shapes;
    if (!m_use_shapes) {
        for (auto&& param : f->get_parameters()) {
            input_shapes[param.get()] = param->get_partial_shape();
            param->set_partial_shape(PartialShape::dynamic(param->get_partial_shape().rank()));
        }
        f->validate_nodes_and_infer_types();
    }

    ngraph::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);
    CC_TRANSFORMATIONS_MODEL_SCOPE(InitNodeInfo)
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    if (m_low_precision_enabled) {
        manager.register_pass<ngraph::pass::DisableConvertConstantFoldingOnConstPath>(
            element::TypeVector{ngraph::element::i8, ngraph::element::u8, ngraph::element::i4, ngraph::element::u4});
    }
    if (!m_use_shapes) {
        manager.register_pass<ngraph::pass::DisableShapeOfConstantFolding>();
    }
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams
    // should be performed before first ConstantFolding call.
    // The passes can deteach graph branches where zero dimesion is calculated.
    // Zero dimensions in shape causes creation empty tensors, which are incorrect during CF.
    // In particular, if zero dim tensor is consumed in body of MultiSubGraphOp
    // RemoveConcatZeroDimInput and RemoveMultiSubGraphOpDanglingParams should be called together.
    CC_TRANSFORMATIONS_MATCH_SCOPE(RemoveConcatZeroDimInput)
    manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(Validate)
    manager.register_pass<ov::pass::Validate>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(RemoveMultiSubGraphOpDanglingParams)
    manager.register_pass<ov::pass::RemoveMultiSubGraphOpDanglingParams>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(FoldSubgraphEmptyInputs)
    manager.register_pass<ov::pass::FoldSubgraphEmptyInputs>();
    manager.register_pass<ngraph::pass::DisableRandomUniformConstantFolding>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(ConstantFolding)
    manager.register_pass<ngraph::pass::ConstantFolding>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(Validate)
    manager.register_pass<ngraph::pass::Validate>();

    // FusedFilteringBoxesBySize transformation has the complex pattern
    // which can be affected by further transformations. So we have to
    // execute it at the beginning of the pipeline. Also, this pass resolves
    // dynamism, so we have to execute type/shape propagation after.
    manager.register_pass<ngraph::pass::FuseFilteringBoxesBySize>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(Validate)
    manager.register_pass<ngraph::pass::Validate>();

    if (!m_use_shapes) {  // Approved Smart Reshape
        CC_TRANSFORMATIONS_FUNCTION_SCOPE(LSTMStatesBroadcast)
        manager.register_pass<ov::pass::LSTMStatesBroadcast>();
        CC_TRANSFORMATIONS_MODEL_SCOPE(Validate)
        manager.register_pass<ov::pass::Validate>();
        CC_TRANSFORMATIONS_MATCH_SCOPE(ReshapeSinkingMatMul)
        manager.register_pass<ov::pass::ReshapeSinkingMatMul>();
        CC_TRANSFORMATIONS_MODEL_SCOPE(Validate)
        manager.register_pass<ov::pass::Validate>();
    }
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvertQuantizeDequantize)
    manager.register_pass<ngraph::pass::ConvertQuantizeDequantize>();
    CC_TRANSFORMATIONS_FUNCTION_SCOPE(SimplifyShapeOfSubGraph)
    manager.register_pass<ngraph::pass::SimplifyShapeOfSubGraph>();
    if (!m_use_shapes) {
        manager.register_pass<ngraph::pass::DisableShapeOfConstantFolding>();
    }
    // workaround until dynamism in NMS is not supported
    manager.register_pass<ngraph::pass::ConvertNmsGatherPathToUnsigned>();
    CC_TRANSFORMATIONS_FUNCTION_SCOPE(StridedSliceOptimization)
    manager.register_pass<ngraph::pass::StridedSliceOptimization>(m_use_shapes);
    CC_TRANSFORMATIONS_MATCH_SCOPE(BroadcastElementwiseFusion)
    manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto transpose_sinking = manager.register_pass<ngraph::pass::GraphRewrite>();
    transpose_sinking->add_matcher<ngraph::pass::TransposeSinking>();
    // SplitSqueezeConcatFusion should work in same GraphRewrite as TransposesSinking,
    // because it replaces pattern that may contain Transposes which must be optimized before
    // the transformation and it also inserts Transpose that can be optimized by TransposeSinking
    CC_TRANSFORMATIONS_MATCH_SCOPE(SplitSqueezeConcatFusion)
    transpose_sinking->add_matcher<ngraph::pass::SplitSqueezeConcatFusion>();
    }
    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto eliminations = manager.register_pass<ngraph::pass::GraphRewrite>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(EliminateUnsqueezeGather)
    eliminations->add_matcher<ngraph::pass::EliminateUnsqueezeGather>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(NopElimination)
    eliminations->add_matcher<ngraph::pass::NopElimination>(m_use_shapes /* do not use shape for elimination */);
    eliminations->set_name("ngraph::pass::CommonEliminations");
    }
    manager.register_pass<ngraph::pass::ConstantFolding>();

    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvertScatterElementsToScatter)
    common_fusions->add_matcher<ngraph::pass::ConvertScatterElementsToScatter>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SoftPlusFusion)
    common_fusions->add_matcher<ngraph::pass::SoftPlusFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SoftPlusToMishFusion)
    common_fusions->add_matcher<ngraph::pass::SoftPlusToMishFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SwishFusion)
    common_fusions->add_matcher<ngraph::pass::SwishFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(HSwishFusion)
    common_fusions->add_matcher<ngraph::pass::HSwishFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(HSigmoidFusion)
    common_fusions->add_matcher<ngraph::pass::HSigmoidFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(NormalizeL2Fusion)
    common_fusions->add_matcher<ngraph::pass::NormalizeL2Fusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ClampFusion)
    common_fusions->add_matcher<ngraph::pass::ClampFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(PadFusion)
    common_fusions->add_matcher<ngraph::pass::PadFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SoftmaxFusion)
    common_fusions->add_matcher<ngraph::pass::SoftmaxFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MVNFusion)
    common_fusions->add_matcher<ngraph::pass::MVNFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(DilatedConvolutionConverter)
    common_fusions->add_matcher<ngraph::pass::DilatedConvolutionConverter>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(GeluFusion)
    common_fusions->add_matcher<ngraph::pass::GeluFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(LeakyReluFusion)
    common_fusions->add_matcher<ngraph::pass::LeakyReluFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(RandomUniformFusion)
    common_fusions->add_matcher<ngraph::pass::RandomUniformFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SplitConcatPairToInterpolateFusion)
    common_fusions->add_matcher<ngraph::pass::SplitConcatPairToInterpolateFusion>(m_use_shapes);
    if (m_use_shapes) {
        CC_TRANSFORMATIONS_MATCH_SCOPE(NearestNeighborUpsamplingFusion)
        common_fusions->add_matcher<ngraph::pass::NearestNeighborUpsamplingFusion>();
    }
    CC_TRANSFORMATIONS_MATCH_SCOPE(DivideFusion)
    common_fusions->add_matcher<ngraph::pass::DivideFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(SubtractFusion)
    common_fusions->add_matcher<ngraph::pass::SubtractFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(TransposeToReshape)
    common_fusions->add_matcher<ngraph::pass::TransposeToReshape>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ReshapeSequenceFusion)
    common_fusions->add_matcher<ngraph::pass::ReshapeSequenceFusion>(m_use_shapes);
    CC_TRANSFORMATIONS_MATCH_SCOPE(MatMulConstTransposesExtraction)
    common_fusions->add_matcher<ngraph::pass::MatMulConstTransposesExtraction>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(PReluFusion)
    common_fusions->add_matcher<ngraph::pass::PReluFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(DepthToSpaceFusion)
    common_fusions->add_matcher<ngraph::pass::DepthToSpaceFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ShuffleChannelsFusion)
    common_fusions->add_matcher<ngraph::pass::ShuffleChannelsFusion>(!m_use_shapes);
    common_fusions->set_name("ngraph::pass::CommonFusions");
    }
    CC_TRANSFORMATIONS_MATCH_SCOPE(BinarizeWeights)
    manager.register_pass<ngraph::pass::BinarizeWeights>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvToBinaryConv)
    manager.register_pass<ngraph::pass::ConvToBinaryConv>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(BatchNormDecomposition)
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvertDivideWithConstant)
    decomp->add_matcher<ngraph::pass::ConvertDivideWithConstant>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvertNegative)
    decomp->add_matcher<ngraph::pass::ConvertNegative>();
    }
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto multiply_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvolutionMultiplyFusion)
    multiply_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(GroupConvolutionMultiplyFusion)
    multiply_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ConvolutionBackpropDataMultiplyFusion)
    multiply_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(GroupConvolutionBackpropDataMultiplyFusion)
    multiply_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MultiplyConvolutionFusion)
    multiply_fusions->add_matcher<ngraph::pass::MultiplyConvolutionFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MultiplyGroupConvolutionFusion)
    multiply_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MultiplyConvolutionBackpropDataFusion)
    multiply_fusions->add_matcher<ngraph::pass::MultiplyConvolutionBackpropDataFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MultiplyGroupConvolutionBackpropDataFusion)
    multiply_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MatMulMultiplyFusion)
    multiply_fusions->add_matcher<ngraph::pass::MatMulMultiplyFusion>();
    multiply_fusions->set_name("ngraph::pass::MultiplyFusions");
    CC_TRANSFORMATIONS_MODEL_SCOPE(ConstantFolding)
    manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    CC_TRANSFORMATIONS_MODEL_SCOPE(GraphRewrite)
    {
    auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(FakeQuantizeMulFusion)
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(FakeQuantizeReshapeFusion)
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(PullTransposeThroughFQUp)
    fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(ReluFakeQuantizeFusion)
    fq_fusions->add_matcher<ngraph::pass::ReluFakeQuantizeFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(AddFakeQuantizeFusion)
    fq_fusions->add_matcher<ngraph::pass::AddFakeQuantizeFusion>();
    CC_TRANSFORMATIONS_MATCH_SCOPE(MulFakeQuantizeFusion)
    fq_fusions->add_matcher<ngraph::pass::MulFakeQuantizeFusion>();
    fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");
    }
    manager.register_pass<ngraph::pass::ReverseInputChannelsFusion>();

    manager.register_pass<ngraph::pass::AlignEltwiseInputRanks>();
    CC_TRANSFORMATIONS_MODEL_SCOPE(ConstantFolding)
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.run_passes(f);

    if (!m_use_shapes) {
        // Restore original shapes to the nGraph Function
        for (auto&& param : f->get_parameters()) {
            param->set_partial_shape(input_shapes.at(param.get()));
        }
        f->validate_nodes_and_infer_types();
    }

    return false;
}
