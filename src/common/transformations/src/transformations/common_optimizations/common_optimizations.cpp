// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/common_optimizations.hpp"

#include <memory>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/moc_transformations.hpp>
#include <transformations/common_optimizations/simplify_shape_of_sub_graph.hpp>
#include <transformations/common_optimizations/transpose_reshape_elimination_for_matmul.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/op_conversions/softmax_decomposition.hpp>
#include <transformations/op_conversions/softsign_decomposition.hpp>

#include "itt.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/batch_to_space_fusion.hpp"
#include "transformations/common_optimizations/binarize_weights.hpp"
#include "transformations/common_optimizations/broadcast_elementwise_fusion.hpp"
#include "transformations/common_optimizations/clamp_fusion.hpp"
#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/conv_to_binary_conv.hpp"
#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"
#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"
#include "transformations/common_optimizations/depth_to_space_fusion.hpp"
#include "transformations/common_optimizations/dilated_convolution_converter.hpp"
#include "transformations/common_optimizations/disable_random_uniform_constant_folding.hpp"
#include "transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp"
#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/gelu_fusion.hpp"
#include "transformations/common_optimizations/hsigmoid_fusion.hpp"
#include "transformations/common_optimizations/hswish_fusion.hpp"
#include "transformations/common_optimizations/interpolate_sequence_fusion.hpp"
#include "transformations/common_optimizations/leaky_relu_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_divides.hpp"
#include "transformations/common_optimizations/matmul_multiply_fusion.hpp"
#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/mvn_fusion.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "transformations/common_optimizations/normalize_l2_fusion.hpp"
#include "transformations/common_optimizations/optimize_strided_slice.hpp"
#include "transformations/common_optimizations/pad_fusion.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/common_optimizations/random_uniform_fusion.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/remove_filtering_boxes_by_size.hpp"
#include "transformations/common_optimizations/shuffle_channels_fusion.hpp"
#include "transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp"
#include "transformations/common_optimizations/softmax_fusion.hpp"
#include "transformations/common_optimizations/softplus_fusion.hpp"
#include "transformations/common_optimizations/softplus_to_mish_fusion.hpp"
#include "transformations/common_optimizations/space_to_batch_fusion.hpp"
#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"
#include "transformations/common_optimizations/strides_optimization.hpp"
#include "transformations/common_optimizations/swish_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/disable_decompression_convert_constant_folding.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"
#include "transformations/op_conversions/convert_deformable_conv_v8_to_v1.hpp"
#include "transformations/op_conversions/convert_depth_to_space.hpp"
#include "transformations/op_conversions/convert_divide.hpp"
#include "transformations/op_conversions/convert_gather_downgrade.hpp"
#include "transformations/op_conversions/convert_gather_upgrade.hpp"
#include "transformations/op_conversions/convert_gelu.hpp"
#include "transformations/op_conversions/convert_interpolate1_to_interpolate4.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_upgrade.hpp"
#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"
#include "transformations/op_conversions/convert_mod.hpp"
#include "transformations/op_conversions/convert_negative.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"
#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/convert_scatter_elements_to_scatter.hpp"
#include "transformations/op_conversions/convert_softmax_downgrade.hpp"
#include "transformations/op_conversions/convert_softmax_upgrade.hpp"
#include "transformations/op_conversions/convert_space_to_depth.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"
#include "transformations/op_conversions/detection_output_downgrade.hpp"
#include "transformations/op_conversions/detection_output_upgrade.hpp"
#include "transformations/op_conversions/einsum_decomposition.hpp"
#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"
#include "transformations/op_conversions/gelu7_downgrade.hpp"
#include "transformations/op_conversions/hsigmoid_decomposition.hpp"
#include "transformations/op_conversions/hswish_decomposition.hpp"
#include "transformations/op_conversions/log_softmax_decomposition.hpp"
#include "transformations/op_conversions/mvn6_decomposition.hpp"
#include "transformations/op_conversions/reduce_l1_decomposition.hpp"
#include "transformations/op_conversions/reduce_l2_decomposition.hpp"
#include "transformations/op_conversions/simplify_ctc_greedy_decoder_seq_len.hpp"

bool ngraph::pass::CommonOptimizations::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(CommonOptimizations);
    ngraph::pass::Manager manager(get_pass_config());
    manager.set_per_pass_validation(false);

    manager.register_pass<ov::pass::DisableDecompressionConvertConstantFolding>();

    // Disable low_precision_enabled as all plugins handle low-precision sub-graph manually
    // before CommonOptimization pipeline execution
    manager.register_pass<ngraph::pass::MOCTransformations>(true, false);

    // Enabling conversion of FP16 IR to legacy representation, each plugin have to disable it
    // after support for FP16 IR is implemented
    manager.register_pass<ov::pass::ConvertCompressedOnlyToLegacy>();

    manager.register_pass<ov::pass::MarkPrecisionSensitiveDivides>();

    // TODO: move to KMB
    manager.register_pass<ngraph::pass::WeightsDequantizeToFakeQuantize>();

    auto common_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    common_fusions->add_matcher<ngraph::pass::DepthToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::ShuffleChannelsFusion>(false);
    common_fusions->add_matcher<ngraph::pass::SpaceToBatchFusion>();
    common_fusions->add_matcher<ngraph::pass::BatchToSpaceFusion>();
    common_fusions->add_matcher<ngraph::pass::InterpolateSequenceFusion>();
    common_fusions->add_matcher<ngraph::pass::SkipGatherBeforeTransposeAndReshape>();
    common_fusions->set_name("ngraph::pass::CommonFusions");

    manager.register_pass<ngraph::pass::ConcatReduceFusion>();

    manager.register_pass<ngraph::pass::ConvertPadToGroupConvolution, false>();
    manager.register_pass<ngraph::pass::ConvertInterpolate1ToInterpolate4, false>();

    auto decomp = manager.register_pass<ngraph::pass::GraphRewrite>();
    decomp->add_matcher<ngraph::pass::Gelu7Downgrade>();
    decomp->add_matcher<ngraph::pass::BidirectionalSequenceDecomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL1Decomposition>();
    decomp->add_matcher<ngraph::pass::ReduceL2Decomposition>();
    decomp->add_matcher<ngraph::pass::HSwishDecomposition>();
    decomp->add_matcher<ngraph::pass::HSigmoidDecomposition>();
    decomp->add_matcher<ngraph::pass::LogSoftmaxDecomposition>();
    decomp->add_matcher<ngraph::pass::ConvertReduceToPooling>();
    decomp->add_matcher<ngraph::pass::ConvertBroadcastToTiles>();
    decomp->add_matcher<ngraph::pass::ConvertMod>();
    decomp->add_matcher<ngraph::pass::ConvertGELU>();
    decomp->add_matcher<ngraph::pass::ConvertMinimum>();
    decomp->add_matcher<ngraph::pass::ConvertSubtract>();
    decomp->add_matcher<ngraph::pass::ConvertDivide>();
    decomp->add_matcher<ngraph::pass::ConvertNegative>();
    decomp->add_matcher<ngraph::pass::ConvertDepthToSpace>();
    decomp->add_matcher<ngraph::pass::ConvertSpaceToDepth>();
    decomp->add_matcher<ngraph::pass::BatchNormDecomposition>();
    decomp->add_matcher<ngraph::pass::MVN6Decomposition>();
    decomp->add_matcher<ngraph::pass::NormalizeL2Decomposition, false>();
    decomp->add_matcher<ngraph::pass::SimplifyCTCGreedyDecoderSeqLen>();
    decomp->add_matcher<ngraph::pass::EinsumDecomposition>();
    decomp->add_matcher<ngraph::pass::SoftmaxDecomposition, false>();
    decomp->add_matcher<ngraph::pass::SoftSignDecomposition>();
    decomp->add_matcher<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    decomp->add_matcher<ngraph::pass::DropoutWithRandomUniformReplacer>();
    decomp->add_matcher<ngraph::pass::TransposeReshapeEliminationForMatmul>();
    decomp->set_name("ngraph::pass::CommonDecompositions");

    // CF is required after all decompositions
    manager.register_pass<ngraph::pass::ConstantFolding>();

    // LinOpSequenceFusion must be executed after all decompositions
    manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    manager.register_pass<ngraph::pass::UnrollIf>();

    auto multiply_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    multiply_fusions->add_matcher<ngraph::pass::ConvolutionMultiplyFusion>();
    multiply_fusions->add_matcher<ngraph::pass::GroupConvolutionMultiplyFusion>();
    multiply_fusions->add_matcher<ngraph::pass::ConvolutionBackpropDataMultiplyFusion>();
    multiply_fusions->add_matcher<ngraph::pass::GroupConvolutionBackpropDataMultiplyFusion>();
    multiply_fusions->add_matcher<ngraph::pass::MultiplyConvolutionFusion>();
    multiply_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionFusion>();
    multiply_fusions->add_matcher<ngraph::pass::MultiplyConvolutionBackpropDataFusion>();
    multiply_fusions->add_matcher<ngraph::pass::MultiplyGroupConvolutionBackpropDataFusion>();
    multiply_fusions->add_matcher<ngraph::pass::MatMulMultiplyFusion>();
    multiply_fusions->set_name("ngraph::pass::MultiplyFusions");

    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertGather8ToGather7>();  // not plugins implemented gather8
    manager.register_pass<ngraph::pass::ConvertGather7ToGather1>();  // not plugins implemented gather7
    manager.register_pass<ngraph::pass::ConvertGather1ToGather7, false>();
    manager.register_pass<ngraph::pass::ConvertGather7ToGather8, false>();
    manager.register_pass<ngraph::pass::ConvertDeformableConv8To1>();
    manager.register_pass<ngraph::pass::ConvertSoftMax8ToSoftMax1>();
    manager.register_pass<ngraph::pass::ConvertSoftMax1ToSoftMax8, false>();
    manager.register_pass<ngraph::pass::ConvertMaxPool8ToMaxPool1>();
    manager.register_pass<ngraph::pass::ConvertMaxPool1ToMaxPool8, false>();
    manager.register_pass<ngraph::pass::ConvertPriorBox8To0>();  // not plugins implemented priorbox8
    manager.register_pass<ngraph::pass::ConvertDetectionOutput1ToDetectionOutput8, false>();
    manager.register_pass<ngraph::pass::ConvertDetectionOutput8ToDetectionOutput1>();

    auto fq_fusions = manager.register_pass<ngraph::pass::GraphRewrite>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeMulFusion>();
    fq_fusions->add_matcher<ngraph::pass::FakeQuantizeReshapeFusion>();
    fq_fusions->add_matcher<ngraph::pass::PullTransposeThroughFQUp>();
    fq_fusions->add_matcher<ngraph::pass::ReluFakeQuantizeFusion>();
    fq_fusions->add_matcher<ngraph::pass::AddFakeQuantizeFusion>();
    fq_fusions->add_matcher<ngraph::pass::MulFakeQuantizeFusion>();
    fq_fusions->set_name("ngraph::pass::FakeQuantizeFusions");

    // StridesOptimization should be at the very end
    // because we cannot insert any MaxPools since they may prevent
    // other optimizations
    manager.register_pass<ngraph::pass::StridesOptimization>();
    manager.register_pass<ngraph::pass::Validate>();

    manager.run_passes(f);

    // Returning value is false because pass::Manager always apply Validation pass
    // if function was changed. This helps to avoid excess Validations after applying
    // this pass. In future when we will return more meaningful status code it will be
    // replaced with real status reported by manager.run_passes() method call.
    return false;
}
