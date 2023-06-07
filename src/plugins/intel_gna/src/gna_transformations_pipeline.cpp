// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_transformations_pipeline.hpp"

#include "gna_itt.hpp"
#include "legacy/net_pass.h"
#include "legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "optimizer/gna_pass_manager.hpp"
#include "transformations/broadcast_const.hpp"
#include "transformations/common_optimizations/add_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/common_optimizations/concat_reduce_fusion.hpp"
#include "transformations/common_optimizations/fq_mul_fusion.hpp"
#include "transformations/common_optimizations/fq_reshape_fusion.hpp"
#include "transformations/common_optimizations/pull_transpose_through_fq.hpp"
#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/common_optimizations/transpose_sinking.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/convert_dwsc_to_scaleshifts.hpp"
#include "transformations/convert_matmul_to_pointwise_convolution.hpp"
#include "transformations/convert_padded_to_valid_convolution.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/decompose_2d_convolution.hpp"
#include "transformations/decompose_mvn.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/handle_transposes_around_matmul.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/insert_copy_layer.hpp"
#include "transformations/insert_identity_layer.hpp"
#include "transformations/insert_reshape_around_matmul.hpp"
#include "transformations/insert_transpose_after_convolution_or_pooling.hpp"
#include "transformations/markup_fusable_transpose.hpp"
#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/op_conversions/gru_cell_decomposition.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "transformations/op_conversions/softsign_decomposition.hpp"
#include "transformations/opset_conversions/convert_opset2_to_opset1.hpp"
#include "transformations/opset_conversions/convert_opset3_to_opset2.hpp"
#include "transformations/pwl_approximation.hpp"
#include "transformations/remove_converts.hpp"
#include "transformations/remove_extra_reshapes.hpp"
#include "transformations/remove_in_out_processing.hpp"
#include "transformations/remove_single_input_concat.hpp"
#include "transformations/reorder_activation_and_pooling.hpp"
#include "transformations/split_convolution_with_large_buffer_size.hpp"
#include "transformations/split_eltwise.hpp"
#include "transformations/substitute_softsign.hpp"
#include "transformations/swap_input_matmul_gna.hpp"
#include "transformations/unfuse_reshape_and_transpose.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gna {

static size_t iterator = 0;

template <typename T, typename... Args>
static std::shared_ptr<T> create_with_unique_name(Args&&... args) {
    auto pass = std::make_shared<T>(std::forward<Args>(args)...);
    auto name = pass->get_name();
    name.append("_" + std::to_string(iterator++));

    if (iterator == std::numeric_limits<size_t>::max()) {
        iterator = 0;
    }
    pass->set_name(name);
    return pass;
}

void TransformationsPipeline::apply(const std::shared_ptr<ov::Model>& model,
                                    ov::intel_gna::PrePostProcessModels* input_output_subgraphs) {
    apply(model, {}, input_output_subgraphs);
}

void TransformationsPipeline::apply(const std::shared_ptr<ov::Model>& model,
                                    const std::vector<std::string>& transformations_list,
                                    ov::intel_gna::PrePostProcessModels* input_output_subgraphs) {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "TransformationsPipeline::apply");

    fake_quantized = ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(model);
    const bool has_convolution = ov::op::util::has_op_with_type<ngraph::opset7::Convolution>(model);
    const bool has_matmul = ov::op::util::has_op_with_type<ngraph::opset7::MatMul>(model);
    const bool has_mvn = ov::op::util::has_op_with_type<ngraph::opset7::MVN>(model) ||
                         ov::op::util::has_op_with_type<ov::op::v0::MVN>(model);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();

    // In OV API 2.0(IRv10) default convertion to fp32 (inputs, outputs and weights) is disabled
    // and we need to run the ConvertPrecision transformation to support old networks.
    std::vector<std::shared_ptr<ov::pass::PassBase>> transformations;
    transformations.push_back(create_with_unique_name<ov::pass::ConvertPrecision>(
        precisions_map{{ngraph::element::f16, ngraph::element::f32}}));
    transformations.push_back(create_with_unique_name<ov::pass::ConvertMVN1ToMVN6>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::DecomposeMVN>());
    transformations.push_back(create_with_unique_name<ov::pass::CommonOptimizations>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::RemoveInputConvert>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::RemoveOutputConvert>());
    transformations.push_back(create_with_unique_name<ov::pass::ConvertSequenceToTensorIterator>());
    transformations.push_back(create_with_unique_name<ov::pass::GRUCellDecomposition>());
    transformations.push_back(create_with_unique_name<ov::pass::LSTMCellDecomposition>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::ConvertDWSCToScaleShifts>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::ConvertPaddedToValidConv>());

    transformations.push_back(
        create_with_unique_name<ov::intel_gna::pass::Decompose2DConvTransposedWithBiasAF>(config.gnaPrecision));
    transformations.push_back(
        create_with_unique_name<ov::intel_gna::pass::Decompose2DConvTransposedWithBias>(config.gnaPrecision));
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::Decompose2DConv>(config.gnaPrecision));
    if (!has_convolution) {
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>());
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>());
        transformations.push_back(create_with_unique_name<ov::intel_gna::pass::ConvertMatmulToPointWiseConvolution>());
    }

    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SplitConvolutionWithFq>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SplitConvolutionWithBias>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SplitConvolution>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertReshapeAroundMatmulWithTranspose>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertReshapeAroundMatmulWithFq>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertReshapeAroundMatmulWithAdd>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertReshapeAroundMatmul>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SwapInputMatMulWithTrailingTranspose>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SwapInputMatMulWithAct>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SwapInputMatMulWithFq>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SwapInputMatMulWithBias>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SwapInputMatMul>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::HandleTransposesAroundMatMul>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertTransposeAfterConvOrPool>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::Unfuse2dto4dReshapeAndTranspose>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::Unfuse4dto2dReshapeAndTranspose>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::RemoveExtraReshapes>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::ReorderActivationAndPooling>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::RemoveSingleInputConcat>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SubstituteSoftsign>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertCopyBeforeLayerToBeEliminated>());
    if (!has_convolution && !has_matmul && !has_mvn) {
        // TODO: Remove this condition when the legacy layout transformation (NCHW->NHWC) is disabled
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::RemoveInputsProcessing>(input_output_subgraphs));
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::RemoveOutputsProcessing>(input_output_subgraphs));
    }
    transformations.push_back(create_with_unique_name<ov::pass::ConvertOpSet3ToOpSet2>());
    transformations.push_back(create_with_unique_name<ov::pass::ConvertOpSet2ToOpSet1>());
    transformations.push_back(create_with_unique_name<ngraph::pass::ConvertOpSet1ToLegacy>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::MarkupFusableTranspose>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::RemoveExtraReshapes>());
    /*
        Put BroadcastAddMultiplyConst here after ConvertOpSet..() transformations since there are conficts with them.
        ngraph::pass::ConvertOpSet1ToLegacy -> ngraph::pass::BiasFusions ->
                                                ngraph::pass::ConvAddFusion, ngraph::pass::ConvMultiplyFusion
        That transormations fuse bias into convolution and recognizes const node as [1, C, 1, 1].
        TODO: move that transformation just beyond RemoveSingleInputConcat pass after removing ConvertOpSet1ToLegacy
            transormations
    */
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::BroadcastAddMultiplyConst>());

    /*
        SplitEltwise has dependency on BroadcastAddMultiplyConst for case when spliting of Constant
        input is doing
    */
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::SplitEltwise>());
    /* The following transformations perform insertion of Identity layer in 3 steps:
        1. Mark inputs with rt_info attribute where precision change from i32 to i16/i8 is happened
        2. Insert Identity after operation which have consumers marked with precision change
        3. Cleanup appropriate attribute from rt_info
    */
    transformations.push_back(
        create_with_unique_name<ov::intel_gna::pass::MarkIdentityCandidates>(config.gnaFlags.input_low_precision));
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertIdentity>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::IdentityCandidatesCleanup>());
    // Breaks fusing of layers before result
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::BreakFusingOfOutputLayers>());
    if (!config.gnaFlags.sw_fp32 && !config.gnaFlags.uniformPwlDesign) {
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::PWLApproximationWithFq>(config.gnaFlags.pwlMaxErrorPercent));
        transformations.push_back(
            create_with_unique_name<ov::intel_gna::pass::PWLApproximation>(config.gnaFlags.pwlMaxErrorPercent));
    }
    transformations.push_back(create_with_unique_name<ov::pass::UnrollTensorIterator>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertCopyBeforeAssignLayer>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::InsertCopyBeforeConcatLayer>());
    transformations.push_back(
        create_with_unique_name<ov::intel_gna::pass::HandleMultiConnectedLayerToConcatAndMemory>());
    transformations.push_back(create_with_unique_name<ov::intel_gna::pass::HandleNonFunctionalSubgraphs>());
    transformations.push_back(
        create_with_unique_name<ov::pass::ConvertPrecision>(precisions_map{{ov::element::i64, ov::element::i32},
                                                                           {ov::element::u64, ov::element::i32},
                                                                           {ov::element::u32, ov::element::i32}}));

    if (transformations_list.empty()) {
        register_all_passes(transformations, manager);
    } else {
        register_passes_with_given_order(transformations_list, transformations, manager);
    }

    const auto& pass_config = manager.get_pass_config();

    // Allowing FP16 Converts to be folded and FP16 constants to upgrade to FP32 data type
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();

    pass_config->disable<ov::pass::FakeQuantizeMulFusion>();
    pass_config->disable<ov::pass::FakeQuantizeReshapeFusion>();
    pass_config->disable<ov::pass::PullTransposeThroughFQUp>();
    pass_config->disable<ov::pass::ReluFakeQuantizeFusion>();
    // Consider to enable after per-channel quantization on FakeQuantize layer is supported in GNAPlugin, see issue
    // 52034
    pass_config->disable<ov::pass::AddFakeQuantizeFusion>();
    // TransposeReduction can be enabled when Transpose-Conv-Transpose patterns will be handled in ngraph
    // transformations
    pass_config->disable<ov::pass::TransposeReduction>();
    // Operations Max and Min aren't supported
    pass_config->disable<ov::pass::ConcatReduceFusion>();

    manager.run_passes(model);

    is_ngraph_passes_used = true;
}

void TransformationsPipeline::register_all_passes(const PassesInstances& passes, ov::pass::Manager& manager) {
    for (auto& pass : passes) {
        manager.register_pass_instance(pass);
    }
}

void TransformationsPipeline::register_passes_with_given_order(const std::vector<std::string>& transformations_list,
                                                               const PassesInstances& passes,
                                                               ov::pass::Manager& manager) {
#ifdef DUMP_TRSANSFORMATION_LIST
    // configuration.json has to be updated each time transformation list was modified.
    for (auto& pass : passes) {
        std::cout << "    { \"name\": \"" << pass->get_name() << "\" }," << std::endl;
    }
#endif

    std::string transformation_name;
    auto find_pass = [&transformation_name](std::shared_ptr<ov::pass::PassBase> pass) {
        return transformation_name.compare(pass->get_name()) == 0;
    };

    PassesInstances passes_copy = passes;

    for (const auto& transformation : transformations_list) {
        transformation_name = transformation;
        auto pass_iterator = std::find_if(passes_copy.begin(), passes_copy.end(), find_pass);
        if (pass_iterator != passes_copy.end()) {
            manager.register_pass_instance(*pass_iterator);
            passes_copy.erase(pass_iterator);
        }
    }
}

IE_SUPPRESS_DEPRECATED_START
void TransformationsPipeline::apply_legacy(const InferenceEngine::CNNNetwork& network, bool runBeforeCopy) {
    OV_ITT_SCOPED_TASK(itt::domains::GNAPlugin, "TransformationsPipeline::apply_legacy");
    auto passes =
        std::make_shared<PassManager>(PassManagerSettings{runBeforeCopy, config.gnaFlags.input_low_precision}, network);
    passes->registerPass<RemoveConstPass>();
    if (!is_ngraph_passes_used) {
        passes->registerPass<UnrollTIPass>();
        passes->registerPass<RemoveConstPass>();
        passes->registerPass<UnrollLSTMCellPass>();
        passes->registerPass<RemoveSingleInputConcatPass>();
        passes->registerPass<BroadcastConstPass>();
        passes->registerPass<SubstituteScaleShiftBroadCastPass>();
    }

    if (fake_quantized)
        passes->registerPass<SubstituteSoftSignPass>();

    // fake quantisation aware passes
    passes->registerPass<FuseFQIntoWeightsPass>();
    passes->registerPass<MoveFakeQuantizeLayerIntoQuantParamsPass>();

    passes->registerPass<TransposeWeightsFromNCHWToNHWCPass>();

    passes->registerPass<SubstitutePReluPass>();

    if (!is_ngraph_passes_used) {
        passes->registerPass<ReorderMaxPoolPass>();
        passes->registerPass<EltwiseSplitOverChannelsPass>();
    }

    passes->registerPass<InsertSplitAligningFilterPass>();

    if (!is_ngraph_passes_used) {
        passes->registerPass<InsertCopyLayerPass>();
    }
    passes->registerPass<FlattenTrivialConcatPass>();
    passes->registerPass<InsertConcatAligningFilterPass>();
    passes->registerPass<ReorderConcatInputsPass>();
    passes->registerPass<RemovePermutationsNHWCToNCHWPass>();
    // Keep legacy inserting of Identity layer here
    // because concat and split aliging passes are not moved to ngraph yet
    passes->registerPass<InsertIdentityLayerPass>();
    passes->registerPass<BreakFusingOfOutputLayersPass>();
    passes->registerPass<InsertDiagonalLayerPass>();
    passes->registerPass<HandleMultipleActivationsForTheLayerPass>();
    passes->registerPass<ForbidActivationFusingPass>();
    passes->registerPass<FuseMultipleIdentitiesPass>();
    passes->registerPass<FuseFullyConnectedWithEltwisePass>();
    legacy_pass_index = passes->run(legacy_pass_index);
}

void TransformationsPipeline::convert_precision_legacy(InferenceEngine::CNNNetwork& network) {
    if (!is_ngraph_passes_used) {
        InferenceEngine::NetPass::ConvertPrecision(network,
                                                   InferenceEngine::Precision::I64,
                                                   InferenceEngine::Precision::I32);
        InferenceEngine::NetPass::ConvertPrecision(network,
                                                   InferenceEngine::Precision::U64,
                                                   InferenceEngine::Precision::I32);
        InferenceEngine::NetPass::ConvertPrecision(network,
                                                   InferenceEngine::Precision::U32,
                                                   InferenceEngine::Precision::I32);
    }
}
IE_SUPPRESS_DEPRECATED_END
}  // namespace intel_gna
}  // namespace ov
