// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset4_decl.hpp"
#include "openvino/opsets/opset6_decl.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/op_conversions/fake_convert_decomposition.hpp"
#include "transformations/utils/utils.hpp"

// prerequisite transformations
#include "low_precision/align_quantization_intervals.hpp"
#include "low_precision/align_quantization_parameters.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/markup_bias.hpp"
#include "low_precision/markup_can_be_quantized.hpp"
#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_quantization_granularity.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/assign_and_read_value.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/batch_to_space.hpp"
#include "low_precision/broadcast.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/gather.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/move_fake_quantize.hpp"
#include "low_precision/multiply_partial.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/pad.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/recurrent_cell.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/slice.hpp"
#include "low_precision/space_to_batch.hpp"
#include "low_precision/split.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/eliminate_fake_quantize.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"

using namespace ov::pass::low_precision;
using namespace ov::pass;

LowPrecision::LowPrecision(const std::vector<PrecisionsRestriction>& precisionRestrictions,
                           const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions,
                           const LayerTransformation::Params params)
    : precisionRestrictions(precisionRestrictions),
      quantizationRestrictions(quantizationRestrictions),
      params(params) {}

template <typename BaseOp>
void make_matcher_type_relaxed(GraphRewrite* transformation) {
    MATCHER_SCOPE(TypeRelaxedReplacer);
    using namespace ov;

    auto p_node = pattern::wrap_type<BaseOp>();

    graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto l_node = as_type_ptr<BaseOp>(m.get_match_root());
        if (!l_node) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected operation type for type relaxed conversion";
        }
        if (std::dynamic_pointer_cast<op::TypeRelaxedBase>(l_node)) {
            return false;
        }

        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionTypeRelaxedMatcher");

        std::vector<element::Type> inputPrecisions;
        for (auto& inputs : l_node->inputs()) {
            inputPrecisions.push_back(inputs.get_element_type());
        }

        std::vector<element::Type> outputPrecisions;
        for (auto& output : l_node->outputs()) {
            outputPrecisions.push_back(output.get_element_type());
        }

        auto replacement = std::make_shared<op::TypeRelaxed<BaseOp>>(*l_node, inputPrecisions, outputPrecisions);

        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(p_node, matcher_name);
    auto match_pass = std::make_shared<MatcherPass>(
        m->get_name(),
        m,
        [m, callback](const std::shared_ptr<Node>& node) -> bool {
            OPENVINO_DEBUG("Running matcher ", m->get_name(), " on ", node);
            if (std::dynamic_pointer_cast<pattern::Matcher>(m)->match(node->output(0))) {
                OPENVINO_DEBUG("Matcher ", m->get_name(), " matched ", node);
                OV_PASS_CALLBACK(m);
                bool status = callback(*m.get());
                // explicitly clear Matcher state because it holds pointers to matched nodes
                m->clear_state();
                return status;
            }
            m->clear_state();
            return false;
        },
        PassProperty::CHANGE_DYNAMIC_STATE);
    transformation->add_matcher(match_pass);
}

TypeRelaxedReplacer::TypeRelaxedReplacer() {
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Clamp>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::ConvolutionBackpropData>(this);
    make_matcher_type_relaxed<opset1::DepthToSpace>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::PRelu>(this);
    make_matcher_type_relaxed<opset1::ReduceMean>(this);
    make_matcher_type_relaxed<opset1::ReduceSum>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<opset1::Interpolate>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
    make_matcher_type_relaxed<op::v0::MVN>(this);
    make_matcher_type_relaxed<opset6::MVN>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
    make_matcher_type_relaxed<opset4::Interpolate>(this);
}

MarkupOptimizations::MarkupOptimizations(
    const std::vector<PrecisionsRestriction>& precisionRestrictions,
    const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions,
    const AttributeParameters& params)
    : precisionRestrictions(precisionRestrictions),
      quantizationRestrictions(quantizationRestrictions),
      params(params) {}

bool MarkupOptimizations::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(MarkupOptimizations);
    Manager markup(get_pass_config(), "LPT:MarkupOptimizations");
    markup.set_per_pass_validation(false);
    markup.register_pass<low_precision::MarkupCanBeQuantized>(params.defaultPrecisions);
    if (!precisionRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupPrecisions>(precisionRestrictions, params.defaultPrecisions);
    }
    if (!quantizationRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
    }
    if (ov::op::util::has_op_with_type<ov::opset1::AvgPool>(m)) {
        markup.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>(params.defaultPrecisions);
    }
    markup.register_pass<low_precision::PropagatePrecisions>(params);
    if (ov::op::util::has_op_with_type<ov::opset1::Concat>(m)) {
        markup.register_pass<low_precision::AlignQuantizationIntervals>(params.defaultPrecisions);
        markup.register_pass<low_precision::AlignQuantizationParameters>(params.defaultPrecisions);
    }
    markup.register_pass<low_precision::MarkupBias>();
    markup.run_passes(m);
    return false;
}

bool LowPrecision::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(LowPrecision);
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecision");

    Manager manager(get_pass_config(), "LowPrecision");
    const auto prerequisites = manager.register_pass<GraphRewrite>();
    const std::vector<ov::element::Type> supportedTypes = {ov::element::i8, ov::element::u8};
    ADD_MATCHER(prerequisites, PullReshapeThroughDequantization, supportedTypes)
    ADD_MATCHER(prerequisites, PullTransposeThroughDequantization, supportedTypes)
    ADD_MATCHER(prerequisites, LinOpSequenceFusion)
    ADD_MATCHER(prerequisites, MoveFakeQuantize)

    manager.register_pass<TypeRelaxedReplacer>();

    AttributeParameters attributeParams(params.deqPrecision, params.defaultPrecisions);
    manager.register_pass<low_precision::MarkupOptimizations>(precisionRestrictions,
                                                              quantizationRestrictions,
                                                              attributeParams);

    const auto common = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(common, AddTransformation, params)
    ADD_MATCHER(common, AssignAndReadValueTransformation, m, params)
    ADD_MATCHER(common, AvgPoolTransformation, params)
    ADD_MATCHER(common, BatchToSpaceTransformation, params)
    ADD_MATCHER(common, BroadcastTransformation, params)
    ADD_MATCHER(common, ClampTransformation, params)
    ADD_MATCHER(common, ConcatTransformation, params)
    ADD_MATCHER(common, ConvolutionTransformation, params)
    ADD_MATCHER(common, ConvolutionBackpropDataTransformation, params)
    ADD_MATCHER(common, DepthToSpaceTransformation, params)
    ADD_MATCHER(common, FakeQuantizeDecompositionTransformation, params)
    // In case of floating point low precision (e.g. fp8), FakeConvert is used for quantization
    if (std::any_of(params.defaultPrecisions.begin(),
                    params.defaultPrecisions.end(),
                    [](const ov::element::Type& type) {
                        return type.is_real();
                    })) {
        ADD_MATCHER(common, FakeConvertDecomposition);
    }
    ADD_MATCHER(common, FakeQuantizeTransformation, params)
    ADD_MATCHER(common, InterpolateTransformation, params)
    ADD_MATCHER(common, GroupConvolutionTransformation, params)
    ADD_MATCHER(common, MatMulTransformation, params)
    ADD_MATCHER(common, MaxPoolTransformation, params)
    ADD_MATCHER(common, MultiplyPartialTransformation, params)
    ADD_MATCHER(common, MVNTransformation, params)
    ADD_MATCHER(common, NormalizeL2Transformation, params)
    ADD_MATCHER(common, PadTransformation, params)
    ADD_MATCHER(common, PReluTransformation, params)
    ADD_MATCHER(common, RecurrentCellTransformation, params)
    ADD_MATCHER(common, ReduceMaxTransformation, params)
    ADD_MATCHER(common, ReduceMeanTransformation, params)
    ADD_MATCHER(common, ReduceMinTransformation, params)
    ADD_MATCHER(common, ReduceSumTransformation, params)
    ADD_MATCHER(common, ReluTransformation, params)
    ADD_MATCHER(common, ReshapeTransformation, params)
    ADD_MATCHER(common, SqueezeTransformation, params)
    ADD_MATCHER(common, ShuffleChannelsTransformation, params)
    ADD_MATCHER(common, SliceTransformation, params)
    ADD_MATCHER(common, SpaceToBatchTransformation, params)
    ADD_MATCHER(common, SplitTransformation, params)
    ADD_MATCHER(common, StridedSliceTransformation, params)
    ADD_MATCHER(common, TransposeTransformation, params)
    ADD_MATCHER(common, GatherTransformation, params)
    ADD_MATCHER(common, UnsqueezeTransformation, params)
    ADD_MATCHER(common, VariadicSplitTransformation, params)

    for (const auto& tr : additional_main_passes) {
        common->add_matcher(tr);
    }

    const auto cleanup = manager.register_pass<GraphRewrite>();
    ADD_MATCHER(cleanup, EliminateFakeQuantizeTransformation, params)
    ADD_MATCHER(cleanup, FoldConvertTransformation, params)
    ADD_MATCHER(cleanup, FuseConvertTransformation, params)
    ADD_MATCHER(cleanup, FuseSubtractToFakeQuantizeTransformation, params)
    ADD_MATCHER(cleanup, FuseMultiplyToFakeQuantizeTransformation, params)

    // WA: precision restrictions for groupConv must be propagated to MultiplyToGroupConvolution transformation
    ADD_MATCHER(cleanup,
                MultiplyToGroupConvolutionTransformation,
                params,
                PrecisionsRestriction::getPrecisionsByOperationType<opset1::GroupConvolution>(precisionRestrictions))

    REGISTER_PASS(manager, FoldFakeQuantizeTransformation, params)
    REGISTER_PASS(manager, ConstantFolding)

    manager.run_passes(m);
    return false;
}

bool LowPrecision::isFunctionQuantized(const std::shared_ptr<const ov::Model>& model,
                                       const std::set<levels>& supported_levels,
                                       bool check_fake_convert) {
    for (const auto& node : model->get_ordered_ops()) {
        if (check_fake_convert && ov::is_type<ov::op::v13::FakeConvert>(node)) {
            return true;
        } else if (const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(node)) {
            if (QuantizationDetails::outputLayoutIsSupported(fakeQuantize, true) &&
                QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels(), supported_levels)) {
                return true;
            }
        } else if (const auto multiSubGraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(node)) {
            // Look inside subraph operations, such as TensorIterator, Loop, If, etc
            for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++) {
                if (isFunctionQuantized(multiSubGraph->get_function(i))) {
                    return true;
                }
            }
        }
    }
    return false;
}
