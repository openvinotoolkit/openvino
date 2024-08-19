// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"

#include <memory>

#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

#include "transformations/utils/utils.hpp"
#include "low_precision/lpt_itt.hpp"

#include "low_precision/align_quantization_intervals.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/markup_bias.hpp"
#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_can_be_quantized.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/markup_quantization_granularity.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/align_quantization_parameters.hpp"

#include "openvino/util/log.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"

// branch specific transformations
#include "low_precision/concat.hpp"

#include "low_precision/fake_quantize_decomposition.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/assign_and_read_value.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/batch_to_space.hpp"
#include "low_precision/broadcast.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
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
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/slice.hpp"
#include "low_precision/space_to_batch.hpp"
#include "low_precision/split.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/gather.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"
#include "low_precision/move_fake_quantize.hpp"

// cleanup transformations
#include "itt.hpp"
#include "low_precision/convert.hpp"
#include "low_precision/eliminate_fake_quantize.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"

ov::pass::low_precision::LowPrecision::LowPrecision(
    const std::vector<PrecisionsRestriction>& precisionRestrictions,
    const std::vector<QuantizationGranularityRestriction>& quantizationRestrictions,
    const LayerTransformation::Params params) :
    precisionRestrictions(precisionRestrictions),
    quantizationRestrictions(quantizationRestrictions),
    params(params) {
}

using namespace ov::pass::low_precision;

template <typename BaseOp>
void make_matcher_type_relaxed(ov::pass::GraphRewrite* transformation) {
    MATCHER_SCOPE(TypeRelaxedReplacer);
    using namespace ov;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!ov::as_type_ptr<BaseOp>(n);
    };

    auto p_node = std::make_shared<pass::pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ov::graph_rewrite_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto l_node = std::dynamic_pointer_cast<BaseOp>(m.get_match_root());
        if (!l_node) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected operation type for type relaxed conversion";
        }
        if (std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(l_node)) {
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

        auto replacement = std::make_shared<ov::op::TypeRelaxed<BaseOp>>(*l_node, inputPrecisions, outputPrecisions);

        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(p_node, matcher_name);
    auto match_pass = std::make_shared<ov::pass::MatcherPass>(
            m->get_name(),
            m,
            [m, callback](const std::shared_ptr<Node>& node) -> bool {
                OPENVINO_DEBUG("Running matcher ", m->get_name(), " on ", node);
                if (std::dynamic_pointer_cast<ov::pass::pattern::Matcher>(m)->match(node->output(0))) {
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
            ov::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    transformation->add_matcher(match_pass);
}

ov::pass::low_precision::TypeRelaxedReplacer::TypeRelaxedReplacer() {
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
    const AttributeParameters& params) :
    precisionRestrictions(precisionRestrictions),
    quantizationRestrictions(quantizationRestrictions),
    params(params) {}

bool ov::pass::low_precision::MarkupOptimizations::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupOptimizations);
    ov::pass::Manager markup(get_pass_config(), "LPT:MarkupOptimizations");
    markup.set_per_pass_validation(false);
    markup.register_pass<low_precision::MarkupCanBeQuantized>(params.defaultPrecisions);
    if (!precisionRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupPrecisions>(precisionRestrictions, params.defaultPrecisions);
    }
    if (!quantizationRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
    }
    if (ov::op::util::has_op_with_type<ov::opset1::AvgPool>(f)) {
        markup.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>(params.defaultPrecisions);
    }
    markup.register_pass<low_precision::PropagatePrecisions>(params);
    if (ov::op::util::has_op_with_type<ov::opset1::Concat>(f)) {
        markup.register_pass<low_precision::AlignQuantizationIntervals>(params.defaultPrecisions);
        markup.register_pass<low_precision::AlignQuantizationParameters>(params.defaultPrecisions);
    }
    markup.register_pass<low_precision::MarkupBias>();
    markup.run_passes(f);
    return false;
}

bool ov::pass::low_precision::LowPrecision::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(LowPrecision);
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecision");

    auto passConfig = get_pass_config();
    ov::pass::Manager manager(passConfig, "LowPrecision");

    auto prerequisites = manager.register_pass<ov::pass::GraphRewrite>();
    const std::vector<ov::element::Type> supportedTypes = {ov::element::i8, ov::element::u8};
    ADD_MATCHER(prerequisites, PullReshapeThroughDequantization, supportedTypes)
    ADD_MATCHER(prerequisites, PullTransposeThroughDequantization, supportedTypes)
    using namespace ov::pass::low_precision;
    using namespace ov::pass;
    ADD_MATCHER(prerequisites, LinOpSequenceFusion)
    ADD_MATCHER(prerequisites, MoveFakeQuantize)

    manager.register_pass<TypeRelaxedReplacer>();

    AttributeParameters attributeParams(params.deqPrecision, params.defaultPrecisions);
    manager.register_pass<ov::pass::low_precision::MarkupOptimizations>(precisionRestrictions,
                                                                            quantizationRestrictions,
                                                                            attributeParams);

    std::shared_ptr<ov::pass::GraphRewrite> common = manager.register_pass<ov::pass::GraphRewrite>();

    ADD_MATCHER(common, AddTransformation, params)
    ADD_MATCHER(common, AssignAndReadValueTransformation, f, params)
    ADD_MATCHER(common, AvgPoolTransformation, params)
    ADD_MATCHER(common, BatchToSpaceTransformation, params)
    ADD_MATCHER(common, BroadcastTransformation, params)
    ADD_MATCHER(common, ClampTransformation, params)
    ADD_MATCHER(common, ConcatTransformation, params)
    ADD_MATCHER(common, ConvolutionTransformation, params)
    ADD_MATCHER(common, ConvolutionBackpropDataTransformation, params)
    ADD_MATCHER(common, DepthToSpaceTransformation, params)
    ADD_MATCHER(common, FakeQuantizeDecompositionTransformation, params)
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

    std::shared_ptr<ov::pass::GraphRewrite> cleanup = manager.register_pass<ov::pass::GraphRewrite>();
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

    manager.run_passes(f);
    return false;
}

bool ov::pass::low_precision::LowPrecision::isFunctionQuantized(
        const std::shared_ptr<const ov::Model>& model,
        const std::set<levels>& supported_levels) {
    std::set<std::shared_ptr<ov::Node>> handledNodes;
    std::deque<std::shared_ptr<ov::Node>> nodes;
    for (const auto& result : model->get_results()) {
        nodes.push_front(result);
    }

    while (!nodes.empty()) {
        const auto node = nodes.front();
        nodes.pop_front();

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            const auto parent = node->get_input_node_shared_ptr(i);
            if (handledNodes.find(parent) != handledNodes.end()) {
                continue;
            }

            if (const auto fakeQuantize = ov::as_type_ptr<ov::opset1::FakeQuantize>(parent)) {
                if (QuantizationDetails::outputLayoutIsSupported(fakeQuantize, true) &&
                    QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels(), supported_levels)) {
                    return true;
                }
            } else if (const auto multiSubGraph = ov::as_type_ptr<ov::op::util::MultiSubGraphOp>(parent)) {
                // Look inside subraph operations, such as TensorIterator, Loop, If, etc
                for (size_t i = 0; i < multiSubGraph->get_internal_subgraphs_size(); i++) {
                    if (isFunctionQuantized(multiSubGraph->get_function(i))) {
                        return true;
                    }
                }
            }

            nodes.push_front(parent);
            handledNodes.insert(parent);
        }
    }

    return false;
}

bool ov::pass::low_precision::LowPrecision::isFQLevelsPresent(
        const std::shared_ptr<const ov::Model>& model,
        const std::set<size_t>& levels) {
    std::vector<std::shared_ptr<ov::Node>> nodes = model->get_ops();
    for (auto& node : nodes) {
        const auto fakeQuantize = as_type_ptr<ov::opset1::FakeQuantize>(node);
        if (fakeQuantize != nullptr) {
            if (levels.count(fakeQuantize->get_levels()) == 1) {
                return true;
            }
        }
    }
    return false;
}
