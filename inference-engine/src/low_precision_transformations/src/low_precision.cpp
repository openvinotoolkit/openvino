// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"

#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <transformations/utils/utils.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>
#include <low_precision/lpt_itt.hpp>

#include "low_precision/align_quantization_intervals.hpp"
#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_can_be_quantized.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/align_quantization_parameters.hpp"

#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"

// branch specific transformations
#include "low_precision/concat.hpp"

#include "low_precision/fake_quantize_decomposition.hpp"

// general transformations
#include "low_precision/add.hpp"
#include "low_precision/avg_pool.hpp"
#include "low_precision/clamp.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/pad.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reduce_max.hpp"
#include "low_precision/reduce_mean.hpp"
#include "low_precision/reduce_min.hpp"
#include "low_precision/reduce_sum.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/split.hpp"
#include "low_precision/shuffle_channels.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"

using namespace ngraph::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::TypeRelaxedReplacer, "TypeRelaxedReplacer", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupOptimizations, "MarkupOptimizations", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::LowPrecision, "LowPrecision", 0);

template <typename NodeType>
class RelaxNode : public ngraph::pass::MatcherPass {
public:
    RelaxNode() {
        ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
            auto l_node = std::dynamic_pointer_cast<NodeType>(m.get_match_root());
            if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(l_node)) {
                return false;
            }
            if (!l_node) {
                THROW_IE_LPT_EXCEPTION(*l_node) << "unexpected operation type";
            }

            OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecisionTypeRelaxedMatcher");

            ngraph::element::TypeVector inputPrecisions;
            for (const auto& inputs : l_node->inputs()) {
                inputPrecisions.push_back(inputs.get_element_type());
            }

            ngraph::element::TypeVector outputPrecisions;
            for (const auto& output : l_node->outputs()) {
                outputPrecisions.push_back(output.get_element_type());
            }

            auto replacement = std::make_shared<ngraph::op::TypeRelaxed<NodeType>>(*l_node, inputPrecisions, outputPrecisions);

            copy_runtime_info(l_node, replacement);
            replace_node(l_node, replacement);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<NodeType>(), "TypeRelaxedReplacer");
    }
};

ngraph::pass::low_precision::TypeRelaxedReplacer::TypeRelaxedReplacer() {
    add_matcher<RelaxNode<opset1::Add>>();
    add_matcher<RelaxNode<opset1::AvgPool>>();
    add_matcher<RelaxNode<opset1::Clamp>>();
    add_matcher<RelaxNode<opset1::Convolution>>();
    add_matcher<RelaxNode<opset1::ConvolutionBackpropData>>();
    add_matcher<RelaxNode<opset1::DepthToSpace>>();
    add_matcher<RelaxNode<opset1::FakeQuantize>>();
    add_matcher<RelaxNode<opset1::GroupConvolution>>();
    add_matcher<RelaxNode<opset1::PRelu>>();
    add_matcher<RelaxNode<opset1::ReduceMean>>();
    add_matcher<RelaxNode<opset1::ReduceSum>>();
    add_matcher<RelaxNode<opset1::Subtract>>();
    add_matcher<RelaxNode<opset1::Interpolate>>();
    add_matcher<RelaxNode<opset1::Multiply>>();
    add_matcher<RelaxNode<opset3::MVN>>();
    add_matcher<RelaxNode<opset6::MVN>>();
    add_matcher<RelaxNode<opset1::NormalizeL2>>();
    add_matcher<RelaxNode<opset4::Interpolate>>();
}

MarkupOptimizations::MarkupOptimizations(
    const std::vector<OperationPrecisionRestriction>& precisionRestrictions,
    const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions) :
    precisionRestrictions(precisionRestrictions),
    quantizationRestrictions(quantizationRestrictions) {}

bool ngraph::pass::low_precision::MarkupOptimizations::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager markup(get_pass_config());
    markup.set_per_pass_validation(false);
    markup.register_pass<low_precision::MarkupCanBeQuantized>();
    if (!precisionRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupPrecisions>(precisionRestrictions);
    }
    if (!quantizationRestrictions.empty()) {
        markup.register_pass<low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
    }
    if (ngraph::op::util::has_op_with_type<ngraph::opset1::AvgPool>(f)) {
        markup.register_pass<low_precision::MarkupAvgPoolPrecisionPreserved>();
    }
    markup.register_pass<low_precision::PropagatePrecisions>();
    if (ngraph::op::util::has_op_with_type<ngraph::opset1::Concat>(f)) {
        markup.register_pass<low_precision::AlignQuantizationIntervals>();
        markup.register_pass<low_precision::AlignQuantizationParameters>();
    }
    markup.run_passes(f);
    return false;
}

ngraph::pass::low_precision::LowPrecision::LowPrecision(
        const std::vector<OperationPrecisionRestriction>& precisionRestrictions,
        const std::vector<OperationPerTensorQuantizationRestriction>& quantizationRestrictions,
        const LayerTransformation::Params params) :
        precisionRestrictions(precisionRestrictions),
        quantizationRestrictions(quantizationRestrictions),
        params(params) {}

bool ngraph::pass::low_precision::LowPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "LowPrecision");

    auto passConfig = get_pass_config();
    ngraph::pass::Manager manager(passConfig);

    auto prerequisites = manager.register_pass<ngraph::pass::GraphRewrite>();
    const element::TypeVector supportedTypes = {ngraph::element::i8, ngraph::element::u8};
    prerequisites->add_matcher<PullReshapeThroughDequantization>(supportedTypes);
    prerequisites->add_matcher<PullTransposeThroughDequantization>(supportedTypes);
    prerequisites->add_matcher<ngraph::pass::LinOpSequenceFusion>();

    manager.register_pass<TypeRelaxedReplacer>();

    manager.register_pass<ngraph::pass::low_precision::MarkupOptimizations>(precisionRestrictions, quantizationRestrictions);

    auto common = manager.register_pass<ngraph::pass::GraphRewrite>();
    common->add_matcher<ngraph::pass::low_precision::AddTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ClampTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ConvolutionBackpropDataTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::DepthToSpaceTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::InterpolateTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::GroupConvolutionTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::MatMulTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::MultiplyTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::MVNTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::NormalizeL2Transformation>(params);
    common->add_matcher<ngraph::pass::low_precision::PadTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::PReluTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReduceMaxTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReduceMeanTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReduceMinTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReduceSumTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReluTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ReshapeTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::SqueezeTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::ShuffleChannelsTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::SplitTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::StridedSliceTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::TransposeTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::UnsqueezeTransformation>(params);
    common->add_matcher<ngraph::pass::low_precision::VariadicSplitTransformation>(params);

    std::shared_ptr<ngraph::pass::GraphRewrite> cleanup = manager.register_pass<ngraph::pass::GraphRewrite>();
    cleanup->add_matcher<ngraph::pass::low_precision::FoldConvertTransformation>(params);
    cleanup->add_matcher<ngraph::pass::low_precision::FuseConvertTransformation>(params);
    cleanup->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>(params);
    cleanup->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>(params);
    // WA: precision restrictions for groupConv must be propagated to MultiplyToGroupConvolution transformation
    cleanup->add_matcher<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>(
        params,
        OperationPrecisionRestriction::getPrecisionsByOperationType<opset1::GroupConvolution>(precisionRestrictions));
    manager.register_pass<ngraph::pass::low_precision::FoldFakeQuantizeTransformation>(params);
    manager.register_pass<ngraph::pass::ConstantFolding>();

    manager.run_passes(f);
    return false;
}

bool ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(const std::shared_ptr<const ngraph::Function>& function) {
    std::set<std::shared_ptr<ngraph::Node>> handledNodes;
    std::deque<std::shared_ptr<ngraph::Node>> nodes;
    for (auto result : function->get_results()) {
        nodes.push_front(result);
    }

    while (!nodes.empty()) {
        auto node = nodes.front();
        nodes.pop_front();

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto parent = node->get_input_node_shared_ptr(i);
            if (handledNodes.find(parent) != handledNodes.end()) {
                continue;
            }

            const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = ov::as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
            if ((fakeQuantize != nullptr) &&
                QuantizationDetails::outputLayoutIsSupported(fakeQuantize) &&
                QuantizationDetails::isSupportedLevel(fakeQuantize->get_levels())) {
                return true;
            }

            nodes.push_front(parent);
            handledNodes.insert(parent);
        }
    }
    return false;
}
