// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"
#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset6.hpp"

#include "low_precision/rt_info/avg_pool_precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_avg_pool_precision_preserved.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/propagate_shared_value.hpp"
#include "low_precision/align_concat_quantization_parameters.hpp"

#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_attribute_to_precision_preserved.hpp"

//#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
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
#include "low_precision/depth_to_space.hpp"
#include "low_precision/fake_quantize.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/interpolate.hpp"
#include "low_precision/mat_mul.hpp"
#include "low_precision/max_pool.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/mvn.hpp"
#include "low_precision/normalize_l2.hpp"
#include "low_precision/prelu.hpp"
#include "low_precision/reshape.hpp"
#include "low_precision/relu.hpp"
#include "low_precision/squeeze.hpp"
#include "low_precision/subtract.hpp"
#include "low_precision/split.hpp"
#include "low_precision/strided_slice.hpp"
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "low_precision/subtract_multiply_to_multiply_add.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::LowPrecision, "LowPrecision", 0);

ngraph::pass::low_precision::LowPrecision::LowPrecision(
    const std::vector<OperationPrecisionRestriction>& restrictions,
    const LayerTransformation::Params params) : restrictions(restrictions), params(params) {
}

using namespace ngraph::pass::low_precision;

template <typename BaseOp>
void make_matcher_type_relaxed(ngraph::pass::GraphRewrite* transformation) {
    using namespace ngraph;

    auto is_op_type = [](std::shared_ptr<Node> n) {
        return !!as_type_ptr<BaseOp>(n);
    };

    auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, is_op_type);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher& m) {
        auto l_node = std::dynamic_pointer_cast<BaseOp>(m.get_match_root());
        if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(l_node)) {
            return false;
        }
        if (!l_node) {
            THROW_IE_LPT_EXCEPTION(*l_node) << "unexpected operation type";
        }

        std::vector<element::Type> inputPrecisions;
        for (auto& inputs : l_node->inputs()) {
            inputPrecisions.push_back(inputs.get_element_type());
        }

        std::vector<element::Type> outputPrecisions;
        for (auto& output : l_node->outputs()) {
            outputPrecisions.push_back(output.get_element_type());
        }

        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(*l_node, inputPrecisions, outputPrecisions);

        copy_runtime_info(l_node, replacement);
        replace_node(l_node, replacement);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "TypeRelaxedReplacer");
    NGRAPH_SUPPRESS_DEPRECATED_START
    transformation->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}

ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer::TypeRelaxedReplacer() {
    make_matcher_type_relaxed<opset1::Add>(this);
    make_matcher_type_relaxed<opset1::AvgPool>(this);
    make_matcher_type_relaxed<opset1::Clamp>(this);
    make_matcher_type_relaxed<opset1::Concat>(this);
    make_matcher_type_relaxed<opset1::Convolution>(this);
    make_matcher_type_relaxed<opset1::DepthToSpace>(this);
    make_matcher_type_relaxed<opset1::FakeQuantize>(this);
    make_matcher_type_relaxed<opset1::GroupConvolution>(this);
    make_matcher_type_relaxed<opset1::PRelu>(this);
    make_matcher_type_relaxed<opset1::Subtract>(this);
    make_matcher_type_relaxed<opset1::Interpolate>(this);
    make_matcher_type_relaxed<opset1::Multiply>(this);
    make_matcher_type_relaxed<op::MVN>(this);
    make_matcher_type_relaxed<opset6::MVN>(this);
    make_matcher_type_relaxed<opset1::NormalizeL2>(this);
    make_matcher_type_relaxed<opset4::Interpolate>(this);
}

bool ngraph::pass::low_precision::LowPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // TODO: to debug only
    // TransformationContext context(f);

    TypeRelaxedReplacer pass;
    pass.run_on_function(f);

    // pass config should be reused
    const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
    ngraph::pass::Manager manager;
    manager.register_pass<PullReshapeThroughDequantization>(supportedTypes);
    manager.register_pass<PullTransposeThroughDequantization>(supportedTypes);
    //manager.register_pass<ngraph::pass::LinOpSequenceFusion>();

    manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(restrictions); // <= MatcherPass in one GraphRewrite
    //manager.register_pass<ngraph::pass::low_precision::MarkupPrecisionPreserved>(); // <= MatcherPass in one GraphRewrite
    manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
    // think about decomposition later:
    // 1. the attribute creation
    // 2. propagation
    // 3. the attribute value set
    manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    manager.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();



    //manager.register_pass<ngraph::pass::low_precision::PropagateSharedValue<PrecisionsAttribute>>();


    // MatherPass set: return true to avoid next mather passes

    //std::shared_ptr<ngraph::pass::GraphRewrite> avgPoolPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    //avgPoolPropagation->add_matcher<CreateAttribute<AvgPoolPrecisionPreservedAttribute, opset1::AvgPool>>();
    //avgPoolPropagation->add_matcher<PropagateAttributeToPrecisionPreserved<AvgPoolPrecisionPreservedAttribute>>();
    //avgPoolPropagation->add_matcher<UpdateSharedValue<AvgPoolPrecisionPreservedAttribute, opset1::FakeQuantize>>();

    std::shared_ptr<ngraph::pass::GraphRewrite> precisionsPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    //precisionsPropagation->add_matcher<CreateAttribute<PrecisionsAttribute, opset1::FakeQuantize>>(AttributeSource::OutputPort); // outputPort
    //precisionsPropagation->add_matcher<PropagateAttributeToPrecisionPreserved<PrecisionsAttribute>>();

    //std::shared_ptr<ngraph::pass::GraphRewrite> intervalsAlignmentPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    //intervalsAlignmentPropagation->add_matcher<CreateAttribute<IntervalsAlignmentAttribute, opset1::FakeQuantize>>();
    //intervalsAlignmentPropagation->add_matcher<PropagateAttributeToPrecisionPreserved<IntervalsAlignmentAttribute>>();

    //std::shared_ptr<ngraph::pass::GraphRewrite> quantizationAlignmentPropagation = manager.register_pass<ngraph::pass::GraphRewrite>();
    //quantizationAlignmentPropagation->add_matcher<CreateAttributeForChild<QuantizationAlignment, opset1::FakeQuantize>>();
    //quantizationAlignmentPropagation->add_matcher<PropagateAttributeToPrecisionPreserved<QuantizationAlignment>>();

    //propagation->add_matcher<ngraph::pass::low_precision::QuantizationAlignmentCompleteOperation<PrecisionsAttribute>>();

    // template specialization by Node, Input, Output VS [Node]Propagation



    //{
    //    // TODO: just to DEBUG: use the same manager
    //    ngraph::pass::Manager manager1;
    //    manager1.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(restrictions);
    //    manager1.run_passes(f);
    //    ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming.1").run_on_function(f);

    //    ngraph::pass::Manager manager2;
    //    manager2.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
    //    manager2.run_passes(f);
    //    ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming.2").run_on_function(f);

    //    ngraph::pass::Manager manager3;
    //    manager3.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    //    manager3.run_passes(f);
    //    ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming.3").run_on_function(f);
    //}

    std::shared_ptr<ngraph::pass::GraphRewrite> common = manager.register_pass<ngraph::pass::GraphRewrite>();
    common->add_matcher<ngraph::pass::low_precision::AddTransformation>();
    common->add_matcher<ngraph::pass::low_precision::AvgPoolTransformation>();
    common->add_matcher<ngraph::pass::low_precision::ClampTransformation>();
    common->add_matcher<ngraph::pass::low_precision::ConcatTransformation>();
    common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>();
    common->add_matcher<ngraph::pass::low_precision::DepthToSpaceTransformation>();
    common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>();
    common->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::InterpolateTransformation>();
    common->add_matcher<ngraph::pass::low_precision::GroupConvolutionTransformation>();
    common->add_matcher<ngraph::pass::low_precision::MatMulTransformation>();
    common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>();
    common->add_matcher<ngraph::pass::low_precision::MultiplyTransformation>();
    common->add_matcher<ngraph::pass::low_precision::MVNTransformation>();
    common->add_matcher<ngraph::pass::low_precision::NormalizeL2Transformation>();
    common->add_matcher<ngraph::pass::low_precision::PReluTransformation>();
    common->add_matcher<ngraph::pass::low_precision::ReluTransformation>();
    common->add_matcher<ngraph::pass::low_precision::ReshapeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::SqueezeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::SplitTransformation>();
    common->add_matcher<ngraph::pass::low_precision::StridedSliceTransformation>();
    common->add_matcher<ngraph::pass::low_precision::TransposeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::UnsqueezeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::VariadicSplitTransformation>();

    std::shared_ptr<ngraph::pass::GraphRewrite> cleanup = manager.register_pass<ngraph::pass::GraphRewrite>();
    //cleanup->add_matcher<ngraph::pass::low_precision::FoldConvertTransformation>();
    //cleanup->add_matcher<ngraph::pass::low_precision::FuseConvertTransformation>();
    cleanup->add_matcher<ngraph::pass::low_precision::FakeQuantizeTransformation>();
    cleanup->add_matcher<ngraph::pass::low_precision::FuseSubtractToFakeQuantizeTransformation>();
    cleanup->add_matcher<ngraph::pass::low_precision::FuseMultiplyToFakeQuantizeTransformation>();
    cleanup->add_matcher<ngraph::pass::low_precision::MultiplyToGroupConvolutionTransformation>();
    cleanup->add_matcher<ngraph::pass::low_precision::SubtractMultiplyToMultiplyAddTransformation>();

    manager.run_passes(f);

    return true;
}

bool ngraph::pass::low_precision::LowPrecision::isFunctionQuantized(const std::shared_ptr<ngraph::Function>& function) {
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

            const std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = as_type_ptr<ngraph::opset1::FakeQuantize>(parent);
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
