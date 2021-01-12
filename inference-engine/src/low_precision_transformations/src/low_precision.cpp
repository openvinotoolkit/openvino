// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/low_precision.hpp"
#include <memory>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "low_precision/markup_precisions.hpp"
#include "low_precision/markup_avg_pool_precisions.hpp"
#include "low_precision/propagate_precisions.hpp"
#include "low_precision/align_concat_quantization_parameters.hpp"

//#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include "low_precision/pull_reshape_through_dequantization.hpp"
#include "low_precision/pull_transpose_through_dequantization.hpp"

// branch specific transformations
#include "low_precision/concat.hpp"
#include "low_precision/concat_multi_channels.hpp"

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
#include "low_precision/transpose.hpp"
#include "low_precision/unsqueeze.hpp"
#include "low_precision/variadic_split.hpp"
#include "low_precision/split.hpp"

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
    const LayerTransformation::Params params) : restrictions(restrictions), params(params){
}

bool ngraph::pass::low_precision::LowPrecision::run_on_function(std::shared_ptr<ngraph::Function> f) {
    // TODO: to debug only
    // TransformationContext context(f);

    // pass config should be reused
    const std::vector<ngraph::element::Type> supportedTypes = { ngraph::element::i8, ngraph::element::u8 };
    ngraph::pass::Manager manager;
    manager.register_pass<PullReshapeThroughDequantization>(supportedTypes);
    manager.register_pass<PullTransposeThroughDequantization>(supportedTypes);
    //prerequisites.register_pass<ngraph::pass::LinOpSequenceFusion>();

    manager.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(restrictions);
    manager.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
    manager.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    manager.register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();

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
    //common->add_matcher<ngraph::pass::low_precision::SplitTransformation>();
    //common->add_matcher<ngraph::pass::low_precision::StridedSliceTransformation>();
    common->add_matcher<ngraph::pass::low_precision::TransposeTransformation>();
    common->add_matcher<ngraph::pass::low_precision::UnsqueezeTransformation>();
    //common->add_matcher<ngraph::pass::low_precision::VariadicSplit>();

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
