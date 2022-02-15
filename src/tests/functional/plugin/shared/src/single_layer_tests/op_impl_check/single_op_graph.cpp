// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/op_impl_check/op_impl_check.hpp>
#include <single_layer_tests/op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::Op> &node) {
    return nullptr;
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::AdaptiveAvgPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 6, 8, 9});
    const auto out_shape = ov::op::v0::Constant::create<int32_t>(element::i64, {2}, {5, 7});
    const auto adaptiveAvgPoolNode = std::make_shared<ov::op::v8::AdaptiveAvgPool>(data, out_shape);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(adaptiveAvgPoolNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "AdaptiveAvgPoolGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::AdaptiveMaxPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 6, 8, 9});
    const auto out_shape = ov::op::v0::Constant::create<int32_t>(element::i32, {2}, {5, 7});
    const auto adaptiveMaxPoolNode = std::make_shared<ov::op::v8::AdaptiveMaxPool>(data, out_shape, ov::element::i32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(adaptiveMaxPoolNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "AdaptiveMaxPoolGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::AvgPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 32});
    const ov::Strides strides{1};
    const ov::Shape pads_begin{0};
    const ov::Shape pads_end{0};
    const ov::Shape kernel{2};
    const auto exclude_pad = false;
    const auto rounding_type = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;
    const auto avgPoolNode = std::make_shared<ov::op::v1::AvgPool>(data,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   kernel,
                                                                   exclude_pad,
                                                                   rounding_type,
                                                                   auto_pad);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(avgPoolNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "AvgPoolGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::BatchNormInference> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3});
    const auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto mean = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto variance = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto epsilon = 0.25f;
    const auto batchNormInterferenceNode = std::make_shared<ov::op::v0::BatchNormInference>(data,
                                                                                            gamma,
                                                                                            beta,
                                                                                            mean,
                                                                                            variance,
                                                                                            epsilon);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(batchNormInterferenceNode)};
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{data, gamma, beta, mean, variance},
                                       "BatchNormInterferenceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::BatchNormInference> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3});
    const auto gamma = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto beta = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto mean = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto variance = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto epsilon = 0.25f;
    const auto batchNormInterferenceNode = std::make_shared<ov::op::v5::BatchNormInference>(data,
                                                                                            gamma,
                                                                                            beta,
                                                                                            mean,
                                                                                            variance,
                                                                                            epsilon);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(batchNormInterferenceNode)};
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{data, gamma, beta, mean, variance},
                                       "BatchNormInterferenceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::BatchToSpace> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 1, 1, 3});
    const auto block_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, 1, 2});
    const auto crops_begin = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 0, 0, 0});
    const auto crops_end = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 0, 0, 0});
    const auto batchToSpaceNode = std::make_shared<ov::op::v1::BatchToSpace>(data,
                                                                             block_shape,
                                                                             crops_begin,
                                                                             crops_end);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(batchToSpaceNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "BatchToSpaceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::BinaryConvolution> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
    const auto kernel = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 3, 3});
    const ov::Strides strides{1, 1};
    const ov::CoordinateDiff pads_begin{0, 0};
    const ov::CoordinateDiff pads_end{0, 0};
    const ov::Strides dilations{1, 1};
    const auto mode = ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const auto pad_value = 1.0f;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;
    const auto binaryConvolutionNode = std::make_shared<ov::op::v1::BinaryConvolution>(data,
                                                                                       kernel,
                                                                                       strides,
                                                                                       pads_begin,
                                                                                       pads_end,
                                                                                       dilations,
                                                                                       mode,
                                                                                       pad_value,
                                                                                       auto_pad);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(binaryConvolutionNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, kernel}, "BinaryConvolutionGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::Bucketize> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 2});
    const auto buckets = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    const auto bucketizeNode = std::make_shared<ov::op::v3::Bucketize>(data, buckets);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(bucketizeNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, buckets}, "BucketizeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::CTCGreedyDecoder> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{100, 3, 1200});
    const auto sequence_mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{100, 3});
    const auto CTCGreedyDecoderNode = std::make_shared<ov::op::v0::CTCGreedyDecoder>(data, sequence_mask, false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(CTCGreedyDecoderNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, sequence_mask}, "CTCGreedyDecoderGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::CTCGreedyDecoderSeqLen> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 100, 1200});
    const auto sequence_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{3});
    const auto CTCGreedyDecoderSeqLenNode = std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(data, sequence_length);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(CTCGreedyDecoderSeqLenNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, sequence_length}, "CTCGreedyDecoderSeqLenGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::CTCLoss> &node) {
    const auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 120, 28});
    const auto logit_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{10});
    const auto labels = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{10, 120});
    const auto label_length = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{10});
    const auto blank_index = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    const auto CTCLossNode = std::make_shared<ov::op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(CTCLossNode)};
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{logits, logit_length, labels, label_length, blank_index},
                                       "CTCLossGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Clamp> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 120, 28});
    const auto clampNode = std::make_shared<ov::op::v0::Clamp>(data, 0.0, 2.1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clampNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "ClampGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Concat> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3, 4}, {2, 7, 4}, {2, 2, 4}});
    const auto concatNode = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{params[0], params[1], params[2]}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concatNode)};
    return std::make_shared<ov::Model>(results, params, "ConcatGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Constant> &node) {
    const auto A = ov::op::v0::Constant::create(ov::element::f32, {2, 2}, {1, 2, 3, 4});
    const auto B = ov::op::v0::Constant::create(ov::element::f32, {2, 2}, {1, 2, 3, 4});
    return std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{}, "ConstantGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Convert> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 4});
    const auto convertNode = std::make_shared<ov::op::v0::Convert>(param, ov::element::i32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convertNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ConvertGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::ConvertLike> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{256, 56});
    const auto like = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto convertNode = std::make_shared<ov::op::v1::ConvertLike>(data, like);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convertNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, like}, "ConvertLikeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Convolution> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 5, 5});
    const auto kernel = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 3, 3});
    const ov::Strides strides{1, 1};
    const ov::CoordinateDiff pads_begin{0, 0};
    const ov::CoordinateDiff pads_end{0, 0};
    const ov::Strides dilations{1, 1};
    const auto auto_pad = ov::op::PadType::SAME_LOWER;
    const auto convolutionNode = std::make_shared<ov::op::v1::Convolution>(data,
                                                                           kernel,
                                                                           strides,
                                                                           pads_begin,
                                                                           pads_end,
                                                                           dilations,
                                                                           auto_pad);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convolutionNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, kernel}, "ConvolutionGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::ConvolutionBackpropData> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 512, 1, 37});
    const auto kernel = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{512, 256, 1, 1});
    const auto output_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 74});
    const ov::Strides strides{1, 2};
    const ov::CoordinateDiff pads_begin{0, 0};
    const ov::CoordinateDiff pads_end{0, 0};
    const ov::Strides dilations{1, 1};
    const auto auto_pad = ov::op::PadType::SAME_LOWER;
    const auto convolutionBackpropDataNode = std::make_shared<ov::op::v1::ConvolutionBackpropData>(data,
                                                                                                   kernel,
                                                                                                   output_shape,
                                                                                                   strides,
                                                                                                   pads_begin,
                                                                                                   pads_end,
                                                                                                   dilations,
                                                                                                   auto_pad);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convolutionBackpropDataNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, kernel}, "ConvolutionBackpropDataGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::CumSum> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    const auto cumSumNode = std::make_shared<ov::op::v0::CumSum>(data);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(cumSumNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "CumSumGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::DeformablePSROIPooling> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 7938, 63, 38});
    const auto coord = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{300, 5});
    const auto deformablePSROIPoolingNode = std::make_shared<ov::op::v1::DeformablePSROIPooling>(data, coord, 882, 0.0625, 3);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(deformablePSROIPoolingNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, coord}, "DeformablePSROIPoolingGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::DepthToSpace> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 3, 1080, 1616});
    const auto depthToSpaceNode = std::make_shared<ov::op::v0::DepthToSpace>(data, ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(depthToSpaceNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "DepthToSpaceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v7::Einsum> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3}, {3}});
    const auto einsumNode = std::make_shared<ov::op::v7::Einsum>(ov::OutputVector{params.front(), params.back()}, "i,i->");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(einsumNode)};
    return std::make_shared<ov::Model>(results, params, "EinsumGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2},
                                                                              {1, 2}});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v0::SquaredDifference>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::SquaredDifference>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Add>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Add>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Divide>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Divide>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::FloorMod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::FloorMod>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Maximum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Maximum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Minimum>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Minimum>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Mod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Mod>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Multiply>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Multiply>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Power>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Power>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Subtract>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Subtract>(params.front(), params.back());
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseNode)};
    return std::make_shared<ov::Model>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateArithmeticReductionKeepDims(const std::shared_ptr<ov::op::Op> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 3});
    const auto axes = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    std::shared_ptr<ov::Node> reduceNode;
    if (ov::is_type<ov::op::v4::ReduceL1>(node)) {
        reduceNode = std::make_shared<ov::op::v4::ReduceL1>(data, axes, true);
    } else if (ov::is_type<ov::op::v4::ReduceL2>(node)) {
        reduceNode = std::make_shared<ov::op::v4::ReduceL2>(data, axes, true);
    } else if (ov::is_type<ov::op::v1::ReduceMax>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceMax>(data, axes, true);
    } else if (ov::is_type<ov::op::v1::ReduceMean>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceMean>(data, axes, true);
    } else if (ov::is_type<ov::op::v1::ReduceMin>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceMin>(data, axes, true);
    } else if (ov::is_type<ov::op::v1::ReduceProd>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceProd>(data, axes, true);
    } else if (ov::is_type<ov::op::v1::ReduceSum>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceSum>(data, axes, true);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reduceNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "ArithmeticReductionKeepDimsGraph");
}

std::shared_ptr<ov::Model> generateLogicalReductionKeepDims(const std::shared_ptr<ov::op::Op> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::PartialShape{3, 3});
    const auto axes = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    std::shared_ptr<ov::Node> reduceNode;
    if (ov::is_type<ov::op::v1::ReduceLogicalAnd>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceLogicalAnd>(data, axes, false);
    } else if (ov::is_type<ov::op::v1::ReduceLogicalOr>(node)) {
        reduceNode = std::make_shared<ov::op::v1::ReduceLogicalOr>(data, axes, false);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reduceNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "LogicalReductionKeepDimsGraph");
}

std::shared_ptr<ov::Model> generateMaxPoolBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 7, 3});
    const ov::Strides strides{1};
    const ov::Strides dilations{1};
    const ov::Shape pads_begin{0};
    const ov::Shape pads_end{0};
    const ov::Shape kernel_shape{3};
    const auto rounding_mode = ov::op::RoundingType::FLOOR;
    const auto auto_pad = ov::op::PadType::VALID;
    std::shared_ptr<ov::Node> maxPoolNode;
    if (ov::is_type<ov::op::v1::MaxPool>(node)) {
        maxPoolNode = std::make_shared<ov::op::v1::MaxPool>(data, strides, pads_begin, pads_end, kernel_shape, rounding_mode, auto_pad);
    } else if (ov::is_type<ov::op::v8::MaxPool>(node)) {
        maxPoolNode = std::make_shared<ov::op::v8::MaxPool>(data, strides, dilations, pads_begin, pads_end, kernel_shape);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(maxPoolNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "MaxPoolBaseGraph");
}

std::shared_ptr<ov::Model> generateScatterBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 4});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2, 1});
    const auto updates = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 1, 4});
    const auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    std::shared_ptr<ov::Node> scatterNode;
    if (ov::is_type<ov::op::v3::ScatterUpdate>(node)) {
        scatterNode = std::make_shared<ov::op::v3::ScatterUpdate>(data, indices, updates, axis);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scatterNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, indices, updates}, "ScatterBaseGraph");
}

std::shared_ptr<ov::Model> generateScatterNDBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2, 1});
    const auto updates = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2});
    std::shared_ptr<ov::Node> scatterNode;
    if (ov::is_type<ov::op::v3::ScatterNDUpdate>(node)) {
        scatterNode = std::make_shared<ov::op::v3::ScatterNDUpdate>(data, indices, updates);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scatterNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, indices, updates}, "ScatterNDBaseGraph");
}

std::shared_ptr<ov::Model> generateUnaryEltwise(const std::shared_ptr<ov::op::Op> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v0::Abs>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Abs>(param);
    } else if (ov::is_type<ov::op::v0::Acos>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Acos>(param);
    } else if (ov::is_type<ov::op::v3::Acosh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v3::Acosh>(param);
    } else if (ov::is_type<ov::op::v0::Asin>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Asin>(param);
    } else if (ov::is_type<ov::op::v3::Asinh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v3::Asinh>(param);
    } else if (ov::is_type<ov::op::v0::Atan>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Atan>(param);
    } else if (ov::is_type<ov::op::v3::Atanh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v3::Atanh>(param);
    } else if (ov::is_type<ov::op::v0::Ceiling>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Ceiling>(param);
    } else if (ov::is_type<ov::op::v0::Cos>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Cos>(param);
    } else if (ov::is_type<ov::op::v0::Cosh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Cosh>(param);
    } else if (ov::is_type<ov::op::v0::Erf>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Erf>(param);
    } else if (ov::is_type<ov::op::v0::Exp>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Exp>(param);
    } else if (ov::is_type<ov::op::v0::Floor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Floor>(param);
    } else if (ov::is_type<ov::op::v7::Gelu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v7::Gelu>(param);
    } else if (ov::is_type<ov::op::v5::HSigmoid>(node)) {
        eltwiseNode = std::make_shared<ov::op::v5::HSigmoid>(param);
    } else if (ov::is_type<ov::op::v4::HSwish>(node)) {
        eltwiseNode = std::make_shared<ov::op::v4::HSwish>(param);
    } else if (ov::is_type<ov::op::v0::Log>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Log>(param);
    } else if (ov::is_type<ov::op::v0::Negative>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Negative>(param);
    } else if (ov::is_type<ov::op::v0::Relu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Relu>(param);
    } else if (ov::is_type<ov::op::v0::Sigmoid>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sigmoid>(param);
    } else if (ov::is_type<ov::op::v0::Sign>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sign>(param);
    } else if (ov::is_type<ov::op::v0::Sin>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sin>(param);
    } else if (ov::is_type<ov::op::v0::Sinh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sinh>(param);
    } else if (ov::is_type<ov::op::v0::Sqrt>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sqrt>(param);
    } else if (ov::is_type<ov::op::v0::Tan>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Tan>(param);
    } else if (ov::is_type<ov::op::v0::Tanh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Tanh>(param);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "UnaryEltwiseGraph");
}
} // namespace

template <typename T>
std::shared_ptr<ov::Model> generateGraph() {
        std::shared_ptr<T> node = std::shared_ptr<T>(new T);
    if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
        return generateBinaryEltwise(node);
    } else if (ov::is_type<ov::op::util::ArithmeticReductionKeepDims>(node)) {
        return generateArithmeticReductionKeepDims(node);
    } else if (ov::is_type<ov::op::util::LogicalReductionKeepDims>(node)) {
        return generateLogicalReductionKeepDims(node);
    } else if (ov::is_type<ov::op::util::MaxPoolBase>(node)) {
        return generateMaxPoolBase(node);
    } else if (ov::is_type<ov::op::util::ScatterBase>(node)) {
        return generateScatterBase(node);
    } else if (ov::is_type<ov::op::util::ScatterNDBase>(node)) {
        return generateScatterNDBase(node);
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(node)) {
        return generateUnaryEltwise(node);
    }
    return generate(node);
}

OpGenerator getOpGeneratorMap() {
    static OpGenerator opGeneratorMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateGraph<NAMESPACE::NAME>},
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return opGeneratorMap;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov
