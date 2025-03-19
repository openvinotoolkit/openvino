// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_impl_check/op_impl_check.hpp"
#include "op_impl_check/single_op_graph.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace op_conformance {

namespace {
std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::Op> &node) {
    return nullptr;
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::AdaptiveAvgPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 6, 8, 9});
    const auto out_shape = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {2}, {5, 7});
    const auto adaptiveAvgPoolNode = std::make_shared<ov::op::v8::AdaptiveAvgPool>(data, out_shape);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(adaptiveAvgPoolNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "AdaptiveAvgPoolGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::AdaptiveMaxPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 6, 8, 9});
    const auto out_shape = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {2}, {5, 7});
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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v14::AvgPool> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 32});
    const ov::Strides strides{1};
    const ov::Shape pads_begin{0};
    const ov::Shape pads_end{0};
    const ov::Shape kernel{2};
    const auto exclude_pad = false;
    const auto rounding_type = ov::op::RoundingType::CEIL_TORCH;
    const auto auto_pad = ov::op::PadType::SAME_LOWER;
    const auto avgPoolNode = std::make_shared<ov::op::v14::AvgPool>(data,
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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v12::GroupNormalization>& node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3, 14, 5, 5});
    const auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{14});
    const auto bias = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{14});
    const auto gn = std::make_shared<ov::op::v12::GroupNormalization>(data, scale, bias, 7, 0.00001f);
    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gn)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, scale, bias}, "GroupNormalizationGraph");
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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Concat> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 7, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 4}})};

    const auto concatNode = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{params[0], params[1], params[2]}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concatNode)};
    return std::make_shared<ov::Model>(results, params, "ConcatGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Constant> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};
    const auto constantNode = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 2}, 2.0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(constantNode)};
    return std::make_shared<ov::Model>(results, params, "ConstantGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Convert> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 4});
    const auto convertNode = std::make_shared<ov::op::v0::Convert>(param, ov::element::i32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convertNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ConvertGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v14::ConvertPromoteTypes> &node) {
    const auto lhs = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{256, 56});
    const auto rhs = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{3});
    const auto convertNode = std::make_shared<ov::op::v14::ConvertPromoteTypes>(lhs, rhs, true);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convertNode->output(0)),
                             std::make_shared<ov::op::v0::Result>(convertNode->output(1))};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{lhs, rhs}, "ConvertPromoteTypesGraph");
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
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3})};
    const auto einsumNode = std::make_shared<ov::op::v7::Einsum>(ov::OutputVector{params.front(), params.back()}, "i,i->");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(einsumNode)};
    return std::make_shared<ov::Model>(results, params, "EinsumGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::EmbeddingSegmentsSum> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 2}})};
    const auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 2, 3, 4});
    const auto segment_ids = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 0, 2, 2});
    const auto num_segments = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{3});
    const auto default_index = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
    const auto per_sample_weights =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{4}, std::vector<float>{0.5, 0.5, 0.5, 0.5});
    const auto embed_seg_sum = std::make_shared<ov::op::v3::EmbeddingSegmentsSum>(params[0],
                                                                                    indices,
                                                                                    segment_ids,
                                                                                    num_segments,
                                                                                    default_index,
                                                                                    per_sample_weights);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(embed_seg_sum)};
    return std::make_shared<ov::Model>(results, params, "EmbeddingSegmentsSum");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{16, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{16, 8}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{16, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3}})};
    const auto attrs = ov::op::v6::ExperimentalDetectronDetectionOutput::Attributes{0.01000000074505806f,
                                                                                    0.2f,
                                                                                    2.0f,
                                                                                    2,
                                                                                    500,
                                                                                    5,
                                                                                    true,
                                                                                    {10.0f, 10.0f, 5.0f, 5.0f}};
    const auto exp_detection_output =
        std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(params.at(0), params.at(1), params.at(2), params.at(3), attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_detection_output)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronDetectionOutput");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{36, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{12, 2, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{3, 2, 6}})};

    const auto attrs =
        ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes{0, 0.699999988079071, 6, 1000};
    const auto exp_gen_prop_sing_img =
        std::make_shared<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(params.at(0),
                                                                                        params.at(1),
                                                                                        params.at(2),
                                                                                        params.at(3),
                                                                                        attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_gen_prop_sing_img)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronGenerateProposalsSingleImage");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{3, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 16, 4, 5}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3, 100, 200}})};

    const auto attrs = ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes{true, 0, 0, 4.0f, 4.0f};
    const auto exp_prior_grid_gen = std::make_shared<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(params[0],
                                                                                                            params[1],
                                                                                                            params[2],
                                                                                                            attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_prior_grid_gen)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronPriorGridGenerator");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 2, 3}})};

    const auto attrs = ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes{3, 2, {4}, false};
    const auto exp_roi_feature_ext =
        std::make_shared<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(NodeVector{params[0], params[1]},
                                                                                attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_roi_feature_ext)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronROIFeatureExtractor");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronTopKROIs> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2})};

    const auto exp_topk_rois = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(params[0], params[1], 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_topk_rois)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronTopKROIs");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ExtractImagePatches> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 10, 10}})};
    const auto ext_img_patch = std::make_shared<ov::op::v3::ExtractImagePatches>(params[0],
                                                                                 ov::Shape{3, 3},
                                                                                 ov::Strides{5, 5},
                                                                                 ov::Shape{1, 1},
                                                                                 ov::op::PadType::VALID);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ext_img_patch)};
    return std::make_shared<ov::Model>(results, params, "ExtractImagePatches");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v9::Eye> &node) {
    const auto rows = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    const auto cols = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{4});
    const auto diag = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    const auto batch = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 2, 2});
    const auto eye = std::make_shared<ov::op::v9::Eye>(rows,
                                                       cols,
                                                       diag,
                                                       batch,
                                                       ov::element::f32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eye)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{rows}, "Eye");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::FakeQuantize> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 3, 4}})};
    const auto input_low = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.f});
    const auto input_high = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{23.f});
    const auto output_low = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{2.f});
    const auto output_high = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{16.f});
    const auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(params[0], input_low, input_high, output_low, output_high, 4);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fake_quantize)};
    return std::make_shared<ov::Model>(results, params, "FakeQuantize");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::GRUSequence> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 10, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 1, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{5})};
    const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30, 10}));
    const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30, 10}));
    const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30}));
    const size_t hidden_size = 10;
    const auto gru_sequence =
        std::make_shared<ov::op::v5::GRUSequence>(params[0],
                                                   params[1],
                                                   params[2],
                                                   W,
                                                   R,
                                                   B,
                                                   hidden_size,
                                                   ov::op::RecurrentSequenceDirection::FORWARD);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_sequence)};
    return std::make_shared<ov::Model>(results, params, "GRUSequence");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::GatherElements> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{7})};
    const auto gather_elements = std::make_shared<ov::op::v6::GatherElements>(params[0], params[1], 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_elements)};
    return std::make_shared<ov::Model>(results, params, "GatherElements");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GatherTree> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 10}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 10}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};
    const auto gather_tree = std::make_shared<ov::op::v1::GatherTree>(params[0], params[1], params[2], params[3]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_tree)};
    return std::make_shared<ov::Model>(results, params, "GatherTree");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GroupConvolution> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 1, 3}})};
    const auto group_convolution = std::make_shared<ov::op::v1::GroupConvolution>(params[0],
                                                                                  params[1],
                                                                                  ov::Strides{1},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::Strides{1});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(group_convolution)};
    return std::make_shared<ov::Model>(results, params, "GroupConvolution");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 1, 3}})};

    const auto group_convolution = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(params[0],
                                                                                  params[1],
                                                                                  ov::Strides{1},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::Strides{1},
                                                                                  ov::op::PadType{ov::op::PadType::EXPLICIT});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(group_convolution)};
    return std::make_shared<ov::Model>(results, params, "GroupConvolutionBackpropData");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::HardSigmoid> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3})};
    const auto alpha = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.5});
    const auto beta = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.6});
    const auto hard_sigmoid = std::make_shared<ov::op::v0::HardSigmoid>(params[0], alpha, beta);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(hard_sigmoid)};
    return std::make_shared<ov::Model>(results, params, "HardSigmoid");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Interpolate> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 2, 4}})};
    const auto out_shape_in = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 2});
    ov::op::v0::Interpolate::Attributes attrs;
    attrs.axes = ov::AxisSet{0, 1, 2, 3};
    attrs.mode = "nearest";
    attrs.align_corners = false;
    attrs.antialias = false;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    const auto interpolate = std::make_shared<ov::op::v0::Interpolate>(params[0], out_shape_in, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(interpolate)};
    return std::make_shared<ov::Model>(results, params, "Interpolat-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v16::Identity>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4, 4})};
    const auto identity = std::make_shared<ov::op::v16::Identity>(params[0]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(identity)};
    return std::make_shared<ov::Model>(results, params, "Identity");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v16::SegmentMax> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 9});
    const auto segment_ids = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {2}, {4, 4});
    const auto num_segments = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {5});
    const auto SegmentMaxNode = std::make_shared<ov::op::v16::SegmentMax>(data, segment_ids, num_segments, ov::op::FillMode::ZERO);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(SegmentMaxNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "SegmentMaxGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Interpolate> &node) {
    using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;
    using TransformMode = op::v4::Interpolate::CoordinateTransformMode;
    using NearestMode = op::v4::Interpolate::NearestMode;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 30, 60}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{{15, 30}})};
    const auto scales = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{0.5f, 0.5f});
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
    const InterpolateAttrs attrs{InterpolateMode::NEAREST,
                                 ShapeCalcMode::SCALES,
                                 std::vector<size_t>{0, 0, 0, 0},
                                 std::vector<size_t>{0, 0, 0, 0},
                                 TransformMode::HALF_PIXEL,
                                 NearestMode::ROUND_PREFER_FLOOR,
                                 false,
                                 -0.75};
    const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(params[0], params[1], scales, axes, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(interpolate)};
    return std::make_shared<ov::Model>(results, params, "Interpolate-4");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v11::Interpolate> &node) {
    using InterpolateAttrs = op::v11::Interpolate::InterpolateAttrs;
    using InterpolateMode = op::v11::Interpolate::InterpolateMode;
    using ShapeCalcMode = op::v11::Interpolate::ShapeCalcMode;
    using TransformMode = op::v11::Interpolate::CoordinateTransformMode;
    using NearestMode = op::v11::Interpolate::NearestMode;
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 2, 30, 60});
    const auto scales = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{0.5f, 0.5f});
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
    const InterpolateAttrs attrs{InterpolateMode::BILINEAR_PILLOW,
                                 ShapeCalcMode::SCALES,
                                 std::vector<size_t>{0, 0, 0, 0},
                                 std::vector<size_t>{0, 0, 0, 0},
                                 TransformMode::HALF_PIXEL,
                                 NearestMode::ROUND_PREFER_FLOOR,
                                 false,
                                 -0.75};
    const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(data, scales, axes, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(interpolate)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{{data}}, "Interpolate-11");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v14::Inverse>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 4, 4})};
    const auto inverse = std::make_shared<ov::op::v14::Inverse>(params[0], false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(inverse)};
    return std::make_shared<ov::Model>(results, params, "Inverse");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::Assign> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1})};
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(params[0], "v0");
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "Assign-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::Assign> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1})};
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(params[0], variable);
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "Assign-6");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LRN> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3, 2, 1}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    const auto lrn = std::make_shared<ov::op::v0::LRN>(params[0], axes, 3, 0.5, 1, 3);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lrn)};
    return std::make_shared<ov::Model>(results, params, "LRN");
}


std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::LogSoftmax> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1})};
    const auto lsm = std::make_shared<ov::op::v5::LogSoftmax>(params[0], 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lsm)};
    return std::make_shared<ov::Model>(results, params, "LogSoftmax");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::LogicalNot> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{{1, 2}})};
    const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(params[0]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(logical_not)};
    return std::make_shared<ov::Model>(results, params, "LogicalNot");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::MVN> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3, 3, 3}})};
    const auto mvn = std::make_shared<ov::op::v0::MVN>(params[0], false, false, 1e-9);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mvn)};
    return std::make_shared<ov::Model>(results, params, "MVN-2");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::MVN> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3, 3, 3}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3});
    const auto mvn = std::make_shared<ov::op::v6::MVN>(params[0], axes, false, 1e-9, ov::op::MVNEpsMode::OUTSIDE_SQRT);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mvn)};
    return std::make_shared<ov::Model>(results, params, "MVN-6");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::MatMul> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], params[1], false, false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
    return std::make_shared<ov::Model>(results, params, "MatMul-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v13::Multinomial>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 5}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1})};
    auto multinomial =
        std::make_shared<ov::op::v13::Multinomial>(params[0], params[1], ov::element::i32, false, false, 0, 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(multinomial)};
    return std::make_shared<ov::Model>(results, params, "Multinomial-13");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v13::NMSRotated> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 5}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v13::NMSRotated>(params[0],
                                                         params[1],
                                                         params[2],
                                                         params[3],
                                                         params[4],
                                                         true,
                                                         ov::element::i32,
                                                         true);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NMSRotated-13");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::NonMaxSuppression> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v1::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v1::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::NonMaxSuppression> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v3::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v3::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::NonMaxSuppression> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v4::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v4::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-4");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::NonMaxSuppression> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v5::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-5");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v9::NonMaxSuppression> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 6, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{})};

    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-9");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::NonZero> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{3, 2}})};
    auto nonzero = std::make_shared<ov::op::v3::NonZero>(params[0], ov::element::i32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nonzero)};
    return std::make_shared<ov::Model>(results, params, "NonZero-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::NormalizeL2> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{0}, std::vector<int64_t>{});
    auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(params[0], axes, 1e-7, ov::op::EpsMode::ADD);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(normalize)};
    return std::make_shared<ov::Model>(results, params, "NormalizeL2-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::OneHot> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{})};
    const auto depth = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{3});
    const auto onvalue = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{1});
    const auto offvalue = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{0});
    const int32_t axes = 0;
    const auto onehot = std::make_shared<ov::op::v1::OneHot>(params[0], depth, onvalue, offvalue, axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(onehot)};
    return std::make_shared<ov::Model>(results, params, "OneHot-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::PRelu> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6})};
    const auto slope = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{2});
    const auto prelu = std::make_shared<ov::op::v0::PRelu>(params[0], slope);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(prelu)};
    return std::make_shared<ov::Model>(results, params, "PRelu-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::PSROIPooling> &node) {
    const std::string mode = "average";
    const size_t n_channel = 8;
    const size_t n_group = 2;
    const size_t n_boxes = 3;
    const size_t spatial_bin_x = 1;
    const size_t spatial_bin_y = 1;
    const float spatial_scale = 1;
    const size_t output_dim = n_channel / (n_group * n_group);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, n_channel, 20, 20}})};
    const auto coordi = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                               ov::Shape{n_boxes, 5},
                                                               std::vector<float>{0, 1, 2, 4, 6, 1, 0, 3, 10, 4, 0, 10, 7, 11, 13});
    const auto psroi_pooling = std::make_shared<ov::op::v0::PSROIPooling>(params[0],
                                                                          coordi,
                                                                          output_dim,
                                                                          n_group,
                                                                          spatial_scale,
                                                                          spatial_bin_x,
                                                                          spatial_bin_y,
                                                                          mode);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(psroi_pooling)};
    return std::make_shared<ov::Model>(results, params, "PSROIPooling-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Pad> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{6})};
    const auto pad_begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{4});
    const auto pad_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, ov::Shape{5});
    const auto pad = std::make_shared<ov::op::v1::Pad>(params[0], pad_begin, pad_end, ov::op::PadMode::CONSTANT);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(pad)};
    return std::make_shared<ov::Model>(results, params, "Pad-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v12::Pad> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{6, 10, 11, 12}})};
    const auto pad_begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{4, -2, 3, -1});
    const auto pad_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{5, -1, -4, 4});
    const auto pad = std::make_shared<ov::op::v12::Pad>(params[0], pad_begin, pad_end, ov::op::PadMode::CONSTANT);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(pad)};
    return std::make_shared<ov::Model>(results, params, "Pad-12");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Parameter> &node) {
    const auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, Shape{3, 4});
    return std::make_shared<ov::Model>(in, ParameterVector{in}, "Parameter-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::PriorBox> &node) {
    ov::op::v0::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f};
    attrs.aspect_ratio = {1.5f};
    attrs.scale_all_sizes = false;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{{300, 300}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{{32, 32}})};

    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params.at(0));
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params.at(1));
    auto Node = std::make_shared<ov::op::v0::PriorBox>(shape_of_1, shape_of_2, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "PrioBoxGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::PriorBox> &node) {
    ov::op::v8::PriorBox::Attributes attrs;
    attrs.min_size = {2.0f};
    attrs.max_size = {5.0f};
    attrs.aspect_ratio = {1.5f};
    attrs.scale_all_sizes = true;
    attrs.min_max_aspect_ratios_order = false;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{{300, 300}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::u16, ov::Shape{{32, 32}})};
    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params.at(0));
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params.at(1));
    auto Node = std::make_shared<ov::op::v8::PriorBox>(shape_of_1, shape_of_2, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "PrioBoxGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::PriorBoxClustered> &node) {
    ov::op::v0::PriorBoxClustered::Attributes attrs;
    attrs.widths = {3.0f};
    attrs.heights = {3.0f};
    attrs.clip = true;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{4, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{50, 50}})};
    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params.at(0));
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params.at(1));
    auto Node = std::make_shared<ov::op::v0::PriorBoxClustered>(shape_of_1, shape_of_2, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "PrioBoxClustedGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Proposal> &node) {
    ov::op::v0::Proposal::Attributes attrs;
    attrs.base_size = 16;
    attrs.min_size = 16;
    attrs.pre_nms_topn = 6000;
    attrs.post_nms_topn = 10;
    attrs.nms_thresh = 0.7f;
    attrs.feat_stride = 16;
    attrs.min_size = 16;
    attrs.ratio = {0.5f};
    attrs.scale = {32.0f};
    attrs.clip_before_nms = true;
    attrs.clip_after_nms = false;
    attrs.normalize = false;
    attrs.box_size_scale = 1.0f;
    attrs.box_coordinate_scale = 1.0f;
    attrs.framework = "";
    attrs.infer_probs = false;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 10, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 4, 10, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3})};
    auto Node = std::make_shared<ov::op::v0::Proposal>(params.at(0), params.at(1), params.at(2), attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ProposalGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Proposal> &node) {
    ov::op::v4::Proposal::Attributes attrs;
    attrs.base_size = 16;
    attrs.min_size = 16;
    attrs.pre_nms_topn = 6000;
    attrs.post_nms_topn = 10;
    attrs.nms_thresh = 0.7f;
    attrs.feat_stride = 16;
    attrs.min_size = 16;
    attrs.ratio = {0.5f};
    attrs.scale = {32.0f};
    attrs.clip_before_nms = true;
    attrs.clip_after_nms = false;
    attrs.normalize = false;
    attrs.box_size_scale = 1.0f;
    attrs.box_coordinate_scale = 1.0f;
    attrs.framework = "";
    attrs.infer_probs = true;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 10, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 4, 10, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3})};
    auto Node = std::make_shared<ov::op::v4::Proposal>(params.at(0), params.at(1), params.at(2), attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ProposalGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ROIAlign> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 1, 16, 16}})};
    const auto coords = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 4}, std::vector<float>{2, 2, 8, 8, 2, 2, 8, 8});
    const auto roisIdx = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 1});
    auto Node = std::make_shared<ov::op::v3::ROIAlign>(params.at(0), coords, roisIdx, 2, 2, 2, 1, "avg");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ROIAlignGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v9::ROIAlign>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 1, 16, 16}})};
    const auto coords = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 4}, std::vector<float>{2, 2, 8, 8, 2, 2, 8, 8});
    const auto roisIdx = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 1});
    const auto pooling_mode = EnumNames<op::v9::ROIAlign::PoolingMode>::as_enum("avg");
    const auto aligned_mode = EnumNames<op::v9::ROIAlign::AlignedMode>::as_enum("half_pixel_for_nn");
    auto Node =
        std::make_shared<ov::op::v9::ROIAlign>(params.at(0), coords, roisIdx, 2, 2, 2, 1, pooling_mode, aligned_mode);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ROIAlignGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::ROIAlignRotated>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 16, 16}})};
    const auto coords = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32,
        ov::Shape{1, static_cast<size_t>(node->get_rois_input_second_dim_size())},
        std::vector<float>(node->get_rois_input_second_dim_size(), 0));
    const auto roisIdx =
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
    auto new_node = std::make_shared<ov::op::v15::ROIAlignRotated>(params.at(0), coords, roisIdx, 2, 2, 2, 1, true);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(new_node)};
    return std::make_shared<ov::Model>(results, params, "ROIAlignRotatedGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::ROIPooling> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3, 8, 8}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 5}})};
    auto Node = std::make_shared<ov::op::v0::ROIPooling>(params.at(0), params.at(1), Shape{1, 1}, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ROIPoolingGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::RandomUniform> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3})};
    const auto min_value = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.f});
    const auto max_value = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{1.f});
    auto Node = std::make_shared<ov::op::v8::RandomUniform>(params.at(0), min_value, max_value, ov::element::f32, 10, 10);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "RandomUniformGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Range> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape())};
    auto Node = std::make_shared<ov::op::v0::Range>(params.at(0), params.at(1), params.at(2));
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "RangeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Range> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape()),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape())};
    auto Node = std::make_shared<ov::op::v4::Range>(params.at(0), params.at(1), params.at(2), ov::element::f32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "RangeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::RegionYolo> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 8, 2, 2}})};
    auto Node = std::make_shared<ov::op::v0::RegionYolo>(params.at(0), 4, 1, 1, true, std::vector<int64_t>{0}, 1, 3);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "RegionYoloGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::ReorgYolo> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 8, 4, 4}})};
    auto Node = std::make_shared<ov::op::v0::ReorgYolo>(params.at(0), ov::Strides{2});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ReorgYoloGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Reshape> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 3}})};
    const auto shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{12});
    auto Node = std::make_shared<ov::op::v1::Reshape>(params.at(0), shape, false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ReshapeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Result> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 2})};
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(params.at(0))};
    return std::make_shared<ov::Model>(results, params, "ResultGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Reverse> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4, 3}})};
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2});
    auto Node = std::make_shared<ov::op::v1::Reverse>(params.at(0), axis, op::v1::Reverse::Mode::INDEX);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ReverseGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::ReverseSequence  > &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{3, 10}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3})};

    auto Node = std::make_shared<ov::op::v0::ReverseSequence>(params.at(0), params.at(1), 0, 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ReverseSequenceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v7::Roll> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{4, 2, 3}})};
    const auto shift = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 3});
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2});
    auto Node = std::make_shared<ov::op::v7::Roll>(params.at(0), shift, axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "RollGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v13::ScaledDotProductAttention> &node) {
    const auto query = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    const auto key = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 5, 4});
    const auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 5, 6});
    const auto attention_mask = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 5});
    const auto scale = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto causal = false;

    const auto op =
        std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, attention_mask, scale, causal);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(op)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{query, key, value, attention_mask, scale}, "ScaledDotProductAttentionGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ScatterElementsUpdate> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};
    const auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2, 2}, std::vector<int64_t>{1, 1, 0, 0});
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto Node = std::make_shared<ov::op::v3::ScatterElementsUpdate>(params.at(0), indices, params.at(1), axis);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ScatterElementsUpdateGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v12::ScatterElementsUpdate> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};
    const auto indices = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2, 2}, std::vector<int64_t>{1, 1, 0, 0});
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto Node = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        params.at(0), indices, params.at(1), axis, ov::op::v12::ScatterElementsUpdate::Reduction::SUM);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ScatterElementsUpdateGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Select> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{{2, 2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 2}})};

    auto Node = std::make_shared<ov::op::v1::Select>(params.at(0), params.at(1), params.at(2), op::AutoBroadcastType::NONE);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SelectGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Selu> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3})};
    const auto alpha = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1.67326324});
    const auto lambda = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1.05070098});
    auto Node = std::make_shared<ov::op::v0::Selu>(params.at(0), alpha, lambda);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SeluGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::ShapeOf> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4, 8, 16, 64}})};
    auto Node = std::make_shared<ov::op::v0::ShapeOf>(params.at(0));
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ShapeOfGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ShapeOf> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4, 8, 16, 64}})};
    auto Node = std::make_shared<ov::op::v3::ShapeOf>(params.at(0));
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ShapeOfGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::ShuffleChannels> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 15, 2, 2}})};
    auto Node = std::make_shared<ov::op::v0::ShuffleChannels>(params.at(0), 1, 5);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "ShuffleChannelsGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::Slice> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4, 3}})};
    const auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 4});
    const auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 4, -5});
    const auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{3, 2, -2});
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2});
    auto Node = std::make_shared<ov::op::v8::Slice>(params.at(0), start, stop, step, axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SliceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Softmax> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 3}})};
    auto Node = std::make_shared<ov::op::v1::Softmax>(params.at(0), 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SoftmaxGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::Softmax> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 3}})};
    auto Node = std::make_shared<ov::op::v8::Softmax>(params.at(0), 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SoftmaxGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::SpaceToBatch> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 1, 3, 2, 1}})};
    const auto blockShape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, 1, 3, 2, 2});
    const auto padsBegin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 1, 0, 3});
    const auto padsEnd = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 2, 0, 0});
    auto Node = std::make_shared<ov::op::v1::SpaceToBatch>(params.at(0), blockShape, padsBegin, padsEnd);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SpaceToBatchGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::SpaceToDepth> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 4, 4}})};
    auto Node = std::make_shared<ov::op::v0::SpaceToDepth>(params.at(0), "BLOCKS_FIRST", 2);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SpaceToDepthGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Split> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 8, 2}})};
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto Node = std::make_shared<ov::op::v1::Split>(params.at(0), axis, 4);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SplitGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Squeeze> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 4, 1, 1, 2}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 2});
    auto Node = std::make_shared<ov::op::v0::Squeeze>(params.at(0), axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SqueezeV0Graph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::Squeeze> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 4, 1, 1, 2}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 2});
    auto Node = std::make_shared<ov::op::v15::Squeeze>(params.at(0), axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SqueezeV15Graph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::StridedSlice> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{128, 1}})};
    const auto begin = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 0});
    const auto end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 0});
    const auto stride = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 1, 1});
    auto Node = std::make_shared<ov::op::v1::StridedSlice>(params.at(0), begin, end, stride,
                                                           std::vector<int64_t>{0, 1, 1},
                                                           std::vector<int64_t>{0, 1, 1},
                                                           std::vector<int64_t>{1, 0, 0},
                                                           std::vector<int64_t>{1, 0, 0},
                                                           std::vector<int64_t>{0, 0, 0});
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "StridedSliceGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Swish> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4}})};
    const auto beta = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{0.6f});
    auto Node = std::make_shared<ov::op::v4::Swish>(params.at(0), beta);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SwishGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Tile> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 1, 3}})};
    const auto repeats = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 1});
    auto Node = std::make_shared<ov::op::v0::Tile>(params.at(0), repeats);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "TileGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::TopK> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3, 2}})};
    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{3});
    auto Node = std::make_shared<ov::op::v1::TopK>(params.at(0),
                                                   k,
                                                   1,
                                                   ov::op::v1::TopK::Mode::MAX,
                                                   ov::op::v1::TopK::SortType::SORT_VALUES);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node->output(0)),
                             std::make_shared<ov::op::v0::Result>(Node->output(1))};
    return std::make_shared<ov::Model>(results, params, "TopKGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::TopK> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3, 2}})};
    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{3});
    auto Node = std::make_shared<ov::op::v3::TopK>(params.at(0),
                                                   k,
                                                   1,
                                                   ov::op::v3::TopK::Mode::MAX,
                                                   ov::op::v3::TopK::SortType::SORT_VALUES);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node->output(0)),
                             std::make_shared<ov::op::v0::Result>(Node->output(1))};
    return std::make_shared<ov::Model>(results, params, "TopKGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v11::TopK> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3, 2}})};
    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{3});
    auto Node = std::make_shared<ov::op::v11::TopK>(params.at(0),
                                                   k,
                                                   -2,
                                                   ov::op::v11::TopK::Mode::MIN,
                                                   ov::op::v11::TopK::SortType::SORT_VALUES,
                                                   ov::element::i64,
                                                   true);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node->output(0)),
                             std::make_shared<ov::op::v0::Result>(Node->output(1))};
    return std::make_shared<ov::Model>(results, params, "TopKGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::Transpose> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 3}})};
    const auto inputOrder = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
    auto Node = std::make_shared<ov::op::v1::Transpose>(params.at(0), inputOrder);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "TransposeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Unsqueeze> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{4, 2}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, -1});
    auto Node = std::make_shared<ov::op::v0::Unsqueeze>(params.at(0), axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "UnsqueezeGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::VariadicSplit> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 8, 2, 2}})};
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    const auto splitLengths = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 3, 2, 2});
    auto Node = std::make_shared<ov::op::v1::VariadicSplit>(params.at(0), axis, splitLengths);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node->output(0)),
                             std::make_shared<ov::op::v0::Result>(Node->output(1)),
                             std::make_shared<ov::op::v0::Result>(Node->output(2)),
                             std::make_shared<ov::op::v0::Result>(Node->output(3))};
    return std::make_shared<ov::Model>(results, params, "VariadicSplitGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v9::GridSample> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 4, 4});
    const auto grid = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 6, 6, 2});
    const auto attributes = ov::op::v9::GridSample::Attributes{};
    const auto grid_sample = std::make_shared<ov::op::v9::GridSample>(data, grid, attributes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(grid_sample->output(0))};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, grid}, "GridSampleGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v10::Unique>& node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 10, 10});
    const auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {-1});
    const auto unique = std::make_shared<ov::op::v10::Unique>(data, axis);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(unique)},
                                       ov::ParameterVector{data}, "UniqueGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v10::IsFinite> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    auto is_finite = std::make_shared<ov::op::v10::IsFinite>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(is_finite)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "is_finite_graph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v10::IsInf> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    auto is_finite = std::make_shared<ov::op::v10::IsInf>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(is_finite)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "is_inf_graph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v10::IsNaN> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    auto is_finite = std::make_shared<ov::op::v10::IsNaN>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(is_finite)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "is_nan_graph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v13::BitwiseNot> &node) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 2});
    auto bitwise = std::make_shared<ov::op::v13::BitwiseNot>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(bitwise)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "BitwiseNotGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v13::FakeConvert>& node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
    const auto shift = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
    const auto op = std::make_shared<ov::op::v13::FakeConvert>(data, scale, shift, "f8e4m3");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(op)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, scale, shift}, "FakeConvert");
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
    } else if (ov::is_type<ov::op::v14::MaxPool>(node)) {
        maxPoolNode = std::make_shared<ov::op::v14::MaxPool>(data, strides, dilations, pads_begin, pads_end, kernel_shape);
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
    } else if (ov::is_type<ov::op::v15::ScatterNDUpdate>(node)) {
        scatterNode = std::make_shared<ov::op::v15::ScatterNDUpdate>(data, indices, updates, ov::op::v15::ScatterNDUpdate::Reduction::SUM);
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
    } else if (ov::is_type<ov::op::v0::Clamp>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Clamp>(param, 0.0, 2.1);
    } else if (ov::is_type<ov::op::v0::Cos>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Cos>(param);
    } else if (ov::is_type<ov::op::v0::Cosh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Cosh>(param);
    } else if (ov::is_type<ov::op::v0::Elu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Elu>(param, 0.5f);
    } else if (ov::is_type<ov::op::v0::Erf>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Erf>(param);
    } else if (ov::is_type<ov::op::v0::Exp>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Exp>(param);
    } else if (ov::is_type<ov::op::v0::Floor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Floor>(param);
    } else if (ov::is_type<ov::op::v0::Gelu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Gelu>(param);
    } else if (ov::is_type<ov::op::v7::Gelu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v7::Gelu>(param);
    } else if (ov::is_type<ov::op::v0::GRN>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::GRN>(param, 1e-6);
    } else if (ov::is_type<ov::op::v5::HSigmoid>(node)) {
        eltwiseNode = std::make_shared<ov::op::v5::HSigmoid>(param);
    } else if (ov::is_type<ov::op::v4::HSwish>(node)) {
        eltwiseNode = std::make_shared<ov::op::v4::HSwish>(param);
    } else if (ov::is_type<ov::op::v0::Log>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Log>(param);
    } else if (ov::is_type<ov::op::v4::Mish>(node)) {
        eltwiseNode = std::make_shared<ov::op::v4::Mish>(param);
    } else if (ov::is_type<ov::op::v0::Negative>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Negative>(param);
    } else if (ov::is_type<ov::op::v0::Relu>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Relu>(param);
    } else if (ov::is_type<ov::op::v5::Round>(node)) {
        eltwiseNode = std::make_shared<ov::op::v5::Round>(param, op::v5::Round::RoundMode::HALF_TO_EVEN);
    } else if (ov::is_type<ov::op::v0::Sigmoid>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sigmoid>(param);
    } else if (ov::is_type<ov::op::v0::Sign>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sign>(param);
    } else if (ov::is_type<ov::op::v0::Sin>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sin>(param);
    } else if (ov::is_type<ov::op::v0::Sinh>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Sinh>(param);
    } else if (ov::is_type<ov::op::v9::SoftSign>(node)) {
        eltwiseNode = std::make_shared<ov::op::v9::SoftSign>(param);
    } else if (ov::is_type<ov::op::v4::SoftPlus>(node)) {
        eltwiseNode = std::make_shared<ov::op::v4::SoftPlus>(param);
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

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2}})};
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
    } else if (ov::is_type<ov::op::v1::Multiply>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Multiply>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Power>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Power>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Subtract>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Subtract>(params.front(), params.back());
    } else if (ov::is_type<ov::op::v1::Mod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Mod>(params.front(), params.back());
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseNode)};
    return std::make_shared<ov::Model>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseBitwise(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 2}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{1, 2})};

    std::shared_ptr<ov::Node> eltwise;
    if (ov::is_type<ov::op::v13::BitwiseAnd>(node)) {
        eltwise = std::make_shared<ov::op::v13::BitwiseAnd>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v13::BitwiseOr>(node)) {
        eltwise = std::make_shared<ov::op::v13::BitwiseOr>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v13::BitwiseXor>(node)) {
        eltwise = std::make_shared<ov::op::v13::BitwiseXor>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v15::BitwiseLeftShift>(node)) {
        eltwise = std::make_shared<ov::op::v15::BitwiseLeftShift>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v15::BitwiseRightShift>(node)) {
        eltwise = std::make_shared<ov::op::v15::BitwiseRightShift>(params[0], params[1]);
    } else {
        return nullptr;
    }
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
    return std::make_shared<ov::Model>(results, params, "BinaryEltwiseBitwiseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseComp(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2})};
    std::shared_ptr<ov::Node> eltwise;
    if (ov::is_type<ov::op::v1::Equal>(node)) {
        eltwise = std::make_shared<ov::op::v1::Equal>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::Greater>(node)) {
        eltwise = std::make_shared<ov::op::v1::Greater>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::GreaterEqual>(node)) {
        eltwise = std::make_shared<ov::op::v1::GreaterEqual>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::Less>(node)) {
        eltwise = std::make_shared<ov::op::v1::Less>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LessEqual>(node)) {
        eltwise = std::make_shared<ov::op::v1::LessEqual>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::NotEqual>(node)) {
        eltwise = std::make_shared<ov::op::v1::NotEqual>(params[0], params[1]);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
    return std::make_shared<ov::Model>(results, params, "BinaryEltwiseComparisonGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseLogical(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1})};

    std::shared_ptr<ov::Node> eltwise;
    if (ov::is_type<ov::op::v1::LogicalAnd>(node)) {
        eltwise = std::make_shared<ov::op::v1::LogicalAnd>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalOr>(node)) {
        eltwise = std::make_shared<ov::op::v1::LogicalOr>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalXor>(node)) {
        eltwise = std::make_shared<ov::op::v1::LogicalXor>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v0::Xor>(node)) {
        eltwise = std::make_shared<ov::op::v0::Xor>(params[0], params[1]);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "BinaryEltwiseLogicalGraph");
}

std::shared_ptr<ov::Model> generateBroadcast(const std::shared_ptr<ov::op::Op> &node) {
    const ov::Shape input_shape{};
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{input_shape})};
    const auto shape_const = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{4}, std::vector<uint64_t>{5, 4, 3, 2});
    std::shared_ptr<ov::Node> broadcast;
    if (ov::is_type<ov::op::v1::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v1::Broadcast>(params[0], shape_const);
    } else if (ov::is_type<ov::op::v3::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params[0], shape_const);
    } else {
        return nullptr;
    }

    return std::make_shared<ov::Model>(broadcast, ParameterVector{params}, "BroadcastGraph");
}

std::shared_ptr<ov::Model> generateConvertColor(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, Shape{1, 3, 2, 1});
    std::shared_ptr<ov::Node> convert;
    if (ov::is_type<ov::op::v8::NV12toBGR>(node)) {
        convert = std::make_shared<ov::op::v8::NV12toBGR>(params);
    } else if (ov::is_type<ov::op::v8::NV12toRGB>(node)) {
        convert = std::make_shared<ov::op::v8::NV12toRGB>(params);
    } else if (ov::is_type<ov::op::v8::I420toBGR>(node)) {
        convert = std::make_shared<ov::op::v8::I420toBGR>(params);
    } else if (ov::is_type<ov::op::v8::I420toRGB>(node)) {
        convert = std::make_shared<ov::op::v8::I420toRGB>(params);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convert)};
    return std::make_shared<ov::Model>(results, ParameterVector{params}, "ConvertColorGraph");
}

std::shared_ptr<ov::Model> generateMultiSubGraph(const std::shared_ptr<ov::op::Op> &node) {
    if (ov::is_type<ov::op::v8::If>(node)) {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, Shape{1});
        auto A = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 8.0);
        auto B = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 2.0);
        auto A_res = std::make_shared<ov::op::v0::Result>(A);
        auto B_res = std::make_shared<ov::op::v0::Result>(B);
        auto then_body = std::make_shared<ov::Model>(OutputVector{A_res}, ParameterVector{});
        auto else_body = std::make_shared<ov::Model>(OutputVector{B_res}, ParameterVector{});
        auto if_op = std::make_shared<ov::op::v8::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto res = if_op->set_output(A_res, B_res);
        return std::make_shared<ov::Model>(OutputVector{res}, ParameterVector{cond}, "MultiSubGraphOp");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v8::MatrixNms> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 2}})};
    const auto nms =
        std::make_shared<ov::op::v8::MatrixNms>(params[0], params[1], ov::op::v8::MatrixNms::Attributes());
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "MatrixNms");
}

std::shared_ptr<ov::Model> generateMulticlassNmsBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 2}})};
    if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        const auto nms = std::make_shared<ov::op::v8::MulticlassNms>(params[0], params[1], ov::op::v8::MulticlassNms::Attributes());
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
        return std::make_shared<ov::Model>(results, params, "MulticlassNms");
    } else if (ov::is_type<ov::op::v9::MulticlassNms>(node)) {
        const auto nms = std::make_shared<ov::op::v9::MulticlassNms>(params[0], params[1], ov::op::v9::MulticlassNms::Attributes());
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
        return std::make_shared<ov::Model>(results, params, "MulticlassNms");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateReadValueBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1})};
    if (ov::is_type<ov::op::v3::ReadValue>(node)) {
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(params[0], "v0");
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
        auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
        return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "ReadValue-3");
    } else if (ov::is_type<ov::op::v6::ReadValue>(node)) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(params[0], variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
        return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "ReadValue-6");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateDeformableConvolutionBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 4, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 18, 2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 3, 3}})};
    std::shared_ptr<ov::Node> deformableConvolutionNode;
    if (ov::is_type<ov::op::v1::DeformableConvolution>(node)) {
        deformableConvolutionNode = std::make_shared<ov::op::v1::DeformableConvolution>(params.at(0), params.at(1), params.at(2),
                                                                                        ov::Strides {1, 1},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::Strides {1, 1});
    } else if (ov::is_type<ov::op::v8::DeformableConvolution>(node)) {
        deformableConvolutionNode = std::make_shared<ov::op::v8::DeformableConvolution>(params.at(0), params.at(1), params.at(2),
                                                                                        ov::Strides {1, 1},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::CoordinateDiff {0, 0},
                                                                                        ov::Strides {1, 1});
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(deformableConvolutionNode)};
    return std::make_shared<ov::Model>(results, params, "DeformableConvolutionBaseGraph");
}

std::shared_ptr<ov::Model> generateDetectionOutputBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 8}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 6}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 1, 8}})};
    ov::op::v0::DetectionOutput::Attributes attrs;
    ov::op::v8::DetectionOutput::Attributes attrs_v8;
    attrs.num_classes = 3;
    attrs_v8.background_label_id = attrs.background_label_id = -1;
    attrs_v8.top_k = attrs.top_k = -1;
    attrs_v8.variance_encoded_in_target = attrs.variance_encoded_in_target = true;
    attrs_v8.keep_top_k = attrs.keep_top_k = {2};
    attrs_v8.code_type = attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs_v8.share_location = attrs.share_location = true;
    attrs_v8.nms_threshold = attrs.nms_threshold = 0.5;
    attrs_v8.confidence_threshold = attrs.confidence_threshold = 0.3;
    attrs_v8.clip_after_nms = attrs.clip_after_nms = false;
    attrs_v8.clip_before_nms = attrs.clip_before_nms = true;
    attrs_v8.decrease_label_id = attrs.decrease_label_id = false;
    attrs_v8.normalized = attrs.normalized = true;
    attrs_v8.input_height = attrs.input_height = 0;
    attrs_v8.input_width = attrs.input_width = 0;
    attrs_v8.objectness_score = attrs.objectness_score = 0;

    std::shared_ptr<ov::Node> DetectionOutputNode;
    if (ov::is_type<ov::op::v0::DetectionOutput>(node)) {
        DetectionOutputNode = std::make_shared<ov::op::v0::DetectionOutput>(params.at(0), params.at(1), params.at(2), attrs);
    } else if (ov::is_type<ov::op::v8::DetectionOutput>(node)) {
        DetectionOutputNode = std::make_shared<ov::op::v8::DetectionOutput>(params.at(0), params.at(1), params.at(2), attrs_v8);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(DetectionOutputNode)};
    return std::make_shared<ov::Model>(results, params, "DetectionOutputBaseGraph");
}

std::shared_ptr<ov::Model> generateEmbeddingBagOffsetsBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 2}})};
    const auto indices = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::i32, ov::Shape{4}));
    const auto offsets = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::i32, ov::Shape{3}));
    const auto default_index = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape(), std::vector<int32_t>{0});

    std::shared_ptr<ov::Node> EmbeddingBagOffsetsNode;
    if (ov::is_type<ov::op::v3::EmbeddingBagOffsetsSum>(node)) {
        EmbeddingBagOffsetsNode = std::make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(params.at(0), indices, offsets, default_index);
    } else if (ov::is_type<ov::op::v15::EmbeddingBagOffsets>(node)) {
        const auto reduction = ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN;
        EmbeddingBagOffsetsNode = std::make_shared<ov::op::v15::EmbeddingBagOffsets>(params.at(0), indices, offsets, default_index, reduction);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(EmbeddingBagOffsetsNode)};
    return std::make_shared<ov::Model>(results, params, "EmbeddingBagOffsetsBaseGraph");
}

std::shared_ptr<ov::Model> generateEmbeddingBagPackedBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 2}})};
    const auto indices = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::i32, ov::Shape{2, 3}));

    std::shared_ptr<ov::Node> EmbeddingBagPackedNode;
    if (ov::is_type<ov::op::v3::EmbeddingBagPackedSum>(node)) {
        EmbeddingBagPackedNode = std::make_shared<ov::op::v3::EmbeddingBagPackedSum>(params.at(0), indices);
    } else if (ov::is_type<ov::op::v15::EmbeddingBagPacked>(node)) {
        const auto reduction = ov::op::v15::EmbeddingBagPacked::Reduction::SUM;
        EmbeddingBagPackedNode = std::make_shared<ov::op::v15::EmbeddingBagPacked>(params.at(0), indices, reduction);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(EmbeddingBagPackedNode)};
    return std::make_shared<ov::Model>(results, params, "EmbeddingBagPackedBaseGraph");
}

std::shared_ptr<ov::Model> generateFFTBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 10, 10, 2}})};
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{2});

    std::shared_ptr<ov::Node> FFTBaseNode;
    if (ov::is_type<ov::op::v7::DFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v7::DFT>(params.at(0), axes);
    } else if (ov::is_type<ov::op::v7::IDFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v7::IDFT>(params.at(0), axes);
    } else if (ov::is_type<ov::op::v9::RDFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v9::RDFT>(params.at(0), axes);
    } else if (ov::is_type<ov::op::v9::IRDFT>(node)) {
        FFTBaseNode = std::make_shared<ov::op::v9::IRDFT>(params.at(0), axes);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(FFTBaseNode)};
    return std::make_shared<ov::Model>(results, params, "FFTBaseGraph");
}

std::shared_ptr<ov::Model> generateGatherBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{{2, 2, 3, 3}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2})};
    const auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(), std::vector<int64_t>{2});

    std::shared_ptr<ov::Node> GatherBaseNode;
    if (ov::is_type<ov::op::v1::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v1::Gather>(params.at(0), params.at(1), axis);
    } else if (ov::is_type<ov::op::v7::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v7::Gather>(params.at(0), params.at(1), axis);
    } else if (ov::is_type<ov::op::v8::Gather>(node)) {
        GatherBaseNode = std::make_shared<ov::op::v8::Gather>(params.at(0), params.at(1), axis);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(GatherBaseNode)};
    return std::make_shared<ov::Model>(results, params, "GatherBaseGraph");
}

std::shared_ptr<ov::Model> generateGatherNDBase(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{{2, 3, 4, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{{2, 3, 3, 2}})};

    std::shared_ptr<ov::Node> GatherNDBaseNode;
    if (ov::is_type<ov::op::v5::GatherND>(node)) {
        GatherNDBaseNode = std::make_shared<ov::op::v5::GatherND>(params.at(0), params.at(1));
    } else if (ov::is_type<ov::op::v8::GatherND>(node)) {
        GatherNDBaseNode = std::make_shared<ov::op::v8::GatherND>(params.at(0), params.at(1));
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(GatherNDBaseNode)};
    return std::make_shared<ov::Model>(results, params, "GatherNDBaseGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v9::GenerateProposals> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2, 3, 4}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 12, 2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 3, 2, 2}})};
    ov::op::v9::GenerateProposals::Attributes attrs;
    attrs.min_size = 1;
    attrs.nms_threshold = 0.8;
    attrs.pre_nms_count = 100;
    attrs.post_nms_count = 100;
    if (ov::is_type<ov::op::v9::GenerateProposals>(node)) {
        const auto gp = std::make_shared<ov::op::v9::GenerateProposals>(
                params[0], params[1], params[2], params[3], attrs);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gp)};
        return std::make_shared<ov::Model>(results, params, "GenerateProposalsGraph");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::Col2Im> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{12, 9});
    const auto output_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {2}, {4, 4});
    const auto kernel_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {2}, {2, 2});
    const auto Col2ImNode = std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Col2ImNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "Col2ImGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::STFT>& node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 48});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {16});
    const auto step_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {4});
    constexpr bool transpose_frames = true;
    const auto stft = std::make_shared<ov::op::v15::STFT>(data, window, frame_size, step_size, transpose_frames);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(stft)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, window}, "STFTGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v16::ISTFT>& node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{9, 2, 2});
    const auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{16});
    const auto frame_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {16});
    const auto step_size = ov::op::v0::Constant::create<int32_t>(ov::element::i32, {}, {4});
    constexpr bool center = true;
    constexpr bool normalized = true;
    const auto stft = std::make_shared<ov::op::v16::ISTFT>(data, window, frame_size, step_size, center, normalized);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(stft)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data, window}, "ISTFTGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::StringTensorUnpack> &node) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::string, ov::PartialShape{2});
    const auto StringTensorUnpackNode = std::make_shared<ov::op::v15::StringTensorUnpack>(data);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(StringTensorUnpackNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{data}, "StringTensorUnpackGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::StringTensorPack> &node) {
    const auto begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    const auto ends = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{2});
    const auto symbols = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{5});
    const auto StringTensorPackNode = std::make_shared<ov::op::v15::StringTensorPack>(begins, ends, symbols);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(StringTensorPackNode)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{begins, ends, symbols}, "StringTensorPackGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::SliceScatter> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 4, 3, 5}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{1, 2, 2, 5}})};
    const auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 4});
    const auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 4, -5});
    const auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{3, 2, -2});
    const auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2});
    auto Node = std::make_shared<ov::op::v15::SliceScatter>(params.at(0), params.at(1), start, stop, step, axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(Node)};
    return std::make_shared<ov::Model>(results, params, "SliceScatterGraph");
}

std::shared_ptr<ov::Model> generateRNNCellBase(const std::shared_ptr<ov::op::Op> &node) {
    std::shared_ptr<ov::Node> RNNCellBaseNode;
    if (ov::is_type<ov::op::v3::GRUCell>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{9, 3}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{9, 3}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{9}));
        RNNCellBaseNode = std::make_shared<ov::op::v3::GRUCell>(params.at(0), params.at(1),
                                                                W, R, B, 3);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode)};
        return std::make_shared<ov::Model>(results, params, "GRUCell3BaseGraph");
    } else if (ov::is_type<ov::op::v0::LSTMCell>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12, 3}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12, 3}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12}));
        const auto P = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{9}));
        RNNCellBaseNode = std::make_shared<ov::op::v0::LSTMCell>(params.at(0), params.at(1), params.at(2),
                                                                 W, R, B, P, 3);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(1))};
        return std::make_shared<ov::Model>(results, params, "LSTMCell1BaseGraph");
    } else if (ov::is_type<ov::op::v4::LSTMCell>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12, 3}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12, 3}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{12}));
        RNNCellBaseNode = std::make_shared<ov::op::v4::LSTMCell>(params.at(0), params.at(1), params.at(2),
                                                                 W, R, B, 3);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(1))};;
        return std::make_shared<ov::Model>(results, params, "LSTMCell4BaseGraph");
    } else if (ov::is_type<ov::op::v5::GRUSequence>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 10, 10}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 1, 10}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{5})};

        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30, 10}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30, 10}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 30}));
        const size_t hidden_size = 10;
        const auto gru_sequence =
            std::make_shared<ov::op::v5::GRUSequence>(params[0],
                                                       params[1],
                                                       params[2],
                                                       W,
                                                       R,
                                                       B,
                                                       hidden_size,
                                                       ov::op::RecurrentSequenceDirection::FORWARD);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gru_sequence)};
        return std::make_shared<ov::Model>(results, params, "GRUSequence");
    } else if (ov::is_type<ov::op::v5::LSTMSequence>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 10, 10}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 1, 10}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{5, 1, 10}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{5})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 40, 10}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 40, 10}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 40}));
        RNNCellBaseNode = std::make_shared<ov::op::v5::LSTMSequence>(params.at(0), params.at(1), params.at(2), params.at(3),
                                                                     W, R, B, 10, ov::op::RecurrentSequenceDirection::FORWARD);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(1)),
                                 std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(2))};
        return std::make_shared<ov::Model>(results, params, "LSTMSeq5BaseGraph");
    } else if (ov::is_type<ov::op::v0::RNNCell>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 3}})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{3, 3}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{3, 3}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{3}));
        RNNCellBaseNode = std::make_shared<ov::op::v0::RNNCell>(params.at(0), params.at(1),
                                                                W, R, B, 3);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode)};
        return std::make_shared<ov::Model>(results, params, "RNNCellBaseGraph");
    } else if (ov::is_type<ov::op::v5::RNNSequence>(node)) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 5, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 1, 3}}),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2})};
        const auto W = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 3, 3}));
        const auto R = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 3, 3}));
        const auto B = std::make_shared<ov::op::v0::Constant>(utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, 3}));
        RNNCellBaseNode = std::make_shared<ov::op::v5::RNNSequence>(params.at(0), params.at(1), params.at(2),
                                                                    W, R, B, 3, ov::op::RecurrentSequenceDirection::FORWARD);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(0)),
                                 std::make_shared<ov::op::v0::Result>(RNNCellBaseNode->output(1))};
        return std::make_shared<ov::Model>(results, params, "RNNSeqBaseGraph");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v15::SearchSorted>& node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{16})};
    const auto values =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2, 3}, std::vector<float>(6, 0));
    auto new_node = std::make_shared<ov::op::v15::SearchSorted>(params.at(0), values);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(new_node)};
    return std::make_shared<ov::Model>(results, params, "SearchSortedGraph");
}

std::shared_ptr<ov::Model> generateSubGraphOp(const std::shared_ptr<ov::op::Op> &node) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                               std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};

    ov::ParameterVector params_body{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}}),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{{2, 2}})};

    const auto body_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});
    const auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3});
    const auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, std::vector<bool>{true});
    // Body
    auto sum = std::make_shared<ov::op::v1::Add>(params_body.at(0), params_body.at(1));
    auto Zo = std::make_shared<ov::op::v1::Multiply>(sum, params_body.at(2));
    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo}, params_body);

    ov::Output<ov::Node> SubGraphOpNode;
    if (ov::is_type<ov::op::v0::TensorIterator>(node)) {
        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(params_body.at(0), params.at(0), 0, 1, 1, -1, 1);
        tensor_iterator->set_sliced_input(params_body.at(1), params.at(1), 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(params_body.at(2), params.at(2), Zo);

        // Output 0 is last Zo
        SubGraphOpNode = tensor_iterator->get_iter_value(Zo, -1);
    } else if (ov::is_type<ov::op::v5::Loop>(node)) {
        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
        loop->set_function(body);

        loop->set_invariant_input(params_body.at(0), params.at(0));
        loop->set_invariant_input(params_body.at(1), params.at(1));
        loop->set_merged_input(params_body.at(2), params.at(2), Zo);

        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});
        SubGraphOpNode = loop->get_iter_value(Zo, -1);
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(SubGraphOpNode)};
    return std::make_shared<ov::Model>(results, params, "SubGraphOpGraph");
}
}  // namespace

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
    } else if (ov::is_type<ov::op::util::BinaryElementwiseBitwise>(node)) {
        return generateBinaryEltwiseBitwise(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseComparison>(node)) {
        return generateBinaryEltwiseComp(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseLogical>(node)) {
        return generateBinaryEltwiseLogical(node);
    } else if (ov::is_type<ov::op::util::BroadcastBase>(node)) {
        return generateBroadcast(node);
    } else if (ov::is_type<ov::op::util::ConvertColorNV12Base>(node) ||
               ov::is_type<ov::op::util::ConvertColorI420Base>(node)) {
        return generateConvertColor(node);
    } else if (ov::is_type<ov::op::util::MulticlassNmsBase>(node)) {
        return generateMulticlassNmsBase(node);
    } else if (ov::is_type<ov::op::util::ReadValueBase>(node)) {
        return generateReadValueBase(node);
    } else if (ov::is_type<ov::op::util::DeformableConvolutionBase>(node)) {
        return generateDeformableConvolutionBase(node);
    } else if (ov::is_type<ov::op::util::DetectionOutputBase>(node)) {
        return generateDetectionOutputBase(node);
    } else if (ov::is_type<ov::op::util::EmbeddingBagOffsetsBase>(node)) {
        return generateEmbeddingBagOffsetsBase(node);
    } else if (ov::is_type<ov::op::util::EmbeddingBagPackedBase>(node)) {
        return generateEmbeddingBagPackedBase(node);
    } else if (ov::is_type<ov::op::util::FFTBase>(node)) {
        return generateFFTBase(node);
    } else if (ov::is_type<ov::op::util::GatherBase>(node)) {
        return generateGatherBase(node);
    } else if (ov::is_type<ov::op::util::GatherNDBase>(node)) {
        return generateGatherNDBase(node);
    } else if (ov::is_type<ov::op::util::RNNCellBase>(node)) {
        return generateRNNCellBase(node);
    } else if (ov::is_type<ov::op::util::SubGraphOp>(node)) {
        return generateSubGraphOp(node);
    } else if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        return generateMultiSubGraph(node);
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
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
#include "openvino/opsets/opset13_tbl.hpp"
#include "openvino/opsets/opset14_tbl.hpp"
#include "openvino/opsets/opset15_tbl.hpp"
#include "openvino/opsets/opset16_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return opGeneratorMap;
}

}  // namespace op_conformance
}  // namespace test
}  // namespace ov
