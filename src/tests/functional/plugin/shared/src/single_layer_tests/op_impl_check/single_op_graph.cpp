// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <single_layer_tests/op_impl_check/op_impl_check.hpp>
#include <single_layer_tests/op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::Op>& node) {
    return nullptr;
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Elu>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 2}});
    const auto elu = std::make_shared<ov::op::v0::Elu>(params[0], 0.5f);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(elu)};
    return std::make_shared<ngraph::Function>(results, params, "ElueGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::EmbeddingSegmentsSum>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{5, 2}});
    const auto indices = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {4}, {0, 2, 3, 4});
    const auto segment_ids = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {4}, {0, 0, 2, 2});
    const auto num_segments = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {}, {3});
    const auto default_index = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {}, {0});
    const auto per_sample_weights =
        ngraph::builder::makeConstant<float>(ov::element::f32, {4}, {0.5, 0.5, 0.5, 0.5});
    const auto embed_seg_sum = std::make_shared<ov::op::v3::EmbeddingSegmentsSum>(params[0],
                                                                                    indices,
                                                                                    segment_ids,
                                                                                    num_segments,
                                                                                    default_index,
                                                                                    per_sample_weights);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embed_seg_sum)};
    return std::make_shared<ngraph::Function>(results, params, "EmbeddingSegmentsSum");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput>& node) {
    const auto rois = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{16, 4}},
        {1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         4.0f, 1.0f, 8.0f,  5.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto deltas = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{16, 8}},
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto scores = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{16, 2}},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto im_info = ngraph::builder::makeConstant<float>(ov::element::f32, {{1, 3}}, {1.0f, 1.0f, 1.0f});
    const auto attrs = ov::op::v6::ExperimentalDetectronDetectionOutput::Attributes{0.01000000074505806f,
                                                                                    0.2f,
                                                                                    2.0f,
                                                                                    2,
                                                                                    500,
                                                                                    5,
                                                                                    true,
                                                                                    {10.0f, 10.0f, 5.0f, 5.0f}};
    const auto exp_detection_output =
        std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(rois, deltas, scores, im_info, attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(exp_detection_output)};
    return std::make_shared<ngraph::Function>(results,
                                                ngraph::ParameterVector{},
                                                "ExperimentalDetectronDetectionOutput");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>& node) {
    const auto im_info = ngraph::builder::makeConstant<float>(ov::element::f32, {{3}}, {1.0f, 1.0f, 1.0f});
    const auto anchors = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{36, 4}},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto deltas = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{12, 2, 6}},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    const auto scores = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {{3, 2, 6}},
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f});
    const auto attrs =
        ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes{0, 0.699999988079071, 6, 1000};
    const auto exp_gen_prop_sing_img =
        std::make_shared<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(im_info,
                                                                                        anchors,
                                                                                        deltas,
                                                                                        scores,
                                                                                        attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(exp_gen_prop_sing_img)};
    return std::make_shared<ngraph::Function>(results,
                                                ngraph::ParameterVector{},
                                                "ExperimentalDetectronGenerateProposalsSingleImage");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator>& node) {
    const auto params =
        ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}});
    const auto attrs = ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes{true, 0, 0, 4.0f, 4.0f};
    const auto exp_prior_grid_gen = std::make_shared<ov::op::v6::ExperimentalDetectronPriorGridGenerator>(params[0],
                                                                                                            params[1],
                                                                                                            params[2],
                                                                                                            attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(exp_prior_grid_gen)};
    return std::make_shared<ngraph::Function>(results, params, "ExperimentalDetectronPriorGridGenerator");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 4}, {1, 2, 2, 3}});
    const auto attrs = ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes{3, 2, {4}, false};
    const auto exp_roi_feature_ext =
        std::make_shared<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(NodeVector{params[0], params[1]},
                                                                                attrs);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(exp_roi_feature_ext)};
    return std::make_shared<ngraph::Function>(results, params, "ExperimentalDetectronROIFeatureExtractor");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronTopKROIs>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 4}, {2}});
    const auto exp_topk_rois = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(params[0], params[1], 1);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(exp_topk_rois)};
    return std::make_shared<ngraph::Function>(results, params, "ExperimentalDetectronTopKROIs");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ExtractImagePatches>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 10, 10}});
    const auto ext_img_patch = std::make_shared<ov::op::v3::ExtractImagePatches>(params[0],
                                                                                 ov::Shape{3, 3},
                                                                                 ov::Strides{5, 5},
                                                                                 ov::Shape{1, 1},
                                                                                 ov::op::PadType::VALID);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ext_img_patch)};
    return std::make_shared<ngraph::Function>(results, params, "ExtractImagePatches");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::FakeQuantize>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2, 3, 4}});
    const auto input_low = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {0.f});
    const auto input_high = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {23.f});
    const auto output_low = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {2.f});
    const auto output_high = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {16.f});
    const auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(params[0], input_low, input_high, output_low, output_high, 4);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(fake_quantize)};
    return std::make_shared<ngraph::Function>(results, params, "FakeQuantize");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::GRN>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 4}});
    const auto grn = std::make_shared<ov::op::v0::GRN>(params[0], 1e-6);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(grn)};
    return std::make_shared<ngraph::Function>(results, params, "GRN");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::GRUSequence>& node) {
    const size_t b = 5;   // batch size
    const size_t i = 10;  // input size
    const size_t h = 10;  // hidden size
    const size_t s = 10;  // sequence length
    const size_t n = 1;   // num of direction
    const auto params =
        ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::f32, ov::element::i64, ov::element::f32, ov::element::f32, ov::element::f32},
                                            {{b, s, i}, {b, n, h}, {b}, {n, h * 3, i}, {n, h * 3, h}, {n, h * 3}});
    const auto gru_sequence = std::make_shared<ov::op::v5::GRUSequence>(params[0],
                                                                        params[1],
                                                                        params[2],
                                                                        params[3],
                                                                        params[4],
                                                                        params[5],
                                                                        h,
                                                                        ov::op::RecurrentSequenceDirection::FORWARD,
                                                                        std::vector<std::string>{"sigmoid", "tanh"},
                                                                        std::vector<float>{},
                                                                        std::vector<float>{},
                                                                        0.7f,
                                                                        false);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_sequence)};
    return std::make_shared<ngraph::Function>(results, params, "GRUSequence");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::GatherElements>& node) {
    const auto params = ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::i32}, {{3}, {7}});
    const auto gather_elements = std::make_shared<ov::op::v6::GatherElements>(params[0], params[1], 0);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather_elements)};
    return std::make_shared<ngraph::Function>(results, params, "GatherElements");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GatherTree>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 10}, {1, 1, 10}, {1}, {}});
    const auto gather_tree = std::make_shared<ov::op::v1::GatherTree>(params[0], params[1], params[2], params[3]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather_tree)};
    return std::make_shared<ngraph::Function>(results, params, "GatherTree");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Gelu>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{8}});
    const auto gelu = std::make_shared<ov::op::v0::Gelu>(params[0]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gelu)};
    return std::make_shared<ngraph::Function>(results, params, "Gelu");
}

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2}, {1, 2}});
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
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseNode)};
    return std::make_shared<ngraph::Function>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseComp(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2}, {2}});
    std::shared_ptr<ov::Node> comp;
    if (ov::is_type<ov::op::v1::Equal>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::EQUAL);
    } else if (ov::is_type<ov::op::v1::Greater>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::GREATER);
    } else if (ov::is_type<ov::op::v1::GreaterEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::GREATER_EQUAL);
    } else if (ov::is_type<ov::op::v1::Less>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::LESS);
    } else if (ov::is_type<ov::op::v1::LessEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::LESS_EQUAL);
    } else if (ov::is_type<ov::op::v1::NotEqual>(node)) {
        comp = ngraph::builder::makeComparison(params[0], params[1], ngraph::helpers::ComparisonTypes::NOT_EQUAL);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(comp)};
    return std::make_shared<ngraph::Function>(results, params, "BinaryEltwiseComparisonGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseLogical(const std::shared_ptr<ov::op::Op>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::boolean, {{1, 2}, {1, 2}});
    std::shared_ptr<ov::Node> eltwiseNode;
    if (ov::is_type<ov::op::v1::LogicalAnd>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalAnd>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalOr>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalOr>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v1::LogicalXor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::LogicalXor>(params[0], params[1]);
    } else if (ov::is_type<ov::op::v0::Xor>(node)) {
        eltwiseNode = std::make_shared<ov::op::v0::Xor>(params[0], params[1]);
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwiseNode)};
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{params}, "BinaryEltwiseLogicalGraph");
}

std::shared_ptr<ov::Model> generateBroadcast(const std::shared_ptr<ov::op::Op>& node) {
    const ov::Shape input_shape{};
    const ov::Shape output_shape{5, 4, 3, 2};
    const auto params = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    const auto shape_const =
        ov::op::v0::Constant::create(ov::element::u64, ov::Shape{output_shape.size()}, output_shape);
    std::shared_ptr<ov::Node> broadcast;
    if (ov::is_type<ov::op::v1::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v1::Broadcast>(params, shape_const);
    } else if (ov::is_type<ov::op::v3::Broadcast>(node)) {
        broadcast = std::make_shared<ov::op::v3::Broadcast>(params, shape_const);
    } else {
        return nullptr;
    }

    return std::make_shared<ngraph::Function>(broadcast, ParameterVector{params}, "BroadcastGraph");
}

std::shared_ptr<ov::Model> generateConvertColor(const std::shared_ptr<ov::op::Op>& node) {
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

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convert)};
    return std::make_shared<ngraph::Function>(results, ParameterVector{params}, "ConvertColorGraph");
}

std::shared_ptr<ov::Model> generateMultiSubGraph(const std::shared_ptr<ov::op::Op>& node) {
    if (ov::is_type<ov::op::v8::If>(node)) {
        auto cond = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, Shape{1});
        auto A = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, 8.0);
        auto B = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, 2.0);
        auto A_res = std::make_shared<ov::op::v0::Result>(A);
        auto B_res = std::make_shared<ov::op::v0::Result>(B);
        auto then_body = std::make_shared<ov::Model>(OutputVector{A_res}, ParameterVector{});
        auto else_body = std::make_shared<ov::Model>(OutputVector{B_res}, ParameterVector{});
        auto if_op = std::make_shared<ov::op::v8::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto res = if_op->set_output(A_res, B_res);
        return std::make_shared<ngraph::Function>(OutputVector{res}, ParameterVector{cond}, "MultiSubGraphOp");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateNmsBase(const std::shared_ptr<ov::op::Op>& node) {
    const auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    const auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    std::shared_ptr<ov::Node> nms;
    if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        nms = std::make_shared<ov::op::v8::MulticlassNms>(boxes, scores, ov::op::v8::MulticlassNms::Attributes());
    } else if (ov::is_type<ov::op::v8::MatrixNms>(node)) {
        nms = std::make_shared<ov::op::v8::MatrixNms>(boxes, scores, ov::op::v8::MatrixNms::Attributes());
    } else {
        return nullptr;
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nms->output(0)),
                                 std::make_shared<ngraph::opset1::Result>(nms->output(1)),
                                 std::make_shared<ngraph::opset1::Result>(nms->output(2))};
    return std::make_shared<ngraph::Function>(results, ParameterVector{boxes, scores}, "NmsBase");
}

std::shared_ptr<ov::Model> generateReadValueAssignBase(const std::shared_ptr<ov::op::Op>& node) {
    auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    if (ov::is_type<ov::op::v3::ReadValue>(node) || ov::is_type<ov::op::v3::Assign>(node)) {
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(in, "v0");
        auto assign = std::make_shared<ov::op::v3::Assign>(read_value, "v0");
        return std::make_shared<ov::Model>(OutputVector{assign}, ParameterVector{in}, "ReadValue_Assign");
    } else if (ov::is_type<ov::op::v6::ReadValue>(node) || ov::is_type<ov::op::v6::Assign>(node)) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(in, variable);
        auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
        return std::make_shared<ov::Model>(OutputVector{assign},
                                           ParameterVector{in},
                                           ov::op::util::VariableVector{variable}, "ReadValue_Assign");
    } else {
        return nullptr;
    }
}
}  // namespace

template <typename T>
std::shared_ptr<ov::Model> generateGraph() {
    std::shared_ptr<T> node = std::shared_ptr<T>(new T);
    if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
        return generateBinaryEltwise(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseComparison>(node)) {
        return generateBinaryEltwiseComp(node);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseLogical>(node)) {
        return generateBinaryEltwiseLogical(node);
    } else if (ov::is_type<ov::op::util::BroadcastBase>(node)) {
        return generateBroadcast(node);
    } else if (ov::is_type<ov::op::util::ConvertColorNV12Base>(node) ||
               ov::is_type<ov::op::util::ConvertColorI420Base>(node)) {
        return generateConvertColor(node);
    } else if (ov::is_type<ov::op::util::MultiSubGraphOp>(node)) {
        return generateMultiSubGraph(node);
    } else if (ov::is_type<ov::op::util::NmsBase>(node)) {
        return generateNmsBase(node);
    } else if (ov::is_type<ov::op::util::AssignBase>(node) || ov::is_type<ov::op::util::ReadValueBase>(node)) {
        return generateReadValueAssignBase(node);
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