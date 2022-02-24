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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Elu> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 2}});
    const auto elu = std::make_shared<ov::op::v0::Elu>(params[0], 0.5f);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(elu)};
    return std::make_shared<ov::Model>(results, params, "ElueGraph");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::EmbeddingSegmentsSum> &node) {
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
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(embed_seg_sum)};
    return std::make_shared<ov::Model>(results, params, "EmbeddingSegmentsSum");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronDetectionOutput> &node) {
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
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_detection_output)};
    return std::make_shared<ov::Model>(results,
                                                ngraph::ParameterVector{},
                                                "ExperimentalDetectronDetectionOutput");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage> &node) {
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
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_gen_prop_sing_img)};
    return std::make_shared<ov::Model>(results,
                                                ngraph::ParameterVector{},
                                                "ExperimentalDetectronGenerateProposalsSingleImage");
}

std::shared_ptr<ov::Model> generate(
    const std::shared_ptr<ov::op::v6::ExperimentalDetectronPriorGridGenerator> &node) {
    const auto params =
        ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 4}, {1, 2, 2, 3}});
    const auto attrs = ov::op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes{3, 2, {4}, false};
    const auto exp_roi_feature_ext =
        std::make_shared<ov::op::v6::ExperimentalDetectronROIFeatureExtractor>(NodeVector{params[0], params[1]},
                                                                                attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_roi_feature_ext)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronROIFeatureExtractor");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::ExperimentalDetectronTopKROIs> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 4}, {2}});
    const auto exp_topk_rois = std::make_shared<ov::op::v6::ExperimentalDetectronTopKROIs>(params[0], params[1], 1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(exp_topk_rois)};
    return std::make_shared<ov::Model>(results, params, "ExperimentalDetectronTopKROIs");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::ExtractImagePatches> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 10, 10}});
    const auto ext_img_patch = std::make_shared<ov::op::v3::ExtractImagePatches>(params[0],
                                                                                 ov::Shape{3, 3},
                                                                                 ov::Strides{5, 5},
                                                                                 ov::Shape{1, 1},
                                                                                 ov::op::PadType::VALID);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ext_img_patch)};
    return std::make_shared<ov::Model>(results, params, "ExtractImagePatches");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::FakeQuantize> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2, 3, 4}});
    const auto input_low = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {0.f});
    const auto input_high = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {23.f});
    const auto output_low = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {2.f});
    const auto output_high = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {16.f});
    const auto fake_quantize = std::make_shared<ov::op::v0::FakeQuantize>(params[0], input_low, input_high, output_low, output_high, 4);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fake_quantize)};
    return std::make_shared<ov::Model>(results, params, "FakeQuantize");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::GRN> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 4}});
    const auto grn = std::make_shared<ov::op::v0::GRN>(params[0], 1e-6);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(grn)};
    return std::make_shared<ov::Model>(results, params, "GRN");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::GRUSequence> &node) {
    const auto params =
        ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::f32, ov::element::i64},
                                           {{5, 10, 10}, {5, 1, 10}, {5}});
    const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 30, 10}, {}, true);
    const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 30, 10}, {}, true);
    const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 30}, {}, true);
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
    const auto params = ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::i32}, {{3}, {7}});
    const auto gather_elements = std::make_shared<ov::op::v6::GatherElements>(params[0], params[1], 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_elements)};
    return std::make_shared<ov::Model>(results, params, "GatherElements");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GatherTree> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 10}, {1, 1, 10}, {1}, {}});
    const auto gather_tree = std::make_shared<ov::op::v1::GatherTree>(params[0], params[1], params[2], params[3]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_tree)};
    return std::make_shared<ov::Model>(results, params, "GatherTree");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Gelu> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{8}});
    const auto gelu = std::make_shared<ov::op::v0::Gelu>(params[0]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gelu)};
    return std::make_shared<ov::Model>(results, params, "Gelu");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GroupConvolution> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 6}, {1, 1, 1, 3}});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 4}, {1, 1, 1, 3}});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3}});
    const auto alpha = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {std::vector<float>{0.5}});
    const auto beta = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {std::vector<float>{0.6}});
    const auto hard_sigmoid = std::make_shared<ov::op::v0::HardSigmoid>(params[0], alpha, beta);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(hard_sigmoid)};
    return std::make_shared<ov::Model>(results, params, "HardSigmoid");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Interpolate> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 2, 4}});
    const auto out_shape_in = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {4}, {1, 1, 1, 2});
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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Interpolate> &node) {
    using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;
    using TransformMode = op::v4::Interpolate::CoordinateTransformMode;
    using NearestMode = op::v4::Interpolate::NearestMode;

    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 2, 4}});
    const auto out_shape_in = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {4}, {1, 1, 1, 2});
    const auto scales = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {1.0});
    const InterpolateAttrs attrs{InterpolateMode::NEAREST,
                                 ShapeCalcMode::SIZES,
                                 std::vector<size_t>{0, 0, 0, 0},
                                 std::vector<size_t>{0, 0, 0, 0},
                                 TransformMode::HALF_PIXEL,
                                 NearestMode::ROUND_PREFER_FLOOR,
                                 false,
                                 -0.75};
    const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(params[0], out_shape_in, scales, attrs);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(interpolate)};
    return std::make_shared<ov::Model>(results, params, "Interpolate-4");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::Assign> &node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(params[0], "v0");
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "Assign-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::Assign> &node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(params[0], variable);
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    return std::make_shared<ov::Model>(results, SinkVector{assign}, params, "Assign-6");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LRN> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3, 2, 1}});
    const auto axes = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {1}, std::vector<int64_t>{2});
    const auto lrn = std::make_shared<ov::op::v0::LRN>(params[0], axes, 3, 0.5, 1, 3);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lrn)};
    return std::make_shared<ov::Model>(results, params, "LRN");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LSTMSequence> &node) {
    const auto params =
        ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::f32, ov::element::f32, ov::element::i32},
                                           {{5, 10, 10}, {5, 1, 10}, {5, 1, 10}, {5}});
    const auto W = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
    const auto R = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40, 10}, {}, true);
    const auto B = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 40}, {}, true);
    const auto P = ngraph::builder::makeConstant<float>(ov::element::f32, {1, 30}, {}, true);
    const int64_t hidden_size = 10;
    const auto lstm_sequence =
        std::make_shared<ov::op::v0::LSTMSequence>(params[0],
                                                   params[1],
                                                   params[2],
                                                   params[3],
                                                   W,
                                                   R,
                                                   B,
                                                   P,
                                                   hidden_size,
                                                   ov::op::RecurrentSequenceDirection::FORWARD);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2))};
    return std::make_shared<ov::Model>(results, params, "LSTMSequence");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::LogSoftmax> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    const auto lsm = std::make_shared<ov::op::v5::LogSoftmax>(params[0], 0);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lsm)};
    return std::make_shared<ov::Model>(results, params, "LogSoftmax");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::LogicalNot> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::boolean, {{1, 2}});
    const auto logical_not = std::make_shared<ov::op::v1::LogicalNot>(params[0]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(logical_not)};
    return std::make_shared<ov::Model>(results, params, "LogicalNot");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::MVN> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 3, 3, 3}});
    const auto mvn = std::make_shared<ov::op::v0::MVN>(params[0], false, false, 1e-9);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mvn)};
    return std::make_shared<ov::Model>(results, params, "MVN-2");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::MVN> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 3, 3, 3}});
    const auto axes = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {2}, std::vector<int64_t>{2, 3});
    const auto mvn = std::make_shared<ov::op::v6::MVN>(params[0], axes, false, 1e-9, ov::op::MVNEpsMode::OUTSIDE_SQRT);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mvn)};
    return std::make_shared<ov::Model>(results, params, "MVN-6");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::MatMul> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 2}, {2, 2}});
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], params[1], false, false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(matmul)};
    return std::make_shared<ov::Model>(results, params, "MatMul-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Mish> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 2}});
    const auto mish = std::make_shared<ov::op::v4::Mish>(params[0]);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mish)};
    return std::make_shared<ov::Model>(results, params, "Mish-4");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::NonMaxSuppression> &node) {
    const auto params = ngraph::builder::makeDynamicParams(
        {ov::element::f32, ov::element::f32, ov::element::i32, ov::element::f32, ov::element::f32},
        {{1, 6, 4}, {1, 1, 6}, {}, {}, {}});
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
    const auto params = ngraph::builder::makeDynamicParams(
        {ov::element::f32, ov::element::f32, ov::element::i32, ov::element::f32, ov::element::f32},
        {{1, 6, 4}, {1, 1, 6}, {}, {}, {}});
    auto nms = std::make_shared<ov::op::v3::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v3::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::NonMaxSuppression> &node) {
    const auto params = ngraph::builder::makeDynamicParams(
        {ov::element::f32, ov::element::f32, ov::element::i32, ov::element::f32, ov::element::f32},
        {{1, 6, 4}, {1, 1, 6}, {}, {}, {}});
    auto nms = std::make_shared<ov::op::v4::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v4::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v5::NonMaxSuppression> &node) {
    const auto params = ngraph::builder::makeDynamicParams(
        {ov::element::f32, ov::element::f32, ov::element::i32, ov::element::f32, ov::element::f32},
        {{1, 6, 4}, {1, 1, 6}, {}, {}, {}});
    auto nms = std::make_shared<ov::op::v5::NonMaxSuppression>(params[0],
                                                               params[1],
                                                               params[2],
                                                               params[3],
                                                               params[4],
                                                               ov::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
                                                               false);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
    return std::make_shared<ov::Model>(results, params, "NonMaxSuppression-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::NonZero> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3, 2}});
    auto nonzero = std::make_shared<ov::op::v3::NonZero>(params[0], ov::element::i32);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nonzero)};
    return std::make_shared<ov::Model>(results, params, "NonZero-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::NormalizeL2> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{4}});
    const auto axes = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {0}, std::vector<int64_t>{});
    auto normalize = std::make_shared<ov::op::v0::NormalizeL2>(params[0], axes, 1e-7, ov::op::EpsMode::ADD);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(normalize)};
    return std::make_shared<ov::Model>(results, params, "NormalizeL2-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::OneHot> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::i32, {{}});
    const auto depth = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {}, std::vector<int32_t>{3});
    const auto onvalue = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {}, std::vector<int32_t>{1});
    const auto offvalue = ngraph::builder::makeConstant<int32_t>(ov::element::i32, {}, std::vector<int32_t>{0});
    const int32_t axes = 0;
    const auto onehot = std::make_shared<ov::op::v1::OneHot>(params[0], depth, onvalue, offvalue, axes);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(onehot)};
    return std::make_shared<ov::Model>(results, params, "OneHot-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::PRelu> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{6}});
    const auto slope = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {2});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, n_channel, 20, 20}});
    const auto coordi = ngraph::builder::makeConstant<float>(ov::element::f32,
                                                             {n_boxes, 5},
                                                             {0, 1, 2, 4, 6, 1, 0, 3, 10, 4, 0, 10, 7, 11, 13});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{6}});
    const auto pad_begin = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {1}, {4});
    const auto pad_end = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {1}, {5});
    const auto pad = std::make_shared<ov::op::v1::Pad>(params[0], pad_begin, pad_end, ov::op::PadMode::CONSTANT);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(pad)};
    return std::make_shared<ov::Model>(results, params, "Pad-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Parameter> &node) {
    const auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, Shape{3, 4});
    return std::make_shared<ov::Model>(in, ParameterVector{in}, "Parameter-1");
}

std::shared_ptr<ov::Model> generateBinaryEltwise(const std::shared_ptr<ov::op::Op> &node) {
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
    } else if (ov::is_type<ov::op::v1::Mod>(node)) {
        eltwiseNode = std::make_shared<ov::op::v1::Mod>(params.front(), params.back());
    } else {
        return nullptr;
    }

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwiseNode)};
    return std::make_shared<ov::Model>(results, params, "BinaryEltwiseGraph");
}

std::shared_ptr<ov::Model> generateBinaryEltwiseComp(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2}, {2}});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::boolean, {{1}, {1}});
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
    return std::make_shared<ov::Model>(results, ngraph::ParameterVector{params}, "BinaryEltwiseLogicalGraph");
}

std::shared_ptr<ov::Model> generateBroadcast(const std::shared_ptr<ov::op::Op> &node) {
    const ov::Shape input_shape{};
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {input_shape});
    const auto shape_const =
        ngraph::builder::makeConstant<uint64_t>(ov::element::u64, {4}, {5, 4, 3, 2});
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
        auto B = std::make_shared<ov::op::v0::Constant>(element::f32, ov::Shape{1}, 2.0);
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

std::shared_ptr<ov::Model> generateNmsBase(const std::shared_ptr<ov::op::Op> &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2, 4}, {1, 2, 2}});
    const auto outputs =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
    if (ov::is_type<ov::op::v8::MatrixNms>(node)) {
        const auto nms =
            std::make_shared<ov::op::v8::MatrixNms>(outputs[0], outputs[1], ov::op::v8::MatrixNms::Attributes());
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
        return std::make_shared<ov::Model>(results, params, "MatrixNms");
    } else if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        const auto nms = std::make_shared<ov::op::v8::MulticlassNms>(outputs[0], outputs[1], ov::op::v8::MulticlassNms::Attributes());
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nms)};
        return std::make_shared<ov::Model>(results, params, "MulticlassNms");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateReadValueBase(const std::shared_ptr<ov::op::Op> &node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
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
    } else if (ov::is_type<ov::op::util::ReadValueBase>(node)) {
        return generateReadValueBase(node);
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