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
    const auto params = ngraph::builder::makeDynamicParams({ov::element::f32, ov::element::f32, ov::element::i64}, {{b, s, i}, {b, n, h}, {b}});
    const auto W = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 3, i},
        {1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,  3.81946,
         5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373, 8.22482, 1.83521,
         5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667,
         2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555,
         9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595,  6.09321, 6.52544, 9.60882, 3.34881,
         3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937, 1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709,
         5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545,
         7.8691,  4.20879, 7.77509, 8.93208, 1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823,
         7.86628, 7.94436, 3.71224, 7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101,
         8.48523, 5.96192, 1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
         2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
         4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714, 2.38005,
         9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923,
         2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715,
         8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
         2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681, 6.60963, 3.92066,
         7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,  7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202,
         9.33716, 3.6506,  2.50256, 1.21691, 5.06801, 8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867,
         1.11441, 5.13591, 9.65186, 4.00767, 5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652,
         3.1734,  1.21496, 9.69154, 4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,
         8.77422, 3.11741, 8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766,
         8.07546, 3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
         9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967,
         7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,  5.86648, 8.73895,
         2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439,  10});
    const auto R = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 3, h},
        {1,       9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985, 8.6168,  3.81946,
         5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215, 8.0055,  7.44373, 8.22482, 1.83521,
         5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319, 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667,
         2.27908, 8.04983, 4.71285, 1.30754, 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555,
         9.13033, 2.07541, 5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595,  6.09321, 6.52544, 9.60882, 3.34881,
         3.07914, 5.80104, 9.54944, 5.43754, 5.8654,  7.88937, 1.40811, 2.2597,  8.13163, 1.26821, 8.94813, 5.86709,
         5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053, 2.13683, 7.21184, 6.82974, 4.18545,
         7.8691,  4.20879, 7.77509, 8.93208, 1.10502, 5.48298, 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823,
         7.86628, 7.94436, 3.71224, 7.95465, 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101,
         8.48523, 5.96192, 1.63073, 5.25228, 7.68488, 2.7276,  5.1788,  3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
         2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,  1.82401, 6.1306,
         4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686, 4.85031, 4.85544, 4.25714, 2.38005,
         9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891, 2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923,
         2.45457, 9.44841, 4.34021, 1.45016, 7.6686,  3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715,
         8.32489, 7.86316, 5.09754, 5.78644, 1.98402, 2.31429, 5.5791,  2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
         2.9146,  1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681, 6.60963, 3.92066,
         7.5521,  5.70463, 7.6313,  2.48866, 7.18352, 4.8413,  7.55702, 7.80702, 4.5785,  9.3268,  2.83159, 1.07202,
         9.33716, 3.6506,  2.50256, 1.21691, 5.06801, 8.27505, 4.31539, 6.48286, 1.31363, 4.1912,  1.70668, 7.23867,
         1.11441, 5.13591, 9.65186, 4.00767, 5.24875, 1.94852, 5.52768, 8.97121, 5.8094,  3.53329, 4.19126, 9.06652,
         3.1734,  1.21496, 9.69154, 4.86971, 4.1166,  6.19361, 2.13874, 9.55039, 3.8225,  9.57548, 2.96554, 3.2383,
         8.77422, 3.11741, 8.3359,  5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474,  4.59474, 6.19214, 8.80766,
         8.07546, 3.29232, 1.74029, 2.4198,  2.88544, 4.75644, 4.12921, 7.29896, 7.27759, 1.67252, 1.32823, 8.1046,
         9.10476, 1.04197, 3.37783, 5.2064,  4.23835, 3.16196, 1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967,
         7.16219, 3.4651,  1.09,    2.9788,  1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892,  5.86648, 8.73895,
         2.66603, 1.75192, 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439,  10});
    const auto B = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 3},
        {1,      9.97466, 9.39302, 2.15312, 9.99136, 3.1248,  4.56923, 4.4912,  7.02771, 9.41985,
         8.6168, 3.81946, 5.72093, 4.99108, 3.0662,  5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
         8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 10});
    const auto gru_sequence = std::make_shared<ov::op::v5::GRUSequence>(params[0],
                                                                        params[1],
                                                                        params[2],
                                                                        W,
                                                                        R,
                                                                        B,
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

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GroupConvolution>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 6}, {1, 1, 1, 3}});
    const auto group_convolution = std::make_shared<ov::op::v1::GroupConvolution>(params[0],
                                                                                  params[1],
                                                                                  ov::Strides{1},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::Strides{1});
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(group_convolution)};
    return std::make_shared<ngraph::Function>(results, params, "GroupConvolution");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 1, 4}, {1, 1, 1, 3}});
    const auto group_convolution = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(params[0],
                                                                                  params[1],
                                                                                  ov::Strides{1},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::CoordinateDiff{0},
                                                                                  ov::Strides{1},
                                                                                  ov::op::PadType{ov::op::PadType::EXPLICIT});
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(group_convolution)};
    return std::make_shared<ngraph::Function>(results, params, "GroupConvolutionBackpropData");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::HardSigmoid>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{3}});
    const auto alpha = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {std::vector<float>{0.5}});
    const auto beta = ngraph::builder::makeConstant<float>(ov::element::f32, {}, {std::vector<float>{0.6}});
    const auto hard_sigmoid = std::make_shared<ov::op::v0::HardSigmoid>(params[0], alpha, beta);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(hard_sigmoid)};
    return std::make_shared<ngraph::Function>(results, params, "HardSigmoid");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::Interpolate>& node) {
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
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(interpolate)};
    return std::make_shared<ngraph::Function>(results, params, "Interpolat-1");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v4::Interpolate>& node) {
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
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(interpolate)};
    return std::make_shared<ngraph::Function>(results, params, "Interpolate-4");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v3::Assign>& node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    auto read_value = std::make_shared<ov::op::v3::ReadValue>(params[0], "v0");
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    return std::make_shared<ngraph::Function>(results, SinkVector{assign}, params, "Assign-3");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v6::Assign>& node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(params[0], variable);
    auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
    return std::make_shared<ngraph::Function>(results, SinkVector{assign}, params, "Assign-6");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LRN>& node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{2, 3, 2, 1}});
    const auto axes = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {1}, std::vector<int64_t>{2});
    const auto lrn = std::make_shared<ov::op::v0::LRN>(params[0], axes, 3, 0.5, 1, 3);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lrn)};
    return std::make_shared<ngraph::Function>(results, params, "LRN");
}

std::shared_ptr<ov::Model> generate(const std::shared_ptr<ov::op::v0::LSTMSequence>& node) {
    const size_t b = 5;   // batch size
    const size_t i = 10;  // input size
    const size_t h = 10;  // hidden size
    const size_t s = 10;  // sequence length
    const size_t n = 1;   // num of direction
    const auto params = ngraph::builder::makeDynamicParams(
        {ov::element::f32, ov::element::f32, ov::element::f32, ov::element::i64},
        {{b, s, i}, {b, n, h}, {b, n, h}, {b}});
    const auto W = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 4, i},
        {1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
        8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
        8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
        7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
        6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
        5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
        3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
        8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
        2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
        1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
        2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
        1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
        2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
        1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
        4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
        2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
        7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
        5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
        2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
        6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
        4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
        4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
        5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
        9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
        8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
        6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
        7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
        1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
        1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
        1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
        3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
        2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
        1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
        5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
        1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
        6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
        5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
        5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
        8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
        5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10});
    const auto R = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 4, h},
        {1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
        8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
        8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
        7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
        6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
        5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
        3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
        8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
        2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
        1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
        2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
        1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
        2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
        1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
        4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
        2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
        7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
        5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
        2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
        6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
        4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
        4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
        5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
        9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
        8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
        6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
        7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
        1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
        1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
        1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
        3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
        2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
        1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
        5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
        1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
        6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
        5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
        5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
        8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
        5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 10});
    const auto B = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 4},
        {1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
        8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
        8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
        7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 10});
    const auto P = ngraph::builder::makeConstant<float>(
        ov::element::f32,
        {n, h * 3},
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});
    const auto lstm_sequence =
        std::make_shared<ov::op::v0::LSTMSequence>(params[0],
                                                    params[1],
                                                    params[2],
                                                    params[3],
                                                    W,
                                                    R,
                                                    B,
                                                    P,
                                                    h,
                                                    ov::op::RecurrentSequenceDirection::FORWARD,
                                                    ov::op::LSTMWeightsFormat::FICO,
                                                    std::vector<float>{},
                                                    std::vector<float>{},
                                                    std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                                    0.7f,
                                                    false);
    //ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_sequence)};
    return std::make_shared<ngraph::Function>(lstm_sequence->outputs(), params, "LSTMSequence");
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::boolean, {{1}, {1}});
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
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2, 4}, {1, 2, 2}});
    const auto outputs =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ov::op::v0::Parameter>(params));
    if (ov::is_type<ov::op::v8::MatrixNms>(node)) {
        const auto nms =
            std::make_shared<ov::op::v8::MatrixNms>(outputs[0], outputs[1], ov::op::v8::MatrixNms::Attributes());
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nms)};
        return std::make_shared<ngraph::Function>(results, params, "MatrixNms");
    } else if (ov::is_type<ov::op::v8::MulticlassNms>(node)) {
        const auto nms = std::make_shared<ov::op::v8::MulticlassNms>(outputs[0], outputs[1], ov::op::v8::MulticlassNms::Attributes());
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nms)};
        return std::make_shared<ngraph::Function>(results, params, "MulticlassNms");
    } else {
        return nullptr;
    }
}

std::shared_ptr<ov::Model> generateReadValueBase(const std::shared_ptr<ov::op::Op>& node) {
    auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1}});
    if (ov::is_type<ov::op::v3::ReadValue>(node)) {
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(params[0], "v0");
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
        auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        return std::make_shared<ngraph::Function>(results, SinkVector{assign}, params, "ReadValue-3");
    } else if (ov::is_type<ov::op::v6::ReadValue>(node)) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, "v0"});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(params[0], variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params[0]);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(add)};
        return std::make_shared<ngraph::Function>(results, SinkVector{assign}, params, "ReadValue-6");
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