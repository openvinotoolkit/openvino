// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/mat_mul_transformation.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

const std::vector<LayerTestsUtils::LayerTransformation::LptVersion> versionValues = {
    LayerTestsUtils::LayerTransformation::LptVersion::cnnNetwork,
    LayerTestsUtils::LayerTransformation::LptVersion::nGraph
};

static const size_t levels = 256ul;
static const float k = 2.f;
static const float lowU8 = 0.f;
static const float highU8 = 255.f / k;
static const float lowI8 = -128.f / k;
static const float highI8 = 127.f / k;
static const ngraph::element::Type precision = ngraph::element::f32;

// 384  / 64 = 6
// 1000 / 64 =>16
// 1024 / 64 = 16
// 2048 / 64 = 32
// 4096 / 64 = 64

std::vector<ngraph::builder::subgraph::MatMulFunctionBranches> branches = {
    // {
    //    {{ 1, 16, 384, 64 }, {}, {}, {}, {}},
    //    {{ 1, 16, 64, 384 }, {}, {}, {}, {}}
    // },

    // {
    //    {{ 1, 16, 384, 64 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 16, 1, 1}, {}}},
    //    {{ 1, 16, 64, 384 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 16, 1, 1}, {}}}
    // },

    {
        {{ 1, 4, 16, 6 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 4, 1, 1}, {}}},
        {{ 1, 4, 6, 16 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1, 4, 1, 1}, {}}}
    },

    // {
    //    {{ 1, 16, 384, 64 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1}, {2.f}}},
    //    {{ 1, 16, 64, 384 }, {ngraph::element::u8}, {ngraph::element::f32}, {}, {{1}, {3.f}}}
    // }
};

// std::vector<std::vector<std::shared_ptr<ngraph::Node>>> branches = {
//    // FullyConnected 2D: 1x2048 & 2048x1000 => 1x1000 (Inception v3 ONNX)
//    {
//        ngraph::builder::makeFakeQuantize(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1ul, 32ul })),
//            precision, levels, {1ul, 1ul}, { lowU8 }, { highU8 }, { lowU8 }, { highU8 }),
//
//        ngraph::builder::makeFakeQuantize(
//            ngraph::builder::makeConstant(ngraph::element::f32, ngraph::Shape({ 32ul, 16ul }), {}, true),
//            precision, levels, {1ul, 1ul}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 }),
//
//        ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16}, {}, true)
//    },
//
//    // FullyConnected 3D: 1x384x1024 & 1024x2 => 1x384x2 (BERT ICV)
//    {
//        ngraph::builder::makeFakeQuantize(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1ul, 6ul, 16ul })),
//            ngraph::element::f32, levels, {1ul, 1ul, 1ul}, { lowU8 }, { highU8 }, { lowU8 }, { highU8 }),
//
//        ngraph::builder::makeFakeQuantize(
//            ngraph::builder::makeConstant(ngraph::element::f32, ngraph::Shape({ 16ul, 2ul }), {}, true),
//            ngraph::element::f32, levels, {1ul, 1ul}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 }),
//
//        ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 1, 2}, {}, true)
//    },
//
//    // FullyConnected 3D : 1x384x1024 & 1024x4096 => 1x384x4096 (BERT MLPerf from FP32)
//    {
//        ngraph::builder::makeFakeQuantize(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1ul, 6ul, 16ul })),
//            ngraph::element::f32, levels, {}, { lowU8 }, { highU8 }, { lowU8 }, { highU8 }),
//
//        ngraph::builder::makeFakeQuantize(
//            ngraph::builder::makeConstant(ngraph::element::f32, ngraph::Shape({ 16ul, 64ul }), {}, true),
//            ngraph::element::f32, levels, {1ul, 64ul}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 }),
//
//        ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 1, 64}, {}, true)
//    },
//
//    // TODO: not completed
//    // // GEMM 4D : 1x16x384x64 & 1x16x64x384 => 1x16x384x384 (BERT MLPerf)
//    // {
//    //    makeFakeQuantize(
//    //        std::make_shared<opset1::Parameter>(element::f32, Shape({ 1ul, 2ul, 6ul, 4ul })),
//    //        element::f32, levels, {}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 }),
//
//    //    makeFakeQuantize(
//    //        std::make_shared<opset1::Parameter>(element::f32, Shape({ 1ul, 2ul, 4ul, 6ul })),
//    //        element::f32, levels, {1ul, 2ul, 1ul, 1ul}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 })
//    // },
//
//    // GEMM 4D : updated: one interval for FQ: 1x16x384x64 & 1x16x64x384 => 1x16x384x384 (BERT MLPerf)
//    {
//        ngraph::builder::makeFakeQuantize(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1ul, 2ul, 6ul, 4ul })),
//            ngraph::element::f32, levels, {}, { lowI8 }, { highI8 }, { lowI8 }, { highI8 }),
//
//        ngraph::builder::makeFakeQuantize(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1ul, 2ul, 4ul, 6ul })),
//            ngraph::element::f32, levels, {}, { lowI8 / 2.f }, { highI8 / 2.f }, { lowI8 / 2.f }, { highI8 / 2.f })
//    },
//
//    // GEMM 4D : 1x16x384x64 & 1x16x64x384 = > 1x16x384x384(BERT MLPerf)
//    {
//        std::make_shared<ngraph::opset1::Multiply>(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 4, 8, 6 })),
//            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 4, 1, 1}, {}, true)),
//
//        std::make_shared<ngraph::opset1::Multiply>(
//            std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape({ 1, 4, 6, 8 })),
//            ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 4, 1, 1}, {}, true))
//    }
// };

INSTANTIATE_TEST_CASE_P(LPT, MatMulTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 384, 1024 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(versionValues),
        ::testing::ValuesIn(branches)),
    MatMulTransformation::getTestCaseName);
}  // namespace
