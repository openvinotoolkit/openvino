// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/one_hot.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I32,
        // Not implemented:
        //InferenceEngine::Precision::I16,
        //InferenceEngine::Precision::U16,
        //InferenceEngine::Precision::I8,
        //InferenceEngine::Precision::U8,
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::UNSPECIFIED,
//        InferenceEngine::Precision::BF16,
//        InferenceEngine::Precision::I8
};

const std::vector<InferenceEngine::Precision> outputPrecisions = {
            InferenceEngine::Precision::UNSPECIFIED,
//        InferenceEngine::Precision::BF16,
//        InferenceEngine::Precision::I8
};

typedef std::pair<ngraph::element::Type, int64_t> depth_pair;
typedef std::pair<ngraph::element::Type, float> set_pair;
using ngraph::element::Type_t;

const std::vector<depth_pair> argDepthConst = { {Type_t::i64, 1}, {Type_t::i64, 5}, {Type_t::i64, 1017} };
const std::vector<set_pair> argOnValueConst = { {Type_t::i64, 0}, {Type_t::i64, 1}, {Type_t::i64, -29}, {Type_t::i64, 7019}};
const std::vector<set_pair> argOffValueConst = { {Type_t::i64, 0}, {Type_t::i64, 1}, {Type_t::i64, -722}, {Type_t::i64, 4825}};
const std::vector<int64_t> argAxisConst = {0};
const std::vector<std::vector<size_t>> input_shapesConst = {{13, 5}, {3, 28}};

const auto oneHotConstParams = testing::Combine(
        testing::ValuesIn(argDepthConst),
        testing::ValuesIn(argOnValueConst),
        testing::ValuesIn(argOffValueConst),
        testing::ValuesIn(argAxisConst),
        testing::ValuesIn(netPrecisions),                   // Net precision
        testing::ValuesIn(inputPrecisions),                 // Input precision
        testing::ValuesIn(outputPrecisions),                // Output precision
        testing::Values(InferenceEngine::Layout::ANY),  // Input layout
        testing::ValuesIn(input_shapesConst),               // Input shapes
        testing::Values(CommonTestUtils::DEVICE_CPU)      // Target device name
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotConst,
        OneHotLayerTest,
        oneHotConstParams,
        OneHotLayerTest::getTestCaseName
);
/*
const std::vector<int64_t> argDepthAx = {13};
const std::vector<float> argOnValueAx = {17};
const std::vector<float> argOffValueAx = {-3};
const std::vector<int64_t> argAxisAx = {0, 1, 3, 5, -4, -5};
const std::vector<std::vector<size_t>> input_shapesAx = {{4, 8, 5, 3, 2, 9}};

const auto oneHotAxParams = testing::Combine(
        testing::ValuesIn(argDepthAx),
        testing::ValuesIn(argOnValueAx),
        testing::ValuesIn(argOffValueAx),
        testing::ValuesIn(argAxisAx),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(outputPrecisions),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(input_shapesAx),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotAxrng,
        OneHotLayerTest,
        oneHotAxParams,
        OneHotLayerTest::getTestCaseName
);
 */
}  // namespace
