// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/one_hot.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I16,
        // Not implemented:
        //InferenceEngine::Precision::I16,
        //InferenceEngine::Precision::U16,
        //InferenceEngine::Precision::I8,
        //InferenceEngine::Precision::U8,
};

using namespace ngraph::element;

const std::vector<Type> argDepthType_IC = { i64 };
const std::vector<int64_t> argDepth_IC = { 1, 5, 1017 };
const std::vector<Type> argSetType_IC = { i64 };
const std::vector<float> argOnValue_IC = { 0, 1, -29, 4098 };
const std::vector<float> argOffValue_IC = { 0, 1, -127, 7019 };
const std::vector<int64_t> argAxis_IC = {0};
const std::vector<std::vector<size_t>> inputShapes_IC = {{13, 5}, {3, 28}};

const auto oneHotParams_IC = testing::Combine(
        testing::ValuesIn(argDepthType_IC),
        testing::ValuesIn(argDepth_IC),
        testing::ValuesIn(argSetType_IC),
        testing::ValuesIn(argOnValue_IC),
        testing::ValuesIn(argOffValue_IC),
        testing::ValuesIn(argAxis_IC),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes_IC),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotIntConst,
        OneHotLayerTest,
        oneHotParams_IC,
        OneHotLayerTest::getTestCaseName
);


const std::vector<Type> argDepthType_Ax = { i64 };
const std::vector<int64_t> argDepth_Ax = { 13 };
const std::vector<Type> argSetType_Ax = { i64 };
const std::vector<float> argOnValue_Ax = { 17 };
const std::vector<float> argOffValue_Ax = { -3 };
const std::vector<int64_t> argAxis_Ax = {0, 1, 3, 5, -4, -5};
const std::vector<std::vector<size_t>> inputShapes_Ax = {{4, 8, 5, 3, 2, 9}};

const auto oneHotParams_Ax = testing::Combine(
        testing::ValuesIn(argDepthType_Ax),
        testing::ValuesIn(argDepth_Ax),
        testing::ValuesIn(argSetType_Ax),
        testing::ValuesIn(argOnValue_Ax),
        testing::ValuesIn(argOffValue_Ax),
        testing::ValuesIn(argAxis_Ax),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes_Ax),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotAxrng,
        OneHotLayerTest,
        oneHotParams_Ax,
        OneHotLayerTest::getTestCaseName
);


const std::vector<Type> argDepthType_T = { i32, i16, i8, u64, u32, u16, u8};
const std::vector<int64_t> argDepth_T = { 1 };
const std::vector<Type> argSetType_T = { i32, i16, i8, u64, u32, u16, u8, f64, f32, f16, bf16, boolean };
const std::vector<float> argOnValue_T = { 1 };
const std::vector<float> argOffValue_T = { 1 };
const std::vector<int64_t> argAxis_T = {-1};
const std::vector<std::vector<size_t>> inputShapes_T = {{4, 9}};

const auto oneHotParams_T = testing::Combine(
        testing::ValuesIn(argDepthType_T),
        testing::ValuesIn(argDepth_T),
        testing::ValuesIn(argSetType_T),
        testing::ValuesIn(argOnValue_T),
        testing::ValuesIn(argOffValue_T),
        testing::ValuesIn(argAxis_T),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes_T),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotArgType,
        OneHotLayerTest,
        oneHotParams_T,
        OneHotLayerTest::getTestCaseName
);

/*
 const std::vector<depth_pair> argDepth_T = { {Type_t::i32, 5}, {Type_t::i16, 5}, {Type_t::i8, 5},
                                             {Type_t::u32, 5}, {Type_t::u16, 5}, {Type_t::u8, 5},
                                             {Type_t::u64, 5},
                                             };

const std::vector<set_pair> argOnValue_T = { {Type_t::f64, 7}, {Type_t::f32, 7}, {Type_t::f16, 7},
                                             {Type_t::i32, 7}, {Type_t::i16, 7}, {Type_t::i8, 7},
                                             {Type_t::u32, 7}, {Type_t::u16, 7}, {Type_t::u8, 7},
                                             {Type_t::u64, 7}, {Type_t::bf16, 7}, {Type_t::boolean, 1},
                                             };

const std::vector<set_pair> argOffValue_T = { {Type_t::f64, 2}, {Type_t::f32, 2}, {Type_t::f16, 2},
                                              {Type_t::i32, 2}, {Type_t::i16, 2}, {Type_t::i8, 2},
                                              {Type_t::u32, 2}, {Type_t::u16, 2}, {Type_t::u8, 2},
                                              {Type_t::u64, 2}, {Type_t::bf16, 2}, {Type_t::boolean, 0},
                                              };

const std::vector<int64_t> argAxis_T = {0};
const std::vector<std::vector<size_t>> inputShapes_T = {{7, 5}};

const auto oneHotParams_T = testing::Combine(
        testing::ValuesIn(argDepth_T),
        testing::ValuesIn(argOnValue_T),
        testing::ValuesIn(argOffValue_T),
        testing::ValuesIn(argAxis_T),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputPrecisions),
        testing::ValuesIn(outputPrecisions),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(input_shapesAx),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(
        smoke_OneHotArgType,
        OneHotLayerTest,
        oneHotParams_T,
        OneHotLayerTest::getTestCaseName
);
 */
}  // namespace
