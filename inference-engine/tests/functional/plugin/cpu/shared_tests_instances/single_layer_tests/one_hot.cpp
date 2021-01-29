// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/one_hot.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U64,
        InferenceEngine::Precision::U16,
        // [NOT_IMPLEMENTED] Input image format U32 is not supported yet...
        //InferenceEngine::Precision::U32,
        /*
            /home/nii/source/openvino_fresh/inference-engine/tests/functional/shared_test_classes/src/base/layer_test_utils.cpp:189: Failure
            OneHot_835 Incorrect input precision for the input. Only I32 and FP32 are supported!
            /home/nii/source/openvino_fresh/inference-engine/src/mkldnn_plugin/nodes/one_hot.cpp:42
            /home/nii/source/openvino_fresh/inference-engine/include/details/ie_exception_conversion.hpp:64
         */
        //InferenceEngine::Precision::U8,
        //InferenceEngine::Precision::I8,
};

using namespace ngraph::element;

const std::vector<ngraph::element::Type> argDepthType_IC = { ngraph::element::i64 };
const std::vector<int64_t> argDepth_IC = { 1, 5, 1017 };
const std::vector<ngraph::element::Type> argSetType_IC = { ngraph::element::i64 };
const std::vector<float> argOnValue_IC = { 0, 1, -29 };
const std::vector<float> argOffValue_IC = { 0, 1, -127 };
const std::vector<int64_t> argAxis_IC = {0};
const std::vector<std::vector<size_t>> inputShapes_IC = {{4, 5}, {3, 7}};

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


const std::vector<ngraph::element::Type> argDepthType_Ax = { ngraph::element::i64 };
const std::vector<int64_t> argDepth_Ax = { 3 };
const std::vector<ngraph::element::Type> argSetType_Ax = { ngraph::element::i64 };
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


const std::vector<ngraph::element::Type> argDepthType_T = { ngraph::element::i32, ngraph::element::i16,
                                                            ngraph::element::i8, ngraph::element::u64,
                                                            ngraph::element::u32, ngraph::element::u16,
                                                            ngraph::element::u8 };
const std::vector<int64_t> argDepth_T = { 1 };
const std::vector<ngraph::element::Type> argSetType_T = { ngraph::element::i32, ngraph::element::i16,
                                                          ngraph::element::i8, ngraph::element::u64,
                                                          ngraph::element::u32, ngraph::element::u16,
                                                          ngraph::element::u8, ngraph::element::f64,
                                                          ngraph::element::f32, ngraph::element::f16,
                                                          ngraph::element::bf16, ngraph::element::boolean };
const std::vector<float> argOnValue_T = { 1 };
const std::vector<float> argOffValue_T = { 1 };
const std::vector<int64_t> argAxis_T = {-1};
const std::vector<std::vector<size_t>> inputShapes_T = {{2, 2}};

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
}  // namespace
