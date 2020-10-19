// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::vector<size_t>> shapesA = {
        {1, 4, 5, 6}
};

const std::vector<std::vector<size_t>> shapesB = {
        {1, 4, 6, 4}
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

// OK
INSTANTIATE_TEST_CASE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(shapesA),
                ::testing::ValuesIn(shapesB),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        MatMulTest::getTestCaseName);

//////////////////////////////////////////////////////////////
//// NEW ADDED DEV TESTS //////////

// OK
INSTANTIATE_TEST_CASE_P(smoke_MatMul_1_2_x_2_1_false_false, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{1, 2}),
                ::testing::Values(std::vector<size_t>{2, 1}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);

// OK
INSTANTIATE_TEST_CASE_P(smoke_MatMul_1_2_x_1_2_false_true, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{1, 2}),
                ::testing::Values(std::vector<size_t>{1, 2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);


//// (For secondaryInputTypes - CONSTANT)
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_2_false_true_const, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));


//// (For secondaryInputTypes - PARAMETER)
// C++ exception with description "Unsupported input dims count for layer MatMul_13151
// /home/kmitrus/projects/ovino/openvino/inference-engine/src/mkldnn_plugin/nodes/mkldnn_gemm_node.cpp:46
// /home/kmitrus/projects/ovino/openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_2_false_true_param, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(secondaryInputTypes[1]), // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);


// C++ exception with description "Unsupported input dims count for layer MatMul_14324
// /home/kmitrus/projects/ovino/openvino/inference-engine/src/mkldnn_plugin/nodes/mkldnn_gemm_node.cpp:46
// /home/kmitrus/projects/ovino/openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_2_false_false_param, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));


//// (For secondaryInputTypes - CONSTANT)
// Aborted (core dumped)
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_2_false_false_const, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

//(Should fail)
// C++ exception with description "Check 'arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1])' failed at ngraph/core/src/op/dot.cpp:132:
// While validating node 'v0::Dot Dot_15575 (Parameter_15572[0]:f32{1,2}, Parameter_15573[0]:f32{1,2}) -> (dynamic?)' with friendly_name 'Dot_15575':
// Paired axes (axis 1 from arg0, axis 0 from arg1) do not have same length (arg0 shape: {1,2}, arg1 shape: {1,2}, reduction axes count: 1).
// INSTANTIATE_TEST_CASE_P(smoke_MatMul_1_2_x_1_2_false_false, MatMulTest,
//         ::testing::Combine(
//                 ::testing::Values(inputPrecisions[0]),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Layout::ANY),
//                 ::testing::Values(std::vector<size_t>{1, 2}),
//                 ::testing::Values(std::vector<size_t>{1, 2}),
//                 ::testing::Values(false),
//                 ::testing::Values(false),
//                 ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
//                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
//                 MatMulTest::getTestCaseName);


// (Should fail)
// C++ exception with description "Check 'arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1])' failed at ngraph/core/src/op/dot.cpp:132:
// While validating node 'v0::Dot Dot_15498 (Parameter_15495[0]:f32{3}, Parameter_15496[0]:f32{5}) -> (dynamic?)' with friendly_name 'Dot_15498':
// Paired axes (axis 0 from arg0, axis 0 from arg1) do not have same length (arg0 shape: {3}, arg1 shape: {5}, reduction axes count: 1).

// INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_x_5_false_true, MatMulTest,
//         ::testing::Combine(
//                 ::testing::Values(inputPrecisions[0]),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Layout::ANY),
//                 ::testing::Values(std::vector<size_t>{3}),
//                 ::testing::Values(std::vector<size_t>{5}),
//                 ::testing::Values(false),
//                 ::testing::Values(true),
//                 ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
//                 ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// (Should fail)
// C++ exception with description "Check 'arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1])' failed at ngraph/core/src/op/dot.cpp:132:
// While validating node 'v0::Dot Dot_15502 (Parameter_15499[0]:f32{3}, Parameter_15500[0]:f32{5}) -> (dynamic?)' with friendly_name 'Dot_15502':
// Paired axes (axis 0 from arg0, axis 0 from arg1) do not have same length (arg0 shape: {3}, arg1 shape: {5}, reduction axes count: 1).

// INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_x_5_false_false, MatMulTest,
//         ::testing::Combine(
//                 ::testing::Values(inputPrecisions[0]),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Layout::ANY),
//                 ::testing::Values(std::vector<size_t>{3}),
//                 ::testing::Values(std::vector<size_t>{5}),
//                 ::testing::Values(false),
//                 ::testing::Values(false),
//                 ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
//                 ::testing::Values(CommonTestUtils::DEVICE_CPU)));

INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_x_5_true_false, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3}),
                ::testing::Values(std::vector<size_t>{5}),
                ::testing::Values(true),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));


INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_2_1_2_true_false, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(std::vector<size_t>{2, 1, 2}),
                ::testing::Values(true),
                ::testing::Values(false),
                ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);

// Core dump for Constant
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_1_3_x_3_false_false, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2, 1, 3}),
                ::testing::Values(std::vector<size_t>{3}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[1]),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_1_3_x_3_false_true, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{2, 1, 3}),
                ::testing::Values(std::vector<size_t>{3}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);


INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_1_2_x_2_false_true, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 1, 2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::ValuesIn(secondaryInputTypes),  // CONSTANT/PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                MatMulTest::getTestCaseName);

/////////////////////////////////////////////////////////////////////////////
// Tests based on transformation to FC/GEMM tests in `convert_matmul_test.cpp`
//
// Test1
INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_1_2_x_2_1_false_true, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 2, 2}),
                ::testing::Values(std::vector<size_t>{2, 1}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// Test2 Aborted (core dumped)
// INSTANTIATE_TEST_CASE_P(MatMul_3_1_2_x_2_false_false, MatMulTest,
//         ::testing::Combine(
//                 ::testing::Values(inputPrecisions[0]),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Layout::ANY),
//                 ::testing::Values(std::vector<size_t>{3, 1, 2}),
//                 ::testing::Values(std::vector<size_t>{2}),
//                 ::testing::Values(false),
//                 ::testing::Values(false), // In the transform tests is false, should be true?
//                 ::testing::Values(secondaryInputTypes[0]),
//                 ::testing::Values(CommonTestUtils::DEVICE_CPU)));

INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_1_2_x_2_false_false_param, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 1, 2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_1_2_x_2_false_false_const, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 1, 2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// Test3 Aborted (core dumped)
// INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_3_1_2_false_false, MatMulTest,
//         ::testing::Combine(
//                 ::testing::Values(inputPrecisions[0]),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//                 ::testing::Values(InferenceEngine::Layout::ANY),
//                 ::testing::Values(std::vector<size_t>{3, 1, 2}),
//                 ::testing::Values(std::vector<size_t>{2}),
//                 ::testing::Values(false),
//                 ::testing::Values(false),
//                 ::testing::Values(secondaryInputTypes[0]),
//                 ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// Test3 - OK
INSTANTIATE_TEST_CASE_P(smoke_MatMul_2_x_3_1_2_false_false_param, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 1, 2}),
                ::testing::Values(std::vector<size_t>{2}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

//Test4 - OK
INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_2_2_x_3_2_1_false_true_const, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 2, 2}),
                ::testing::Values(std::vector<size_t>{3, 2, 1}),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// Test5 - convertable to FC
INSTANTIATE_TEST_CASE_P(MatMul_3_2_2_x_2_2_false_true_const, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 2, 2}),
                ::testing::Values(std::vector<size_t>{2, 2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(secondaryInputTypes[0]),  // CONSTANT
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

// Test5 (PARAM)
INSTANTIATE_TEST_CASE_P(smoke_MatMul_3_2_2_x_2_2_false_true_param, MatMulTest,
        ::testing::Combine(
                ::testing::Values(inputPrecisions[0]),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>{3, 2, 2}),
                ::testing::Values(std::vector<size_t>{2, 2}),
                ::testing::Values(false),
                ::testing::Values(true),
                ::testing::Values(secondaryInputTypes[1]),  // PARAMETER
                ::testing::Values(CommonTestUtils::DEVICE_CPU)));

} // namespace
