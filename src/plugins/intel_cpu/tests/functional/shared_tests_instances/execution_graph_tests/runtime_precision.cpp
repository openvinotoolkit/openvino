// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/runtime_precision.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;
using namespace InferenceEngine;

namespace {

const std::vector<RuntimePrecisionSpecificParams> params = {
        /* {Ngraph function builder, function input precision, expected runtime precisions} */
        {makeEltwiseFunction, {Precision::FP32, Precision::FP32}, {{"Eltwise", Precision::FP32}}},
        {makeEltwiseFunction, {Precision::U16, Precision::U16}, {{"Eltwise", Precision::I32}}},
        {makeEltwiseFunction, {Precision::BF16, Precision::BF16}, {{"Eltwise", Precision::BF16}}},
        {makeEltwiseFunction, {Precision::U8, Precision::U8}, {{"Eltwise", Precision::U8}}},
        {makeEltwiseFunction, {Precision::I8, Precision::I8}, {{"Eltwise", Precision::I8}}},
        {makeFakeQuantizeReluFunction, {Precision::FP32}, {{"Relu", Precision::FP32}}},
        {makeFakeQuantizeReluFunction, {Precision::U8}, {{"Relu", Precision::U8}}},
        {makeFakeQuantizeBinaryConvolutionFunction, {Precision::FP32}, {{"FakeQuantize", Precision::FP32}, {"BinaryConvolution", Precision::BIN}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExecGraph, ExecGraphRuntimePrecision,
                        ::testing::Combine(
                                ::testing::ValuesIn(params),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ExecGraphRuntimePrecision::getTestCaseName);
}  // namespace
