// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/runtime_precision.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

const std::vector<RuntimePrecisionSpecificParams> params = {
        /* {Ngraph function builder, function input precision, expected runtime precisions} */
        {makeEltwiseFunction, {ov::element::f32, ov::element::f32}, {{"Eltwise", ov::element::f32}}},
        {makeEltwiseFunction, {ov::element::u16, ov::element::u16}, {{"Eltwise", ov::element::i32}}},
        {makeEltwiseFunction, {ov::element::bf16, ov::element::bf16}, {{"Eltwise", ov::element::bf16}}},
        {makeEltwiseFunction, {ov::element::u8, ov::element::u8}, {{"Eltwise", ov::element::u8}}},
        {makeEltwiseFunction, {ov::element::i8, ov::element::i8}, {{"Eltwise", ov::element::i8}}},
        {makeFakeQuantizeReluFunction, {ov::element::f32}, {{"Relu", ov::element::f32}}},
        {makeFakeQuantizeReluFunction, {ov::element::u8}, {{"Relu", ov::element::u8}}},
        {makeFakeQuantizeBinaryConvolutionFunction, {ov::element::f32}, {{"FakeQuantize", ov::element::f32}, {"BinaryConvolution", ov::element::u1}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExecGraph, ExecGraphRuntimePrecision,
                        ::testing::Combine(
                                ::testing::ValuesIn(params),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ExecGraphRuntimePrecision::getTestCaseName);
}  // namespace
