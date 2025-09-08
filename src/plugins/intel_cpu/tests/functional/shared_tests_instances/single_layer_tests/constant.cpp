// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/constant.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConstantLayerTest;

std::vector<ov::Shape> shapes{
    {2, 2, 3},
    {3, 4, 1},
    {1, 1, 12},
};

std::vector<ov::element::Type> model_types{
    ov::element::bf16, ov::element::f16,
    ov::element::f32,  ov::element::f64,
    ov::element::u4,   ov::element::u8,
    ov::element::u16,  ov::element::u32,
    ov::element::i4,   ov::element::i8,
    ov::element::i16,  ov::element::i32,
};

std::vector<std::string> data{"0", "1", "2", "3", "4", "5", "6", "7", "0", "1", "2", "3"};

std::vector<ov::element::Type> model_types_with_negative_values{
    ov::element::bf16, ov::element::f16,
    ov::element::f32,  ov::element::f64,
    ov::element::i4,   ov::element::i8,
    ov::element::i16,  ov::element::i32,
};

std::vector<std::string> dataWithNegativeValues{"1", "-2", "3", "-4", "5", "-6",
                                                "7", "-1", "2", "-3", "4", "-5"};

INSTANTIATE_TEST_SUITE_P(smoke_Constant, ConstantLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(model_types), ::testing::Values(data),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_with_negative_values, ConstantLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(model_types_with_negative_values),
                                           ::testing::Values(dataWithNegativeValues),
                                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ConstantLayerTest::getTestCaseName);
}  // namespace
