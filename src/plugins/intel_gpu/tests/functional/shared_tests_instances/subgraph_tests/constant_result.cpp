// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/constant_result.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConstantSubgraphType;
using ov::test::ConstantResultSubgraphTest;

const std::vector<ConstantSubgraphType> types = {
    ConstantSubgraphType::SINGLE_COMPONENT,
    ConstantSubgraphType::SEVERAL_COMPONENT
};

const std::vector<ov::Shape> shapes = {
    {1, 3, 10, 10},
    {2, 3, 4, 5}
};

const std::vector<ov::element::Type> model_types = {
    ov::element::u8,
    ov::element::i8,
    ov::element::u16,
    ov::element::i16,
    ov::element::i32,
    ov::element::u64,
    ov::element::i64,
    ov::element::f32,
    ov::element::boolean
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ConstantResultSubgraphTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(types),
                            ::testing::ValuesIn(shapes),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConstantResultSubgraphTest::getTestCaseName);
}  // namespace

