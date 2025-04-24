// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/constant_result.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {

const std::vector<ConstantSubgraphType> types = {ConstantSubgraphType::SINGLE_COMPONENT,
                                                 ConstantSubgraphType::SEVERAL_COMPONENT};

const std::vector<ov::Shape> shapes = {{1, 3, 10, 10}, {2, 3, 4, 5}};

const std::vector<ov::element::Type> precisions = {ov::element::u8,
                                                   ov::element::i8,
                                                   ov::element::u16,
                                                   ov::element::i16,
                                                   ov::element::u32,
                                                   ov::element::i32,
                                                   ov::element::u64,
                                                   ov::element::i64,
                                                   ov::element::f32,
                                                   ov::element::boolean};

INSTANTIATE_TEST_SUITE_P(smoke_Check,
                         ConstantResultSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(types),
                                            ::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConstantResultSubgraphTest::getTestCaseName);

}  // namespace
