// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/reshape_squeeze_reshape_relu.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {
std::vector<ShapeAxesTuple> inputs_squeeze{
    {{1, 1, 3}, {0, 1}},
    {{1, 1, 3}, {1}},
    {{1, 3, 1}, {0, 2}},
    {{3, 1, 1}, {1}},
    {{1, 4, 1, 3}, {0, 2}},
    {{3, 1, 2, 4, 4, 3}, {1}},
    {{1, 1, 1, 1, 1, 3}, {0, 1, 2, 3, 4}},
    {{1}, {0}},
};

std::vector<ShapeAxesTuple> inputs_unsqueeze{
    {{1}, {0}},
    {{1}, {0, 1}},
    {{1}, {0, 1, 2}},
    {{1, 2, 3}, {0}},
    {{1, 1, 3}, {1, 2}},
    {{1, 4, 1, 3}, {0, 2}},
};

std::vector<ov::element::Type> input_types = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<ov::test::utils::SqueezeOpType> opTypes = {ov::test::utils::SqueezeOpType::SQUEEZE,
                                                             ov::test::utils::SqueezeOpType::UNSQUEEZE};

INSTANTIATE_TEST_SUITE_P(smoke_reshape_squeeze_reshape_relu,
                         ReshapeSqueezeReshapeRelu,
                         ::testing::Combine(::testing::ValuesIn(inputs_squeeze),
                                            ::testing::ValuesIn(input_types),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(opTypes)),
                         ReshapeSqueezeReshapeRelu::getTestCaseName);
}  // namespace
