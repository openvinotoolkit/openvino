// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/get_output_before_activation.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::OutputBeforeActivation;
using ov::test::midOutputType;

namespace {
    std::vector<size_t> input_sizes = {
        80,
        32,
        64,
        100
    };

    std::vector<midOutputType> midLayerTypes {
        midOutputType::Mul,
        midOutputType::Sub,
        midOutputType::Sum
    };

    std::map<std::string, std::string> additional_config = {};
} // namespace

INSTANTIATE_TEST_SUITE_P(OutputBeforeActivation, OutputBeforeActivation,
    ::testing::Combine(
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(input_sizes),
        ::testing::ValuesIn(midLayerTypes),
        ::testing::Values(additional_config)),
    OutputBeforeActivation::getTestCaseName);
