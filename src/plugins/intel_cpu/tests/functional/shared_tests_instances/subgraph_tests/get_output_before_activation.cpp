// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/get_output_before_activation.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::OutputBeforeActivationNew;
using ov::test::midOutputType;

namespace {
    const std::vector<std::vector<ov::Shape>> shapes_static = {
        {{1, 80}},
        {{1, 32}},
        {{1, 64}},
        {{1, 100}}
    };

    std::vector<midOutputType> midLayerTypes {
        midOutputType::Mul,
        midOutputType::Sub,
        midOutputType::Sum
    };

    std::map<std::string, std::string> additional_config = {};
} // namespace

INSTANTIATE_TEST_SUITE_P(OutputBeforeActivationNew, OutputBeforeActivationNew,
    ::testing::Combine(
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
        ::testing::ValuesIn(midLayerTypes),
        ::testing::Values(additional_config)),
    OutputBeforeActivationNew::getTestCaseName);
