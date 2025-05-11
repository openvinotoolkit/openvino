// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/nonzero.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

namespace {
using ov::test::NonZeroLayerTest;

std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{1000}},
    {{4, 1000}},
    {{2, 4, 1000}},
    {{2, 4, 4, 1000}},
    {{2, 4, 4, 2, 1000}},
};

const std::vector<ov::element::Type> model_types = {
    ov::element::i32,
    ov::element::f16,
    ov::element::u8,
};

std::map<std::string, std::string> config = {};

INSTANTIATE_TEST_SUITE_P(smoke_nonzero, NonZeroLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                            ::testing::ValuesIn(model_types),
                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                            ::testing::Values(config)),
                        NonZeroLayerTest::getTestCaseName);
} // namespace
