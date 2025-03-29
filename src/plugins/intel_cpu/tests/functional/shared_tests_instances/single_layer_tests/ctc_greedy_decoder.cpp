// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/ctc_greedy_decoder.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::CTCGreedyDecoderLayerTest;

// Common params
const std::vector<ov::element::Type> model_type = {
    ov::element::f32,
    ov::element::f16
};
std::vector<bool> mergeRepeated{true, false};

std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{ 50, 3, 3 }},
    {{ 50, 3, 7 }},
    {{ 50, 3, 8 }},
    {{ 50, 3, 16 }},
    {{ 50, 3, 128 }},
    {{ 50, 3, 49 }},
    {{ 50, 3, 55 }},
    {{ 1, 1, 16 }}};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(model_type),
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(mergeRepeated),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderBasic, CTCGreedyDecoderLayerTest,
                        basicCases,
                        CTCGreedyDecoderLayerTest::getTestCaseName);
}  // namespace
