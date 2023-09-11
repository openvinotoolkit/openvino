// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/precision_propagation_convertion.hpp"
#include <gtest/gtest.h>
#include <ngraph/ngraph.hpp>

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::test::InputShape>> input_shapes = {
        { {{}, {{1, 3, 16, 16}}}, {{}, {{1, 1, 1, 16}}} },
        // DS
        { {{-1, -1, -1, -1}, {{1, 3, 16, 16}, {1, 1, 1, 16}, {1, 3, 16, 16}}}, {{-1, -1, -1, -1}, {{1, 3, 16, 16}, {1, 1, 1, 16}, {1, 3, 16, 16}}} },
        { {{1, 16, -1, {1, 16}}, {{1, 16, 32, 1}, {1, 16, 1, 16}, {1, 16, 32, 1}}}, {{1, 1, -1, {1, 20}}, {{1, 1, 1, 16}, {1, 1, 8, 16}, {1, 1, 1, 16}}} }
};

const std::vector<std::vector<float>> fake_quantize_intervals = {
    {0.f, 2.55f, 0.f, 2.55f},
    {-1.28f, 1.27f, -1.28f, 1.27f}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_PrecisionPropagation_Convertion, PrecisionPropagationConvertion,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes),
                                 ::testing::ValuesIn(fake_quantize_intervals),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         PrecisionPropagationConvertion::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
