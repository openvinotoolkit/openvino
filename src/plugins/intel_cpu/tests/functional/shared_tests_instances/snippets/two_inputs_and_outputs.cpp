// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/two_inputs_and_outputs.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

const std::vector<std::vector<InputShape>> input_shapes = {
        { {{}, {{5, 5, 256, 1}}}, {{}, {{5, 5, 256, 1}}} },
        { {{}, {{5, 5, 16, 35}}}, {{}, {{5, 5, 16, 35}}} },
        { {{}, {{5, 5, 256, 1}}}, {{}, {{5, 5, 256, 35}}} },
        { {{}, {{5, 5, 256, 1}}}, {{}, {{5, 5, 1, 1}}} },

        { {{}, {{5, 5, 16, 35}}}, {{}, {{5, 5, 1, 1}}} },
        { {{}, {{5, 5, 16, 35}}}, {{}, {{5, 5, 16, 1}}} },
        { {{}, {{5, 5, 5, 35}}}, {{}, {{5, 5, 1, 35}}} },
        { {{}, {{5, 5, 16, 1}}}, {{}, {{5, 5, 1, 35}}} },

        { {{}, {{5, 5, 35, 16}}}, {{}, {{5, 5, 35, 16}}} },
        { {{}, {{5, 5, 35, 16}}}, {{}, {{5, 5, 1, 16}}} },

        { {{}, {{5, 5, 35, 17}}}, {{}, {{5, 5, 35, 17}}} },
        { {{}, {{5, 5, 35, 17}}}, {{}, {{5, 5, 1, 17}}} },
        { {{}, {{5, 5, 35, 18}}}, {{}, {{5, 5, 35, 18}}} },
        { {{}, {{5, 5, 35, 18}}}, {{}, {{5, 5, 1, 18}}} },

        // DS
        { {{-1, -1, -1, -1}, {{5, 5, 256, 1}, {3, 3, 8, 15}, {5, 5, 256, 1}}},
          {{-1, -1, -1, -1}, {{5, 5, 256, 1}, {1, 1, 8,  1}, {5, 5, 256, 1}}} },
        { {{{1, 5}, {2, 6}, {3, 7}, {4, 8}}, {{1, 2, 3, 4}, {5, 6, 7, 8}, {1, 2, 3, 4}}},
          {{1, 1, 1, -1}, {{1, 1, 1, 4}, {1, 1, 1, 8}, {1, 1, 1, 4}}} },
        { {{1, -1, {1, 10}, {4, 12}}, {{1, 5, 3, 4}, {1, 10, 8, 12}, {1, 5, 3, 4}}},
          {{1, 1, -1, {1, 12}}, {{1, 1, 3, 1}, {1, 1, 8, 12}, {1, 1, 3, 1}}} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise,
                         TwoInputsAndOutputs,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(2),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TwoInputsAndOutputs::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise,
                         TwoInputsAndOutputsWithReversedOutputs,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::Values(2),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TwoInputsAndOutputsWithReversedOutputs::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov