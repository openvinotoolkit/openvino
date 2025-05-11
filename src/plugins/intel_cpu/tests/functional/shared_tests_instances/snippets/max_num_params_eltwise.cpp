// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/max_num_params_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
// Note that we need these shapes to cover all cases of code emission (none/one/multiple of scalar/vector tiles)
std::vector<InputShape> input_shapes {{{}, {{1, 64, 10, 10}}},
                                      {{}, {{1, 1, 17, 37}}},
                                      {{}, {{1, 1, 1, 1}}},
                                      {{}, {{1, 1, 1, 7}}},
                                      {{}, {{1, 1, 1, 128}}},
                                      {{}, {{1, 1, 1, 14}}},
                                      {{}, {{1, 1, 1, 16}}},
                                      {{}, {{1, 1, 1, 30}}},
                                      // DS
                                      {{-1, -1, -1, -1}, {{1, 64, 10, 10}, {1, 1, 17, 37}, {1, 64, 10, 10}}},
                                      {{1, {1, 64}, {10, 20}, -1}, {{1, 64, 10, 10}, {1, 1, 17, 37}, {1, 64, 10, 10}}},
                                      {{1, 1, 1, {1, 128}}, {{1, 1, 1, 1}, {1, 1, 1, 7}, {1, 1, 1, 128}, {1, 1, 1, 14}, {1, 1, 1, 16}, {1, 1, 1, 1}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MaxNumParamsEltwise, MaxNumParamsEltwise,
                         ::testing::Combine(
                             ::testing::ValuesIn(input_shapes),
                             ::testing::Values(2), // Subgraph + Concat
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MaxNumParamsEltwise::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov