// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compatibility_string.hpp"

namespace ov {
namespace test {
namespace behavior {

INSTANTIATE_TEST_SUITE_P(CompatibilityStringTestParameters,
                         CompatibilityStringTest,
                         ::testing::Combine(::testing::Values("PLUGIN", "DRIVER"),
                                            ::testing::Values("NO_WEIGHTS_COPY", "ALL_WEIGHTS_COPY"),
                                            ::testing::Values("NO", "YES")),
                         CompatibilityStringTest::getTestCaseName);

}  // namespace behavior
}  // namespace test
}  // namespace ov
