// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/if_const_non_const_bodies.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_If_ConstNonConst,
                         IfConstNonConst,
                         ::testing::Combine(::testing::Values(false), ::testing::Values(false)),
                         IfConstNonConst::getTestCaseName);

}  // namespace test
}  // namespace ov
