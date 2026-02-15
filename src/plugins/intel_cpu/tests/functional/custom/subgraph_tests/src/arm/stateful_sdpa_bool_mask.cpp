// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/stateful_sdpa_bool_mask.hpp"

#include <gtest/gtest.h>

namespace ov {
namespace test {

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_ARM_StatefulSdpaBoolMask,
                         StatefulSdpaBoolMaskTest,
                         ::testing::Values(ov::element::f16),
                         StatefulSdpaBoolMaskTest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov
