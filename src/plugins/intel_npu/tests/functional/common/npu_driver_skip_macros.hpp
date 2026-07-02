// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/npu_driver_aware_test.hpp"

// ---------------------------------------------------------------------------
// Driver-aware skip macros for ov_npu_func_tests.
//
// Call these at the TOP of a TEST_P / TEST_F body, not inside a helper function.
// GTEST_SKIP() expands to a bare `return;` — invoking it from a helper only
// exits the helper, leaving the test body running. These macros expand inline
// at the call site so `return` correctly exits the test.
//
// Convention: every skip reason MUST include a tracking ticket (E#XXXXXX).
// Temporary skips (regression windows) must also document the expected fix.
// ---------------------------------------------------------------------------

// Skip the test when the driver type matches `type` (PV, RELEASE, or LATEST).
// No-op when --driver_type CLI arg is not provided (dev machine).
//
// Example:
//   NPU_SKIP_IF_DRIVER_TYPE_IS(PV, "E#12345: feature X not supported on PV driver");
#define NPU_SKIP_IF_DRIVER_TYPE_IS(type, reason)                                             \
    do {                                                                                      \
        if (::ov::test::behavior::isDriverType(::ov::test::utils::DriverType::type)) {      \
            GTEST_SKIP() << (reason);                                                        \
        }                                                                                     \
    } while (0)

// Skip the test when the driver type does NOT match `type`.
// Use to restrict a test to a single driver type (e.g. PV-only validation).
//
// Example:
//   NPU_SKIP_UNLESS_DRIVER_TYPE_IS(PV, "E#12345: PV-specific blob test, irrelevant on other drivers");
#define NPU_SKIP_UNLESS_DRIVER_TYPE_IS(type, reason)                                         \
    do {                                                                                      \
        if (!::ov::test::behavior::isDriverType(::ov::test::utils::DriverType::type)) {     \
            GTEST_SKIP() << (reason);                                                        \
        }                                                                                     \
    } while (0)
