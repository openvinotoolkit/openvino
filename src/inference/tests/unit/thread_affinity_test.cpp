// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include "dev/threading/thread_affinity.hpp"

using namespace ov::threading;

#if defined(_WIN32)

// Regression test for: pin_thread_to_vacant_core crashes with read access violation
// when cpu_ids is empty (e.g. THREADING=SEQ build, VM, restricted process affinity on Windows).
// Fix: added cpu_ids.empty() guard that returns false instead of indexing out-of-bounds.

TEST(ThreadAffinityTest, PinThreadToVacantCore_EmptyCpuIds_ReturnsFalse) {
    CpuSet mask = nullptr;
    std::vector<int> empty_cpu_ids = {};

    // Before fix: read access violation crash on cpu_ids[thrIdx]
    // After fix:  returns false gracefully
    bool result = pin_thread_to_vacant_core(0, 1, 4, mask, empty_cpu_ids);

    EXPECT_FALSE(result);
}

TEST(ThreadAffinityTest, PinThreadToVacantCore_ValidCpuIds_DoesNotThrow) {
    // Normal path: ensure guard did not break the valid case
    CpuSet mask = nullptr;
    std::vector<int> cpu_ids = {0, 1, 2, 3};

    EXPECT_NO_THROW(pin_thread_to_vacant_core(0, 1, 4, mask, cpu_ids));
}

#endif  // defined(_WIN32)
