// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

TEST(StubTest, AlwaysPass) {
    // Some target platforms for the vectorized tests do not have any cases right now.
    // In order to make the build pass on these platforms, the build system will include this
    // file as the only source for the ov_cpu_unit_tests_vectorized, and the test binary
    // will always pass the run.
}
