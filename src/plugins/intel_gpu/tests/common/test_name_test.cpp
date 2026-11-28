// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

TEST(smoke_GpuTestNames, NoHyphensInRegisteredNames) {
    const auto* tests = ::testing::UnitTest::GetInstance();
    for (int i = 0; i < tests->total_test_suite_count(); ++i) {
        const auto* suite = tests->GetTestSuite(i);
        ASSERT_EQ(std::string{suite->name()}.find('-'), std::string::npos) << suite->name();
        for (int j = 0; j < suite->total_test_count(); ++j) {
            const auto* test = suite->GetTestInfo(j);
            ASSERT_EQ(std::string{test->name()}.find('-'), std::string::npos) << suite->name() << '.' << test->name();
        }
    }
}
