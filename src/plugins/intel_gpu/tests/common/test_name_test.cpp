// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_name.hpp"

#include <gtest/gtest.h>

#include <string>

TEST(smoke_GpuTestNames, ReplacesHyphensInParameterNames) {
    auto generate_name = ov::test::utils::make_hyphen_free_test_name([](bool negative) {
        return negative ? std::string{"axis=-1_epsilon=1e-09"} : std::string{"axis=1"};
    });

    EXPECT_EQ(generate_name(true), "axis=_1_epsilon=1e_09");
    EXPECT_EQ(generate_name(false), "axis=1");
}

class GpuTestNameIntegration : public ::testing::TestWithParam<int> {};

TEST_P(GpuTestNameIntegration, PreservesParameters) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    EXPECT_STREQ(info->name(), GetParam() < 0 ? "PreservesParameters/_1" : "PreservesParameters/1");
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         GpuTestNameIntegration,
                         ::testing::Values(-1, 1),
                         ov::test::utils::make_hyphen_free_test_name([](const ::testing::TestParamInfo<int>& info) {
                             return std::to_string(info.param);
                         }));

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
