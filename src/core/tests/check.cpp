// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/file_util.hpp"

using namespace std;

TEST(check, check_true_string_info) {
    OPENVINO_ASSERT(true, "this should not throw");
}

TEST(check, check_true_non_string_info) {
    OPENVINO_ASSERT(true, "this should not throw", 123);
}

TEST(check, check_true_no_info) {
    OPENVINO_ASSERT(true);
}

TEST(check, check_false_string_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false, "this should throw"); }, ov::AssertFailure);
}

TEST(check, check_false_non_string_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false, "this should throw", 123); }, ov::AssertFailure);
}

TEST(check, check_false_no_info) {
    EXPECT_THROW({ OPENVINO_ASSERT(false); }, ov::AssertFailure);
}

TEST(check, check_with_explanation) {
    bool check_failure_thrown = false;

    try {
        OPENVINO_ASSERT(false, "xyzzyxyzzy", 123);
    } catch (const ov::AssertFailure& e) {
        check_failure_thrown = true;
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "Check 'false' failed at", e.what());
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "xyzzyxyzzy123", e.what());
    }

    EXPECT_TRUE(check_failure_thrown);
}

TEST(check, ov_throw_exception_check_relative_path_to_source) {
    // github actions use sccache which doesn't support /d1trimfile compile option
    if (std::getenv("GITHUB_ACTIONS")) {
        GTEST_SKIP();
    }
    using namespace testing;
    const auto path = ov::util::path_join({"src", "core", "tests", "check.cpp"}).string();
    const auto exp_native_slash = "Exception from " + path + ":";
    const auto exp_fwd_slash = "Exception from src/core/tests/check.cpp:";
    OV_EXPECT_THROW(OPENVINO_THROW("Test message"),
                    ov::Exception,
                    AnyOf(StartsWith(exp_native_slash), StartsWith(exp_fwd_slash)));
}
