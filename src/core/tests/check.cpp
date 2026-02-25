// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/file_utils.hpp"
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

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
TEST(check, error_message_with_fs_path_and_unicode) {
    const auto path = ov::test::utils::to_fs_path("这是.folder") / ov::test::utils::to_fs_path(L"这.txt");
    auto description = std::string("Error detail");
    const auto exp_error_str = std::string("Test read file: \"这是.folder") +
                               ov::test::utils::FileTraits<char>::file_separator +
                               std::string("这.txt\", because: Error detail");

    std::stringstream error;
    ov::write_all_to_stream(error, "Test read file: ", path, ", because: ", description);

    EXPECT_EQ(error.str(), exp_error_str);
}
#endif
