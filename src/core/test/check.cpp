// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ngraph/check.hpp"

using namespace ngraph;
using namespace std;

TEST(check, check_true_string_info)
{
    NGRAPH_CHECK(true, "this should not throw");
}

TEST(check, check_true_non_string_info)
{
    NGRAPH_CHECK(true, "this should not throw", 123);
}

TEST(check, check_true_no_info)
{
    NGRAPH_CHECK(true);
}

TEST(check, check_false_string_info)
{
    EXPECT_THROW({ NGRAPH_CHECK(false, "this should throw"); }, CheckFailure);
}

TEST(check, check_false_non_string_info)
{
    EXPECT_THROW({ NGRAPH_CHECK(false, "this should throw", 123); }, CheckFailure);
}

TEST(check, check_false_no_info)
{
    EXPECT_THROW({ NGRAPH_CHECK(false); }, CheckFailure);
}

TEST(check, check_with_explanation)
{
    bool check_failure_thrown = false;

    try
    {
        NGRAPH_CHECK(false, "xyzzyxyzzy", 123);
    }
    catch (const CheckFailure& e)
    {
        check_failure_thrown = true;
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "Check 'false' failed at", e.what());
        EXPECT_PRED_FORMAT2(testing::IsSubstring, "xyzzyxyzzy123", e.what());
    }

    EXPECT_TRUE(check_failure_thrown);
}
