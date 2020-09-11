//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
