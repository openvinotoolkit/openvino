// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(uint4, convert_u4_to_string)
{
    vector<uint8_t> values{171, 16};
    auto constant = make_shared<op::Constant>(element::u4, Shape{3}, &values[0]);

    vector<string> ref{"10", "11", "1"};
    for (size_t i = 0; i < 3; ++i)
    {
        ASSERT_EQ(constant->convert_value_to_string(i), ref[i]);
    }
}
