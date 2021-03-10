//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
