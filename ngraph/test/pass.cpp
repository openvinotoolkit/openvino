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

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/constant_to_broadcast.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(pass, constant_to_broadcast)
{
    Shape shape{128, 256, 1, 1};
    vector<float> v = {3};
    auto c = make_shared<op::Constant>(element::f32, shape, v);
    auto f = make_shared<Function>(c, ParameterVector{});

    {
        ngraph::pass::Manager pm;
        pm.register_pass<pass::ConstantToBroadcast>();
        EXPECT_EQ(count_ops_of_type<op::Broadcast>(f), 0);
        pm.run_passes(f);
        EXPECT_EQ(count_ops_of_type<op::Broadcast>(f), 1);
    }
}
