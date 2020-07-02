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

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_gather_upgrade_pass)
{
    auto params = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    size_t axis = 1;

    auto gather_v0 = make_shared<op::v0::Gather>(params, indices, axis);
    auto result = make_shared<op::Result>(gather_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{params, indices});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto gather_s1_result = f->get_results().at(0);
    auto gather_v1_node =
        as_type_ptr<op::v1::Gather>(gather_s1_result->get_input_node_shared_ptr(0));
    ASSERT_TRUE(gather_v1_node);
    EXPECT_EQ(gather_v1_node->get_axis(), axis);
}
