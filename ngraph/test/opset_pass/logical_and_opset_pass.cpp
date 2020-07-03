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
#include "opset0_downgrade.hpp"
#include "opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_logical_and_upgrade_pass)
{
    const auto a = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto b = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto and_v0 = make_shared<op::v0::And>(a, b);
    const auto result = make_shared<op::Result>(and_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{a, b});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto and_v1 = as_type_ptr<op::v1::LogicalAnd>(pass_replacement_node);
    ASSERT_TRUE(and_v1);

    const auto values_out_element_type = and_v1->get_output_element_type(0);
    EXPECT_EQ(values_out_element_type, element::boolean);
}

TEST(opset_transform, opset1_logical_and_downgrade_pass)
{
    const auto a = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto b = make_shared<op::Parameter>(element::boolean, Shape{5, 10, 15});
    const auto and_v1 = make_shared<op::v1::LogicalAnd>(a, b);
    const auto result = make_shared<op::Result>(and_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{a, b});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto and_v0 = as_type_ptr<op::v0::And>(pass_replacement_node);
    ASSERT_TRUE(and_v0);

    const auto values_out_element_type = and_v0->get_output_element_type(0);
    EXPECT_EQ(values_out_element_type, element::boolean);
}
