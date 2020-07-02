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

TEST(opset_transform, opset1_reverse_upgrade_pass)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    const AxisSet reverse_axes{1, 2};

    const auto reverse_v0 = make_shared<op::v0::Reverse>(data, reverse_axes);
    const auto result = make_shared<op::Result>(reverse_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto reverse_v1 = as_type_ptr<op::v1::Reverse>(pass_replacement_node);
    ASSERT_TRUE(reverse_v1);
    EXPECT_EQ(reverse_v1->get_mode(), op::v1::Reverse::Mode::INDEX);

    const auto& rev_axes_input_shape = reverse_v1->get_input_shape(1);
    // should match the number of elements of v0::Reverse reverse_axes attribute
    EXPECT_EQ(rev_axes_input_shape, Shape{2});
}

TEST(opset_transform, opset0_reverse_downgrade_pass_index_mode)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    const auto reverse_axes =
        make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto mode = op::v1::Reverse::Mode::INDEX;

    const auto reverse_v1 = make_shared<op::v1::Reverse>(data, reverse_axes, mode);
    const auto result = make_shared<op::Result>(reverse_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto reverse_v0 = as_type_ptr<op::v0::Reverse>(pass_replacement_node);
    ASSERT_TRUE(reverse_v0);
    EXPECT_EQ(reverse_v0->get_reversed_axes(), AxisSet({1, 2}));
}

TEST(opset_transform, opset0_reverse_downgrade_pass_mask_mode)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    const auto reverse_axes =
        make_shared<op::Constant>(element::boolean, Shape{3}, vector<bool>{true, false, true});
    auto mode = op::v1::Reverse::Mode::MASK;

    const auto reverse_v1 = make_shared<op::v1::Reverse>(data, reverse_axes, mode);
    const auto result = make_shared<op::Result>(reverse_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto reverse_v0 = as_type_ptr<op::v0::Reverse>(pass_replacement_node);
    ASSERT_TRUE(reverse_v0);
    EXPECT_EQ(reverse_v0->get_reversed_axes(), AxisSet({0, 2}));
}

TEST(opset_transform, opset0_reverse_downgrade_pass_axes_not_constant)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 2});
    const auto axes = make_shared<op::Parameter>(element::boolean, Shape{3});

    const auto reverse_v1 = make_shared<op::v1::Reverse>(data, axes, op::v1::Reverse::Mode::MASK);
    const auto result = make_shared<op::Result>(reverse_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, axes});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Unable to convert Reverse:v1 to Reverse:v0"));
    }
    catch (...)
    {
        FAIL() << "Reverse:v1 pass failed for unexpected reason";
    }
}
