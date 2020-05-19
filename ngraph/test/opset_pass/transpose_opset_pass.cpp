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
#include "ngraph/pass/opset0_downgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_transpose_downgrade_pass)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{4, 5, 6, 7});
    AxisVector order{2, 1, 3, 0};
    const auto order_node = op::Constant::create(element::i64, Shape{order.size()}, order);

    auto transpose = make_shared<op::v1::Transpose>(data, order_node);
    auto result = make_shared<op::Result>(transpose);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto reshape_result = f->get_results().at(0);
    auto reshape_node = as_type_ptr<op::v0::Reshape>(reshape_result->get_input_node_shared_ptr(0));

    ASSERT_TRUE(reshape_node);
    EXPECT_EQ(reshape_node->get_input_order(), order);
    EXPECT_EQ(reshape_node->get_output_shape(0), Shape({6, 5, 7, 4}));
}

TEST(opset_transform, opset1_transpose_downgrade_pass_data_shape_not_staic)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    AxisVector order{2, 1, 3, 0};
    const auto order_node = op::Constant::create(element::i64, Shape{order.size()}, order);

    auto transpose = make_shared<op::v1::Transpose>(data, order_node);
    auto result = make_shared<op::Result>(transpose);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();

    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Transpose Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Unable to convert Transpose:v1 to Reshape:v0 "
                                         "if data shape is dynamic. Node:"));
    }
    catch (...)
    {
        FAIL() << "Transpose pass failed for unexpected reason";
    }
}

TEST(opset_transform, opset1_transpose_downgrade_pass_order_not_constant)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{4, 5, 6, 7});
    const auto order_node = make_shared<op::Parameter>(element::i64, Shape{4});

    auto transpose = make_shared<op::v1::Transpose>(data, order_node);
    auto result = make_shared<op::Result>(transpose);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, order_node});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();

    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Transpose Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Unable to convert Transpose:v1 to Reshape:v0 "
                                         "if order node is not constant. Node:"));
    }
    catch (...)
    {
        FAIL() << "Transpose pass failed for unexpected reason";
    }
}
