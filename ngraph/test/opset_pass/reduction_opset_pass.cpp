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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "opset0_downgrade.hpp"
#include "opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

//------------------------------------------------------------------------------
//
//                  Helper Functions
//
//------------------------------------------------------------------------------

template <typename OpV0, typename OpV1>
void test_reduce_op_opset1_upgrade_pass()
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const AxisSet reduction_axes{1, 2};

    const auto v0_node = make_shared<OpV0>(data, reduction_axes);
    const auto result = make_shared<op::Result>(v0_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    const auto v1_node = as_type_ptr<OpV1>(pass_replacement_node);

    ASSERT_TRUE(v1_node);
    EXPECT_EQ(v1_node->get_keep_dims(), false);
    EXPECT_EQ(v1_node->get_output_element_type(0), element::f32);
    EXPECT_EQ(v1_node->get_output_shape(0), (Shape{1}));
}

template <typename OpV0, typename OpV1>
void test_reduce_op_opset0_downgrade_pass()
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});

    const auto v1_node = make_shared<OpV1>(data, axes, true);
    const auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto reshape_replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    const auto reshape_node = as_type_ptr<op::Reshape>(reshape_replacement_node);
    ASSERT_TRUE(reshape_node);
    EXPECT_EQ(reshape_node->get_output_element_type(0), element::f32);
    EXPECT_EQ(reshape_node->get_output_shape(0), (Shape{1, 1, 3}));

    const auto op_replace_node = reshape_replacement_node->input_value(0).get_node_shared_ptr();
    const auto v0_node = as_type_ptr<OpV0>(op_replace_node);
    ASSERT_TRUE(v0_node);
}

template <typename OpV1>
void test_reduce_op_opset0_downgrade_pass_axes_not_constant()
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes = make_shared<op::Parameter>(element::f32, Shape{1});

    const auto v1_node = make_shared<OpV1>(data, axes, true);
    const auto result = make_shared<op::Result>(v1_node);
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
                             string("reduction axes are not constant (for keep_dims=true)"));
    }
    catch (...)
    {
        FAIL() << "ReduceSum pass failed for unexpected reason";
    }
}

template <typename OpV1>
void test_reduce_op_opset0_downgrade_pass_output_not_static()
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{0, 1});

    const auto v1_node = make_shared<OpV1>(data, axes, true);
    const auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    try
    {
        pass_manager.run_passes(f);
        FAIL() << "Exception after Opset0Downgrade pass was not thrown.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), string("output shape is dynamic (for keep_dims=true)"));
    }
    catch (...)
    {
        FAIL() << "ReduceSum pass failed for unexpected reason";
    }
}

template <typename OpV1>
void test_reduce_op_opset0_downgrade_pass_out_shape_if_keep_dims()
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = true;
    auto v1_node = make_shared<OpV1>(arg, axes, keep_dims);
    const auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(replacement_node->get_output_partial_shape(0).compatible(PartialShape{3, 1, 1}));
}

template <typename OpV1>
void test_reduce_op_opset0_downgrade_pass_out_shape_if_not_keep_dims()
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto keep_dims = false;
    auto v1_node = make_shared<OpV1>(arg, axes, keep_dims);
    const auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(replacement_node->get_output_partial_shape(0).compatible(PartialShape{3}));
}

//------------------------------------------------------------------------------
//
//                  Test Cases
//
//------------------------------------------------------------------------------

TEST(opset_transform, opset1_reduce_sum_upgrade_pass)
{
    test_reduce_op_opset1_upgrade_pass<op::Sum, op::v1::ReduceSum>();
}

TEST(opset_transform, opset0_reduce_sum_downgrade_pass)
{
    test_reduce_op_opset0_downgrade_pass<op::v0::Sum, op::v1::ReduceSum>();
}

TEST(opset_transform, opset0_reduce_sum_downgrade_pass_axes_not_constant_axes)
{
    test_reduce_op_opset0_downgrade_pass_axes_not_constant<op::v1::ReduceSum>();
}

TEST(opset_transform, opset0_reduce_sum_downgrade_pass_output_not_static)
{
    test_reduce_op_opset0_downgrade_pass_output_not_static<op::v1::ReduceSum>();
}

TEST(opset_transform, opset0_reduce_sum_downgrade_pass_out_shape_if_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_keep_dims<op::v1::ReduceSum>();
}

TEST(opset_transform, opset0_reduce_sum_downgrade_pass_out_shape_if_not_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_not_keep_dims<op::v1::ReduceSum>();
}

TEST(opset_transform, opset1_reduce_prod_upgrade_pass)
{
    test_reduce_op_opset1_upgrade_pass<op::Product, op::v1::ReduceProd>();
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass)
{
    test_reduce_op_opset0_downgrade_pass<op::v0::Product, op::v1::ReduceProd>();
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_axes_not_constant_axes)
{
    test_reduce_op_opset0_downgrade_pass_axes_not_constant<op::v1::ReduceProd>();
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_output_not_static)
{
    test_reduce_op_opset0_downgrade_pass_output_not_static<op::v1::ReduceProd>();
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_out_shape_if_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_keep_dims<op::v1::ReduceProd>();
}

TEST(opset_transform, opset0_reduce_prod_downgrade_pass_out_shape_if_not_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_not_keep_dims<op::v1::ReduceProd>();
}

TEST(opset_transform, opset1_reduce_max_upgrade_pass)
{
    test_reduce_op_opset1_upgrade_pass<op::Max, op::v1::ReduceMax>();
}

TEST(opset_transform, opset0_reduce_max_downgrade_pass)
{
    test_reduce_op_opset0_downgrade_pass<op::v0::Max, op::v1::ReduceMax>();
}

TEST(opset_transform, opset0_reduce_max_downgrade_pass_axes_not_constant_axes)
{
    test_reduce_op_opset0_downgrade_pass_axes_not_constant<op::v1::ReduceMax>();
}

TEST(opset_transform, opset0_reduce_max_downgrade_pass_output_not_static)
{
    test_reduce_op_opset0_downgrade_pass_output_not_static<op::v1::ReduceMax>();
}

TEST(opset_transform, opset0_reduce_max_downgrade_pass_out_shape_if_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_keep_dims<op::v1::ReduceMax>();
}

TEST(opset_transform, opset0_reduce_max_downgrade_pass_out_shape_if_not_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_not_keep_dims<op::v1::ReduceMax>();
}

TEST(opset_transform, opset1_reduce_min_upgrade_pass)
{
    test_reduce_op_opset1_upgrade_pass<op::Min, op::v1::ReduceMin>();
}

TEST(opset_transform, opset0_reduce_min_downgrade_pass)
{
    test_reduce_op_opset0_downgrade_pass<op::v0::Min, op::v1::ReduceMin>();
}

TEST(opset_transform, opset0_reduce_min_downgrade_pass_axes_not_constant_axes)
{
    test_reduce_op_opset0_downgrade_pass_axes_not_constant<op::v1::ReduceMin>();
}

TEST(opset_transform, opset0_reduce_min_downgrade_pass_output_not_static)
{
    test_reduce_op_opset0_downgrade_pass_output_not_static<op::v1::ReduceMin>();
}

TEST(opset_transform, opset0_reduce_min_downgrade_pass_out_shape_if_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_keep_dims<op::v1::ReduceMin>();
}

TEST(opset_transform, opset0_reduce_min_downgrade_pass_out_shape_if_not_keep_dims)
{
    test_reduce_op_opset0_downgrade_pass_out_shape_if_not_keep_dims<op::v1::ReduceMin>();
}
