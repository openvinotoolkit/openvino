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
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_dyn_slice_upgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{7, 4, 6, 8});
    Coordinate lower_bounds{2, 1, 4, 0};
    Coordinate upper_bounds{4, 3, 5, 1};
    Strides strides{1, 2, 1, 2};

    auto slice_v0 = make_shared<op::v0::Slice>(arg, lower_bounds, upper_bounds, strides);

    const auto result = make_shared<op::Result>(slice_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto strided_slice_v1 = as_type_ptr<op::v1::StridedSlice>(pass_replacement_node);
    ASSERT_TRUE(strided_slice_v1);
    auto begin_const =
        as_type_ptr<op::Constant>(strided_slice_v1->input_value(1).get_node_shared_ptr());
    ASSERT_TRUE(begin_const);
    auto end_const =
        as_type_ptr<op::Constant>(strided_slice_v1->input_value(2).get_node_shared_ptr());
    ASSERT_TRUE(end_const);
    auto strides_const =
        as_type_ptr<op::Constant>(strided_slice_v1->input_value(3).get_node_shared_ptr());
    ASSERT_TRUE(strides_const);

    EXPECT_EQ(strided_slice_v1->get_begin_mask(), vector<int64_t>(4, 0));
    EXPECT_EQ(strided_slice_v1->get_end_mask(), vector<int64_t>(4, 0));
    EXPECT_EQ(begin_const->get_vector<int64_t>(),
              vector<int64_t>(lower_bounds.begin(), lower_bounds.end()));
    EXPECT_EQ(end_const->get_vector<int64_t>(),
              vector<int64_t>(upper_bounds.begin(), upper_bounds.end()));
    EXPECT_EQ(strides_const->get_vector<int64_t>(),
              vector<int64_t>(strides.begin(), strides.end()));
}

TEST(opset_transform, opset1_strided_slice_downgrade_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{5, 7, 6, 8});
    auto begin = op::Constant::create(element::i64, Shape{4}, {1, 2, 1, 2});
    auto end = op::Constant::create(element::i64, Shape{4}, {3, 4, 5, 6});

    auto strided_slice_v1 = make_shared<op::v1::StridedSlice>(
        data, begin, end, vector<int64_t>{0, 0, 1, 0}, vector<int64_t>{1, 0, 0, 0});

    const auto result = make_shared<op::Result>(strided_slice_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto slice_v0 = as_type_ptr<op::v0::Slice>(pass_replacement_node);
    ASSERT_TRUE(slice_v0);
    EXPECT_EQ(slice_v0->get_lower_bounds(), Coordinate({1, 2, 0, 2}));
    EXPECT_EQ(slice_v0->get_upper_bounds(), Coordinate({5, 4, 5, 6}));
    EXPECT_EQ(slice_v0->get_strides(), Strides({1, 1, 1, 1}));
}

TEST(opset_transform, opset1_strided_slice_downgrade_pass_dynamic_input_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto begin = op::Constant::create(element::i64, Shape{4}, {1, 2, 1, 2});
    auto end = op::Constant::create(element::i64, Shape{4}, {3, 4, 5, 6});

    auto strided_slice_v1 = make_shared<op::v1::StridedSlice>(
        data, begin, end, vector<int64_t>{0, 0, 1, 0}, vector<int64_t>{1, 0, 0, 0});

    const auto result = make_shared<op::Result>(strided_slice_v1);
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
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string(
                "Unable to convert StridedSlice:v1 to Slice:v0 if input rank is not static."));
    }
    catch (...)
    {
        FAIL() << "StridedSlice pass failed for unexpected reason";
    }
}

TEST(opset_transform, opset1_strided_slice_downgrade_pass_end_not_constant)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{5, 7, 6, 8});
    auto begin = op::Constant::create(element::i64, Shape{4}, {1, 2, 1, 2});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});

    auto strided_slice_v1 = make_shared<op::v1::StridedSlice>(
        data, begin, end, vector<int64_t>{0, 0, 1, 0}, vector<int64_t>{1, 0, 0, 0});

    const auto result = make_shared<op::Result>(strided_slice_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, end});

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
                             std::string("Unable to convert StridedSlice:v1 to Slice:v0 "
                                         "if begin, end or strides are not constant"));
    }
    catch (...)
    {
        FAIL() << "StridedSlice pass failed for unexpected reason";
    }
}
