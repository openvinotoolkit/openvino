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
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/dyn_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "opset0_downgrade.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(dyn_elimination, transpose)
{
    Shape shape_in{2, 4, 6, 8};
    auto param = make_shared<op::Parameter>(element::boolean, shape_in);

    auto constant_perm =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{2, 3, 1, 0});

    auto transpose = make_shared<op::Transpose>(param, constant_perm);

    auto f = make_shared<Function>(transpose, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 1);

    auto new_reshape = as_type_ptr<op::Reshape>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_reshape);

    ASSERT_EQ(new_reshape->get_input_order(), (AxisVector{2, 3, 1, 0}));
    ASSERT_EQ(new_reshape->get_output_shape(0), (Shape{6, 8, 4, 2}));
    ASSERT_EQ(new_reshape->get_output_element_type(0), element::boolean);
}

// For now, we can't handle the case where the input has dynamic shapes,
// because the classic Reshape op demands a Shape. Probably won't be able to
// deal with this until/unless we make a "StaticTranspose". Just make sure
// we don't crash or mangle the graph.
TEST(dyn_elimination, transpose_dyn_shape)
{
    PartialShape shape_in{2, 4, Dimension::dynamic(), 8};

    auto param = make_shared<op::Parameter>(element::boolean, shape_in);

    auto constant_perm =
        make_shared<op::Constant>(element::i64, Shape{4}, vector<int64_t>{2, 3, 1, 0});

    auto transpose = make_shared<op::Transpose>(param, constant_perm);

    auto f = make_shared<Function>(transpose, ParameterVector{param});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Transpose>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_transpose = as_type_ptr<op::Transpose>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_transpose);

    ASSERT_EQ(new_transpose->get_output_element_type(0), element::boolean);
    ASSERT_TRUE(new_transpose->get_output_partial_shape(0).relaxes(
        PartialShape{Dimension::dynamic(), 8, 4, 2}));
}

TEST(dyn_elimination, slice)
{
    // input has shape [2,4,6,8,2,2,2]
    // slice in numpy syntax is [0:,:4,2:6:2,7:3:-2,np.newaxis,...,1]
    // shape should be [2,4,2,2,1,2,2] (so sayeth numpy!)
    Shape shape_in{2, 4, 6, 8, 2, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, shape_in);
    auto constant_lb =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{0, 3, 2, 7, 0, 0, 1});
    auto constant_ub =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{0, 4, 6, 3, 0, 0, 0});
    auto constant_strides =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{1, 1, 2, -2, 0, 0, 0});
    AxisSet lower_bounds_mask{1};
    AxisSet upper_bounds_mask{0};
    AxisSet new_axis_mask{4};
    AxisSet shrink_mask{6};
    AxisSet ellipsis_mask{5};

    auto sl = make_shared<op::DynSlice>(input,
                                        constant_lb,
                                        constant_ub,
                                        constant_strides,
                                        lower_bounds_mask,
                                        upper_bounds_mask,
                                        new_axis_mask,
                                        shrink_mask,
                                        ellipsis_mask);

    ASSERT_EQ(sl->get_element_type(), element::f32);
    ASSERT_EQ(sl->get_shape(), (Shape{2, 4, 2, 2, 1, 2, 2}));

    auto f = make_shared<Function>(sl, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::DynSlice>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Slice>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Reverse>(f), 1);

    ASSERT_EQ(f->get_results().at(0)->get_element_type(), element::f32);
    ASSERT_EQ(f->get_results().at(0)->get_shape(), (Shape{2, 4, 2, 2, 1, 2, 2}));
}

TEST(dyn_elimination, replace_slice)
{
    // input has shape [2,4,6,8,2,2,2]
    // slice in numpy syntax is [0:,:4,2:6:2,7:3:-2,np.newaxis,...,1]
    // slice shape should be [2,4,2,2,1,2,2] (so sayeth numpy!)
    Shape shape_in{2, 4, 6, 8, 2, 2, 2};
    Shape shape_slice{2, 4, 2, 2, 1, 2, 2};
    auto input = make_shared<op::Parameter>(element::f32, shape_in);
    auto replacement = make_shared<op::Parameter>(element::f32, shape_slice);
    auto constant_lb =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{0, 3, 2, 7, 0, 0, 1});
    auto constant_ub =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{0, 4, 6, 3, 0, 0, 0});
    auto constant_strides =
        make_shared<op::Constant>(element::i64, Shape{7}, vector<int64_t>{1, 1, 2, -2, 0, 0, 0});
    AxisSet lower_bounds_mask{1};
    AxisSet upper_bounds_mask{0};
    AxisSet new_axis_mask{4};
    AxisSet shrink_mask{6};
    AxisSet ellipsis_mask{5};

    auto rsl = make_shared<op::DynReplaceSlice>(input,
                                                replacement,
                                                constant_lb,
                                                constant_ub,
                                                constant_strides,
                                                lower_bounds_mask,
                                                upper_bounds_mask,
                                                new_axis_mask,
                                                shrink_mask,
                                                ellipsis_mask);

    ASSERT_EQ(rsl->get_element_type(), element::f32);
    ASSERT_EQ(rsl->get_shape(), (Shape{2, 4, 6, 8, 2, 2, 2}));

    auto f = make_shared<Function>(rsl, ParameterVector{input, replacement});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::DynReplaceSlice>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::ReplaceSlice>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Reshape>(f), 1);
    ASSERT_EQ(count_ops_of_type<op::Reverse>(f), 1);

    ASSERT_EQ(f->get_results().at(0)->get_element_type(), element::f32);
    ASSERT_EQ(f->get_results().at(0)->get_shape(), (Shape{2, 4, 6, 8, 2, 2, 2}));
}

TEST(dyn_elimination, range)
{
    auto constant_start = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{0});
    auto constant_stop = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{5});
    auto constant_step = make_shared<op::Constant>(element::i64, Shape{}, vector<int64_t>{2});

    auto range = make_shared<op::Range>(constant_start, constant_stop, constant_step);

    ASSERT_EQ(range->get_element_type(), element::i64);
    ASSERT_EQ(range->get_shape(), (Shape{3}));

    auto f = make_shared<Function>(range, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Range>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto replacement = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));

    ASSERT_NE(replacement, nullptr);
    ASSERT_EQ(replacement->get_element_type(), element::i64);
    ASSERT_EQ(replacement->get_shape(), (Shape{3}));

    auto vals = replacement->get_vector<int64_t>();

    ASSERT_EQ(vals, (vector<int64_t>{0, 2, 4}));
}

TEST(dyn_elimination, range_f64)
{
    auto constant_start = make_shared<op::Constant>(element::f64, Shape{}, vector<double>{-0.5});
    auto constant_stop = make_shared<op::Constant>(element::f64, Shape{}, vector<double>{2});
    auto constant_step = make_shared<op::Constant>(element::f64, Shape{}, vector<double>{0.25});

    auto range = make_shared<op::Range>(constant_start, constant_stop, constant_step);

    ASSERT_EQ(range->get_element_type(), element::f64);
    ASSERT_EQ(range->get_shape(), (Shape{10}));

    auto f = make_shared<Function>(range, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::DynElimination>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Range>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto replacement = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));

    ASSERT_NE(replacement, nullptr);
    ASSERT_EQ(replacement->get_element_type(), element::f64);
    ASSERT_EQ(replacement->get_shape(), (Shape{10}));

    auto vals = replacement->get_vector<double>();

    ASSERT_TRUE(test::all_close_f(
        vals, vector<double>{-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75}));
}
