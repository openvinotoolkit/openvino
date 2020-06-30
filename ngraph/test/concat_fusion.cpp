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

#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pass/concat_fusion.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(concat_fusion, single_branch)
{
    Shape shape_a{12, 8, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto concat_1 = make_shared<op::Concat>(NodeVector{A}, 2);
        auto concat_2 = make_shared<op::Concat>(NodeVector{concat_1}, 2);
        auto concat_3 = make_shared<op::Concat>(
            NodeVector{concat_2, concat_2, concat_2, concat_2, concat_2, concat_2, concat_2}, 2);
        auto concat_4 = make_shared<op::Concat>(
            NodeVector{concat_3, concat_3, concat_3, concat_3, concat_3, concat_3, concat_3}, 3);
        auto f_concat_1 = make_shared<Function>(NodeVector{concat_4}, ParameterVector{A});
        return f_concat_1;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
    size_t num_reshapes_optimized = count_ops_of_type<op::Reshape>(optimized_f);
    size_t num_broadcast_optimzed = count_ops_of_type<op::Broadcast>(optimized_f);

    ASSERT_EQ(num_reshapes_optimized, 1);
    ASSERT_EQ(num_broadcast_optimzed, 1);
}

TEST(concat_fusion, multiple_branches_1)
{
    Shape shape_a{16, 8, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto concat_1 = make_shared<op::Concat>(NodeVector{A}, 2);
        auto concat_2 = make_shared<op::Concat>(NodeVector{concat_1}, 2);
        auto concat_3 = make_shared<op::Concat>(
            NodeVector{concat_2, concat_2, concat_2, concat_2, concat_2, concat_2, concat_2}, 2);
        auto concat_4 = make_shared<op::Concat>(
            NodeVector{concat_3, concat_3, concat_3, concat_3, concat_3, concat_3, concat_3}, 3);

        auto concat_5 = make_shared<op::Concat>(NodeVector{A, A}, 2);
        auto concat_6 = make_shared<op::Concat>(NodeVector{concat_5, concat_5, concat_5}, 3);
        auto f_concat_1 = make_shared<Function>(NodeVector{concat_4, concat_6}, ParameterVector{A});
        return f_concat_1;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

    size_t num_reshapes_optimized = count_ops_of_type<op::Reshape>(optimized_f);
    size_t num_broadcast_optimzed = count_ops_of_type<op::Broadcast>(optimized_f);

    ASSERT_EQ(num_reshapes_optimized, 2);
    ASSERT_EQ(num_broadcast_optimzed, 2);
}

TEST(concat_fusion, multiple_branches_2)
{
    Shape shape_a{16, 8, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto concat_3 = make_shared<op::Concat>(NodeVector{A, A, A, A, A, A, A}, 2);
        auto concat_4 = make_shared<op::Concat>(
            NodeVector{concat_3, concat_3, concat_3, concat_3, concat_3, concat_3, concat_3}, 3);

        auto concat_6 = make_shared<op::Concat>(NodeVector{A, A, A}, 3);
        auto f_concat_1 = make_shared<Function>(NodeVector{concat_4, concat_6}, ParameterVector{A});
        return f_concat_1;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(baseline_input_shape));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

    size_t num_reshapes_optimized = count_ops_of_type<op::Reshape>(optimized_f);
    size_t num_broadcast_optimized = count_ops_of_type<op::Broadcast>(optimized_f);

    ASSERT_EQ(num_reshapes_optimized, 2);
    ASSERT_EQ(num_broadcast_optimized, 2);
}

TEST(concat_fusion, non_fusable_self_concat)
{
    Shape shape_a{32, 1, 1, 1};
    Shape shape_b{32, 1, 1};
    auto generate_func = [shape_a, shape_b]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);

        auto concat_1 = make_shared<op::Concat>(NodeVector{A, A, A, A}, 1);
        auto concat_2 = make_shared<op::Concat>(
            NodeVector{concat_1, concat_1, concat_1, concat_1, concat_1, concat_1, concat_1}, 2);
        auto concat_3 = make_shared<op::Concat>(NodeVector{concat_2, concat_2}, 1);
        auto concat_4 = make_shared<op::Concat>(NodeVector{concat_3, concat_3, concat_3}, 3);

        auto concat_5 = make_shared<op::Concat>(NodeVector{B, B, B, B, B, B, B}, 1);
        auto concat_6 = make_shared<op::Concat>(NodeVector{concat_5, concat_5, concat_5}, 2);
        auto broadcast = make_shared<op::Broadcast>(concat_6, Shape{32, 8, 7, 3}, AxisSet{1});
        auto add = make_shared<op::v1::Add>(concat_4, broadcast);
        auto f_concat_1 = make_shared<Function>(NodeVector{add}, ParameterVector{A, B});
        return f_concat_1;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape_1 = baseline_f->get_parameters().at(0)->get_shape();
    auto baseline_input_shape_2 = baseline_f->get_parameters().at(1)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val_1(shape_size(baseline_input_shape_1));
    vector<float> tensor_val_2(shape_size(baseline_input_shape_2));
    rng.initialize(tensor_val_1);
    rng.initialize(tensor_val_2);
    args.push_back(tensor_val_1);
    args.push_back(tensor_val_2);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

    size_t num_reshapes_optimized = count_ops_of_type<op::Reshape>(optimized_f);
    size_t num_broadcast_optimzed = count_ops_of_type<op::Broadcast>(optimized_f);

    ASSERT_EQ(num_reshapes_optimized, 3);
    ASSERT_EQ(num_broadcast_optimzed, 4);
}

TEST(concat_fusion, self_concat_with_fan_out)
{
    Shape shape_a{8, 1, 1, 1};
    Shape shape_b{8, 4, 1, 1};
    auto generate_func = [shape_a, shape_b]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        auto B = make_shared<op::Parameter>(element::f32, shape_b);

        auto concat_1 = make_shared<op::Concat>(NodeVector{A, A, A, A, A, A, A}, 2);
        auto concat_2 =
            make_shared<op::Concat>(NodeVector{concat_1, concat_1, concat_1, concat_1}, 1);
        auto concat_3 =
            make_shared<op::Concat>(NodeVector{concat_2, concat_2, concat_2, concat_2}, 3);

        auto concat_4 = make_shared<op::Concat>(NodeVector{B, B, B, B, B, B, B}, 2);
        auto concat_5 = make_shared<op::Concat>(NodeVector{concat_4, concat_4, concat_4}, 3);
        auto concat_6 = make_shared<op::Concat>(NodeVector{concat_2, concat_4}, 3);
        auto f_concat_1 =
            make_shared<Function>(NodeVector{concat_3, concat_6}, ParameterVector{A, B});
        return f_concat_1;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape_1 = baseline_f->get_parameters().at(0)->get_shape();
    auto baseline_input_shape_2 = baseline_f->get_parameters().at(1)->get_shape();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConcatElimination>();
    pass_manager.register_pass<pass::SelfConcatFusion>();
    pass_manager.run_passes(optimized_f);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val_1(shape_size(baseline_input_shape_1));
    vector<float> tensor_val_2(shape_size(baseline_input_shape_2));
    rng.initialize(tensor_val_1);
    rng.initialize(tensor_val_2);
    args.push_back(tensor_val_1);
    args.push_back(tensor_val_2);

    auto baseline_results = execute(baseline_f, args, "INTERPRETER");
    auto optimized_results = execute(optimized_f, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));

    size_t num_reshapes_optimized = count_ops_of_type<op::Reshape>(optimized_f);
    size_t num_broadcast_optimzed = count_ops_of_type<op::Broadcast>(optimized_f);

    ASSERT_EQ(num_reshapes_optimized, 3);
    ASSERT_EQ(num_broadcast_optimzed, 3);
}

TEST(concat_fusion, pass_property)
{
    {
        auto pass = std::make_shared<ngraph::pass::ConcatElimination>();
        ASSERT_FALSE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
        ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
    }

    {
        auto pass = std::make_shared<ngraph::pass::SelfConcatFusion>();
        ASSERT_TRUE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
        ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
    }
}
