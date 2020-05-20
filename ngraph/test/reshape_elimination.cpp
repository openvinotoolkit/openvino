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
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "ngraph/util.hpp"
#include "util/all_close.hpp"
#include "util/matcher.hpp"
#include "util/random.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

#ifndef NGRAPH_JSON_DISABLE
TEST(reshape_elimination, remove_reshape)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(reshape_elimination, remove_tranpose)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/tranpose.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}

TEST(reshape_elimination, bn_bprop_rewrite)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_bprop.json");
    const string json_string = file_util::read_file_to_string(json_path);
    stringstream ss(json_string);
    shared_ptr<Function> func = ngraph::deserialize(ss);
    size_t count_before = count_ops_of_type<op::Reshape>(func);
    pass_manager.run_passes(func);
    size_t count_after = count_ops_of_type<op::Reshape>(func);
    ASSERT_TRUE(count_after < count_before);
}
#endif

#ifdef NGRAPH_INTERPRETER_ENABLE
TEST(reshape_elimination, transpose_reshape_pattern_fuse)
{
    auto generate_func = []() {
        auto input = make_shared<op::Parameter>(element::f32, Shape{8, 2, 4, 6});
        auto transpose = make_shared<op::Reshape>(input, AxisVector{0, 2, 1, 3}, Shape{8, 2, 4, 6});
        auto reshape =
            make_shared<op::Reshape>(transpose, AxisVector{0, 1, 2, 3}, Shape{8, 4, 2, 6});
        return make_shared<Function>(reshape, ParameterVector{input});
    };

    auto fuse_func = generate_func();
    auto nofuse_func = generate_func();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.run_passes(fuse_func);
    ASSERT_TRUE(count_ops_of_type<op::Reshape>(fuse_func) == 1);
    ASSERT_TRUE(count_ops_of_type<op::Reshape>(nofuse_func) == 2);

    test::Uniform<float> rng(0.0f, 100.0f);
    vector<vector<float>> args;
    vector<float> tensor_val(shape_size(Shape{8, 2, 4, 6}));
    rng.initialize(tensor_val);
    args.push_back(tensor_val);

    auto baseline_results = execute(fuse_func, args, "INTERPRETER");
    auto optimized_results = execute(nofuse_func, args, "INTERPRETER");

    EXPECT_TRUE(test::all_close(baseline_results.at(0), optimized_results.at(0)));
}
#endif

TEST(reshape_elimination, transpose_reshape_pattern_nofuse)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{8, 2, 4, 6});
    auto transpose = make_shared<op::Reshape>(input, AxisVector{0, 2, 1, 3}, Shape{8, 2, 4, 6});
    auto reshape = make_shared<op::Reshape>(transpose, AxisVector{2, 1, 0, 3}, Shape{8, 4, 2, 6});
    auto f = make_shared<Function>(reshape, ParameterVector{input});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    pass_manager.run_passes(f);
    ASSERT_TRUE(count_ops_of_type<op::Reshape>(f) == 2);
}

TEST(reshape_elimination, dot_transpose_to_dot_w_transpose_args)
{
    Shape shape_w{2, 4};
    Shape shape_x{4, 1};
    auto W = make_shared<op::Parameter>(element::f32, shape_w);
    auto x = make_shared<op::Parameter>(element::f32, shape_x);

    auto dot = make_shared<op::Dot>(W, x);
    auto reshape_dot = std::make_shared<op::Reshape>(dot, AxisVector{1, 0}, Shape{1, 2});
    auto graph = make_shared<op::Abs>(reshape_dot);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ReshapeElimination>();
    auto func = make_shared<Function>(graph, ParameterVector{W, x});
    pass_manager.run_passes(func);
    auto gdot = graph->get_argument(0);
    ASSERT_TRUE(as_type_ptr<op::Dot>(gdot));
    ASSERT_TRUE(as_type_ptr<op::Reshape>(gdot->get_argument(0)));
    ASSERT_TRUE(as_type_ptr<op::Reshape>(gdot->get_argument(1)));
    ASSERT_EQ(gdot->get_argument(0)->get_argument(0), x);
    ASSERT_EQ(gdot->get_argument(1)->get_argument(0), W);
    ASSERT_EQ(gdot->get_shape(), (Shape{1, 2}));
}

#ifdef NGRAPH_INTERPRETER_ENABLE
TEST(reshape_elimination, recurrent_reshapes)
{
    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_r_1{3, 2, 2, 4, 6};
        Shape shape_r_2{6, 8, 3, 2};
        Shape shape_r_3{6, 8, 6};
        Shape shape_r_4{6, 2, 2, 2, 6};
        Shape shape_r_5{2, 3, 2, 2, 2, 3, 2};
        Shape shape_r_6{48, 6};

        auto r_1 = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r_1);
        auto r_2 = make_shared<op::Reshape>(r_1, AxisVector{0, 1, 2, 3, 4}, shape_r_2);
        auto r_3 = make_shared<op::Reshape>(r_2, AxisVector{0, 1, 2, 3}, shape_r_3);
        auto r_4 = make_shared<op::Reshape>(r_3, AxisVector{0, 1, 2}, shape_r_4);
        auto r_5 = make_shared<op::Reshape>(r_4, AxisVector{0, 1, 2, 3, 4}, shape_r_5);
        auto r_6 = make_shared<op::Reshape>(r_5, AxisVector{0, 1, 2, 3, 4, 5, 6}, shape_r_6);

        auto f = make_shared<Function>(r_6, ParameterVector{A});
        return f;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_recurrent_reshapes.png");
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
    ASSERT_EQ(num_reshapes_optimized, 1);
}

TEST(reshape_elimination, recurrent_reshapes_elimination)
{
    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_r_1{3, 2, 2, 4, 6};
        Shape shape_r_2{6, 8, 3, 2};
        Shape shape_r_3{6, 8, 6};
        Shape shape_r_4{6, 2, 2, 2, 6};
        Shape shape_r_5{2, 3, 2, 2, 2, 3, 2};
        Shape shape_r_6{48, 6};
        Shape shape_r_7{2, 2, 3, 3, 2, 4};

        auto r_1 = make_shared<op::Reshape>(A, AxisVector{0, 1, 2, 3, 4, 5}, shape_r_1);
        auto r_2 = make_shared<op::Reshape>(r_1, AxisVector{0, 1, 2, 3, 4}, shape_r_2);
        auto r_3 = make_shared<op::Reshape>(r_2, AxisVector{0, 1, 2, 3}, shape_r_3);
        auto r_4 = make_shared<op::Reshape>(r_3, AxisVector{0, 1, 2}, shape_r_4);
        auto r_5 = make_shared<op::Reshape>(r_4, AxisVector{0, 1, 2, 3, 4}, shape_r_5);
        auto r_6 = make_shared<op::Reshape>(r_5, AxisVector{0, 1, 2, 3, 4, 5, 6}, shape_r_6);
        auto r_7 = make_shared<op::Reshape>(r_6, AxisVector{0, 1}, shape_r_7);
        auto f = make_shared<Function>(r_7, ParameterVector{A});
        return f;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes_elimination.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_1_recurrent_reshapes_elimination.png");
    pass_manager.register_pass<pass::ReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_2_recurrent_reshapes_elimination.png");
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
    ASSERT_EQ(num_reshapes_optimized, 0);
}

TEST(reshape_elimination, recurrent_reshapes_fan_out)
{
    Shape shape_a{4, 6, 10, 2};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_r_1{6, 4, 5, 4};
        Shape shape_r_2{24, 20};
        auto reshape_1 = make_shared<op::Reshape>(A, AxisVector{0, 3, 2, 1}, shape_r_1);
        auto reshape_2 = make_shared<op::Reshape>(reshape_1, AxisVector{0, 1, 2, 3}, shape_r_2);
        auto reshape_3 = make_shared<op::Reshape>(reshape_2, AxisVector{0, 1}, shape_a);
        auto f_ = make_shared<Function>(NodeVector{reshape_2, reshape_3}, ParameterVector{A});
        return f_;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes_fan_out.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_recurrent_reshapes_fan_out.png");
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
    ASSERT_EQ(num_reshapes_optimized, 2);
}

TEST(reshape_elimination, recurrent_reshapes_fan_out_at_end)
{
    Shape shape_a{12, 8, 1, 1};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto reshape_1 = make_shared<op::Reshape>(A, AxisVector{0, 3, 2, 1}, Shape{4, 3, 8, 1});
        auto reshape_2 = make_shared<op::Reshape>(reshape_1, AxisVector{0, 1, 2, 3}, shape_a);
        auto reshape_3 =
            make_shared<op::Reshape>(reshape_2, AxisVector{0, 1, 2, 3}, Shape{4, 3, 8, 1});
        auto abs_1 = make_shared<op::Abs>(reshape_3);
        auto f_ = make_shared<Function>(NodeVector{abs_1, reshape_3}, ParameterVector{A});
        return f_;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>("before_recurrent_reshapes_fan_out_at_end.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_recurrent_reshapes_fan_out_at_end.png");
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
    ASSERT_EQ(num_reshapes_optimized, 1);
}

TEST(reshape_elimination, recurrent_reshapes_multiple_fusions)
{
    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_r_1{3, 2, 2, 4, 6};
        Shape shape_r_2{6, 8, 3, 2};
        Shape shape_r_3{6, 8, 6};
        Shape shape_r_4{6, 2, 2, 2, 6};
        Shape shape_r_5{2, 3, 2, 2, 2, 3, 2};
        Shape shape_r_6{48, 6};

        auto r_1 = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r_1);
        auto r_2 = make_shared<op::Reshape>(r_1, AxisVector{0, 1, 2, 3, 4}, shape_r_2);
        auto r_3 = make_shared<op::Reshape>(r_2, AxisVector{0, 1, 2, 3}, shape_r_3);
        auto r_4 = make_shared<op::Reshape>(r_3, AxisVector{1, 0, 2}, shape_r_4);
        auto r_5 = make_shared<op::Reshape>(r_4, AxisVector{0, 1, 2, 3, 4}, shape_r_5);
        auto r_6 = make_shared<op::Reshape>(r_5, AxisVector{0, 1, 2, 3, 4, 5, 6}, shape_r_6);

        auto f = make_shared<Function>(r_6, ParameterVector{A});
        return f;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>(
    //     "before_recurrent_reshapes_multiple_fusions.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>(
    //     "after_recurrent_reshapes_multiple_fusions.png");
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
    ASSERT_EQ(num_reshapes_optimized, 2);
}

TEST(reshape_elimination, nonrecurrent_reshapes)
{
    Shape shape_a{8, 6, 1, 1};
    Shape shape_r{2, 24};
    auto generate_func = [shape_a, shape_r]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);

        auto reshape_1 = make_shared<op::Reshape>(A, AxisVector{3, 0, 2, 1}, shape_r);
        auto abs_1 = make_shared<op::Abs>(reshape_1);
        auto reshape_2 = make_shared<op::Reshape>(abs_1, AxisVector{0, 1}, shape_a);
        auto abs_2 = make_shared<op::Abs>(reshape_2);
        auto reshape_3 = make_shared<op::Reshape>(abs_2, AxisVector{0, 1, 2, 3}, shape_a);
        auto f_ = make_shared<Function>(NodeVector{reshape_3}, ParameterVector{A});
        return f_;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>("before_nonrecurrent_reshapes.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>("after_nonrecurrent_reshapes.png");
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
    ASSERT_EQ(num_reshapes_optimized, 3);
}

TEST(reshape_elimination, recurrent_reshapes_multiple_branches)
{
    Shape shape_a{2, 2, 3, 3, 2, 4};
    auto generate_func = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape shape_r_1{3, 2, 2, 4, 6};
        Shape shape_r_2{6, 8, 3, 2};
        Shape shape_r_3{6, 8, 6};
        Shape shape_r_4{6, 2, 2, 2, 6};
        Shape shape_r_5{2, 3, 2, 2, 2, 3, 2};
        Shape shape_r_6{48, 6};

        auto r_1 = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r_1);
        auto r_2 = make_shared<op::Reshape>(r_1, AxisVector{0, 1, 2, 3, 4}, shape_r_2);
        auto r_3 = make_shared<op::Reshape>(r_2, AxisVector{0, 1, 2, 3}, shape_r_3);
        auto r_4 = make_shared<op::Reshape>(r_3, AxisVector{0, 1, 2}, shape_r_4);
        auto r_5 = make_shared<op::Reshape>(r_4, AxisVector{0, 1, 2, 3, 4}, shape_r_5);
        auto r_6 = make_shared<op::Reshape>(r_5, AxisVector{0, 1, 2, 3, 4, 5, 6}, shape_r_6);

        auto r_7 = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r_2);
        auto r_8 = make_shared<op::Reshape>(r_7, AxisVector{0, 1, 2, 3}, shape_r_3);

        auto f = make_shared<Function>(NodeVector{r_6, r_8}, ParameterVector{A});
        return f;
    };

    auto baseline_f = generate_func();
    auto optimized_f = generate_func();
    auto baseline_input_shape = baseline_f->get_parameters().at(0)->get_shape();

    pass::Manager pass_manager;
    // pass_manager.register_pass<pass::VisualizeTree>(
    //     "before_recurrent_reshapes_multiple_branches.png");
    pass_manager.register_pass<pass::RecurrentReshapeElimination>();
    // pass_manager.register_pass<pass::VisualizeTree>(
    //     "after_recurrent_reshapes_multiple_branches.png");
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
    ASSERT_EQ(num_reshapes_optimized, 2);
}
#endif

TEST(reshape_elimination, pass_property)
{
    {
        auto pass = std::make_shared<ngraph::pass::ReshapeElimination>();
        ASSERT_FALSE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
        ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
    }
    {
        auto pass = std::make_shared<ngraph::pass::RecurrentReshapeElimination>();
        ASSERT_FALSE(pass->get_property(pass::PassProperty::REQUIRE_STATIC_SHAPE));
        ASSERT_FALSE(pass->get_property(pass::PassProperty::CHANGE_DYNAMIC_STATE));
    }
}
