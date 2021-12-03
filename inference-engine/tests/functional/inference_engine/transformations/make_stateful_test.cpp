// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>

#include <transformations/init_node_info.hpp>
#include <openvino/pass/make_stateful.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace opset8;
using namespace std;

TEST(TransformationTests, make_stateful_by_tensor_name) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});

        auto add = make_shared<Add>(X, Y);
        auto squeeze = make_shared<Squeeze>(add);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(squeeze);
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        std::map<std::string, std::string> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pair_names);

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // create ReadValue for X
        auto variable_x = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "xres0"});
        auto const_zero_x = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_x = make_shared<ReadValue>(const_zero_x, variable_x);

        // create ReadValue for Y
        auto variable_y = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "yres1"});
        auto const_zero_y = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_y = make_shared<ReadValue>(const_zero_y, variable_y);

        auto add = make_shared<Add>(read_val_x, read_val_y);
        auto squeeze = make_shared<Squeeze>(add);
        auto assign_x = make_shared<Assign>(add, variable_x);
        auto assign_y = make_shared<Assign>(squeeze, variable_y);

        f_ref = make_shared<Function>(ResultVector{}, SinkVector{assign_x, assign_y}, ParameterVector{});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, make_stateful_by_param_res) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        std::vector<std::pair<std::string, std::string>> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(ov::pass::MakeStateful::ParamResPairs{{X, result0}, {Y, result1}});
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // create ReadValue for X
        auto variable_x = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "xres0"});
        auto const_zero_x = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_x = make_shared<ReadValue>(const_zero_x, variable_x);

        // create ReadValue for Y
        auto variable_y = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "yres1"});
        auto const_zero_y = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_y = make_shared<ReadValue>(const_zero_y, variable_y);

        auto add = make_shared<Add>(read_val_x, read_val_y);
        auto assign_x = make_shared<Assign>(add, variable_x);
        auto assign_y = make_shared<Assign>(add, variable_y);

        f_ref = make_shared<Function>(ResultVector{}, SinkVector{assign_x, assign_y}, ParameterVector{});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, make_stateful_dynamic_shapes) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, PartialShape::dynamic());
        auto Y = make_shared<Parameter>(element::f32, PartialShape::dynamic());
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        map<std::string, std::string> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pair_names);

        EXPECT_THROW(manager.run_passes(f), ::ov::AssertFailure);
    }
}

TEST(TransformationTests, make_stateful_one_output_to_several_results) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        std::map<std::string, std::string> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pair_names);

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // create ReadValue for X
        auto variable_x = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "xres0"});
        auto const_zero_x = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_x = make_shared<ReadValue>(const_zero_x, variable_x);

        // create ReadValue for Y
        auto variable_y = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "yres1"});
        auto const_zero_y = make_shared<Constant>(element::f32, Shape{32, 1, 10}, 0);
        auto read_val_y = make_shared<ReadValue>(const_zero_y, variable_y);

        auto add = make_shared<Add>(read_val_x, read_val_y);
        auto assign_x = make_shared<Assign>(add, variable_x);
        auto assign_y = make_shared<Assign>(add, variable_y);

        f_ref = make_shared<Function>(ResultVector{}, SinkVector{assign_x, assign_y}, ParameterVector{});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
