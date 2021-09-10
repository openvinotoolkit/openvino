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
#include <transformations/common_optimizations/replace_inputs_outputs_with_memory.h>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace opset8;
using namespace std;

TEST(TransformationTests, replace_in_out_with_memory_by_name) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->set_friendly_name("x");
        Y->set_friendly_name("y");

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->set_friendly_name("res0");
        result1->set_friendly_name("res1");

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        std::vector<std::pair<std::string, std::string>> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ReplaceInputsOutputsWithMemory>(ov::pass::ReplaceInputsOutputsWithMemory::findInputsOutputsByName(f, pair_names));
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->set_friendly_name("x");
        Y->set_friendly_name("y");

        // create ReadValue for X
        auto variable_x = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "xres0"});
        auto const_zero_x = make_shared<Constant>(X->get_element_type(), ngraph::Shape{1}, 0);
        auto shape_of_x = make_shared<ShapeOf>(X);

        auto broadcast_x = make_shared<Broadcast>(const_zero_x, shape_of_x);
        auto read_val_x = make_shared<ReadValue>(broadcast_x, variable_x);

        // create ReadValue for Y
        auto variable_y = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "yres1"});
        auto const_zero_y = make_shared<Constant>(Y->get_element_type(), ngraph::Shape{1}, 0);
        auto shape_of_y = make_shared<ShapeOf>(Y);
        auto broadcast_y = make_shared<Broadcast>(const_zero_y, shape_of_y);
        auto read_val_y = make_shared<ReadValue>(broadcast_y, variable_y);

        auto add = make_shared<Add>(read_val_x, read_val_y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        auto assign_x = make_shared<Assign>(add, variable_x);
        auto assign_y = make_shared<Assign>(add, variable_y);
        result0->set_friendly_name("res0");
        result1->set_friendly_name("res1");

        f_ref = make_shared<Function>(ResultVector{result0, result1}, SinkVector{assign_x, assign_y}, ParameterVector{X, Y});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, replace_in_out_with_memory_direct_param_res) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->set_friendly_name("x");
        Y->set_friendly_name("y");

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->set_friendly_name("res0");
        result1->set_friendly_name("res1");

        f = make_shared<Function>(ResultVector{result0, result1}, ParameterVector{X, Y});
        std::vector<std::pair<std::string, std::string>> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::ReplaceInputsOutputsWithMemory>(ov::pass::ReplaceInputsOutputsWithMemory::InOutPairs{{X, result0}, {Y, result1}});
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->set_friendly_name("x");
        Y->set_friendly_name("y");

        // create ReadValue for X
        auto variable_x = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "xres0"});
        auto const_zero_x = make_shared<Constant>(X->get_element_type(), ngraph::Shape{1}, 0);
        auto shape_of_x = make_shared<ShapeOf>(X);

        auto broadcast_x = make_shared<Broadcast>(const_zero_x, shape_of_x);
        auto read_val_x = make_shared<ReadValue>(broadcast_x, variable_x);

        // create ReadValue for Y
        auto variable_y = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "yres1"});
        auto const_zero_y = make_shared<Constant>(Y->get_element_type(), ngraph::Shape{1}, 0);
        auto shape_of_y = make_shared<ShapeOf>(Y);
        auto broadcast_y = make_shared<Broadcast>(const_zero_y, shape_of_y);
        auto read_val_y = make_shared<ReadValue>(broadcast_y, variable_y);

        auto add = make_shared<Add>(read_val_x, read_val_y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        auto assign_x = make_shared<Assign>(add, variable_x);
        auto assign_y = make_shared<Assign>(add, variable_y);
        result0->set_friendly_name("res0");
        result1->set_friendly_name("res1");

        f_ref = make_shared<Function>(ResultVector{result0, result1}, SinkVector{assign_x, assign_y}, ParameterVector{X, Y});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
