// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_stateless.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace opset8;
using namespace std;

namespace op_util = ov::op::util;

/**
 * @brief stateful model without results:
 *
 *   ReadValue(x), ReadValue(y) -> Add -> Assign(x)
 *                                      -> Assign(y)
 */
std::shared_ptr<ov::Model> get_stateful_model(const std::string& x_id = "xstate0",
                                              const std::string& y_id = "ystate1") {
    auto variable_x = std::make_shared<op_util::Variable>(op_util::VariableInfo{Shape{32, 1, 10}, element::f32, x_id});
    auto variable_y = std::make_shared<op_util::Variable>(op_util::VariableInfo{Shape{32, 1, 10}, element::f32, y_id});
    auto read_val_x = make_shared<ReadValue>(variable_x);
    auto read_val_y = make_shared<ReadValue>(variable_y);

    auto node = make_shared<Add>(read_val_x, read_val_y);
    auto assign_x = make_shared<Assign>(node, variable_x);
    auto assign_y = make_shared<Assign>(node, variable_y);
    assign_x->add_control_dependency(read_val_x);
    assign_y->add_control_dependency(read_val_y);

    auto model = make_shared<Model>(ResultVector{}, SinkVector{assign_x, assign_y}, ParameterVector{});
    model->validate_nodes_and_infer_types();
    return model;
}

/**
 * @brief the expected stateless counterpart of get_stateful_model:
 *
 *   Parameter(x), Parameter(y) -> Add -> Result
 *                                      -> Result
 */
std::shared_ptr<ov::Model> get_stateless_ref_model() {
    auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
    // MakeStateless names the new Parameters' output tensors after the variable ids
    X->get_output_tensor(0).set_names({"xstate0"});
    Y->get_output_tensor(0).set_names({"ystate1"});
    auto node = make_shared<Add>(X, Y);
    auto result0 = make_shared<Result>(node);
    auto result1 = make_shared<Result>(node);
    auto model = make_shared<Model>(ResultVector{result0, result1}, ParameterVector{X, Y});
    model->validate_nodes_and_infer_types();
    return model;
}

TEST(TransformationTests, make_stateless_basic) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = get_stateful_model();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateless>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_stateless_ref_model(); }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_variables().size(), 0);
    EXPECT_EQ(f->get_sinks().size(), 0);
    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_results().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_friendly_name(), "xstate0");
    EXPECT_EQ(f->get_parameters()[1]->get_friendly_name(), "ystate1");
}

TEST(TransformationTests, make_stateless_selected_variables) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        f = get_stateful_model("state_a", "state_b");

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateless>(std::unordered_set<std::string>{"state_a"});
        manager.run_passes(f);
    }

    // only the selected variable is converted, the other one remains a state
    EXPECT_EQ(f->get_variables().size(), 1);
    EXPECT_EQ(f->get_variables()[0]->get_info().variable_id, "state_b");
    EXPECT_EQ(f->get_sinks().size(), 1);
    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters()[0]->get_friendly_name(), "state_a");
    EXPECT_EQ(f->get_results().size(), 1);
}

TEST(TransformationTests, make_stateless_read_only_variable) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        // ReadValue without a corresponding Assign:
        //   ReadValue(x) * Constant -> Multiply -> Result
        auto variable_x =
            std::make_shared<op_util::Variable>(op_util::VariableInfo{Shape{32, 1, 10}, element::f32, "xstate0"});
        auto read_val_x = make_shared<ReadValue>(variable_x);
        auto constant = make_shared<Constant>(element::f32, Shape{32, 1, 10}, vector<float>{1});
        auto node = make_shared<Multiply>(read_val_x, constant);
        auto result = make_shared<Result>(node);

        f = make_shared<Model>(ResultVector{result}, ParameterVector{});
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateless>();
        manager.run_passes(f);
    }

    {
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        X->get_output_tensor(0).set_names({"xstate0"});
        auto constant = make_shared<Constant>(element::f32, Shape{32, 1, 10}, vector<float>{1});
        auto node = make_shared<Multiply>(X, constant);
        auto result = make_shared<Result>(node);
        f_ref = make_shared<Model>(ResultVector{result}, ParameterVector{X});
        f_ref->validate_nodes_and_infer_types();
    }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;

    EXPECT_EQ(f->get_variables().size(), 0);
    EXPECT_EQ(f->get_sinks().size(), 0);
    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_results().size(), 1);
}

TEST(TransformationTests, make_stateless_written_but_not_read_variable) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // Assign without a corresponding ReadValue:
        //   Parameter -> Result
        //   Parameter -> Assign
        auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
        auto result = make_shared<Result>(X);
        auto variable_x =
            std::make_shared<op_util::Variable>(op_util::VariableInfo{Shape{32, 1, 10}, element::f32, "xstate0"});
        auto assign_x = make_shared<Assign>(X, variable_x);

        f = make_shared<Model>(ResultVector{result}, SinkVector{assign_x}, ParameterVector{X});
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateless>();
        manager.run_passes(f);
    }

    // the written value is exposed as a Result instead of being dropped
    EXPECT_EQ(f->get_variables().size(), 0);
    EXPECT_EQ(f->get_sinks().size(), 0);
    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_results().size(), 2);
}

TEST(TransformationTests, make_stateless_multiple_writers) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto variable_x =
            std::make_shared<op_util::Variable>(op_util::VariableInfo{Shape{32, 1, 10}, element::f32, "xstate0"});
        auto read_val_x = make_shared<ReadValue>(variable_x);
        auto node = make_shared<Add>(read_val_x, read_val_x);
        auto assign_x0 = make_shared<Assign>(node, variable_x);
        auto assign_x1 = make_shared<Assign>(node, variable_x);
        assign_x0->add_control_dependency(read_val_x);
        assign_x1->add_control_dependency(read_val_x);

        f = make_shared<Model>(ResultVector{}, SinkVector{assign_x0, assign_x1}, ParameterVector{});
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateless>();
        try {
            manager.run_passes(f);
        } catch (::ov::AssertFailure& ex) {
            EXPECT_STR_CONTAINS(ex.what(), "doesn't support several Assign operations");
            return;
        } catch (...) {
            FAIL() << "Expected ::ov::AssertFailure";
        }
        FAIL() << "Expected ::ov::AssertFailure";
    }
}
