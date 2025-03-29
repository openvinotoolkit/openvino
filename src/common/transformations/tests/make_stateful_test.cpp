// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_stateful.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace opset8;
using namespace std;

std::shared_ptr<ov::Model> get_test_model(bool insert_squeeze, bool use_friendly_names) {
    std::shared_ptr<ov::Model> model;
    auto X = make_shared<Parameter>(element::f32, Shape{32, 1, 10});
    auto Y = make_shared<Parameter>(element::f32, Shape{32, 1, 10});

    if (!use_friendly_names) {
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});
    } else {
        X->set_friendly_name("x");
        Y->set_friendly_name("y");
    }

    // -> Add  -> Squeeze -> Result
    //         -> Result
    // or
    // -> Add -> Result
    //        -> Result
    std::shared_ptr<Node> node;
    node = make_shared<Add>(X, Y);
    auto result0 = make_shared<Result>(node);
    auto result1 = make_shared<Result>(node);

    if (!use_friendly_names) {
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});
    } else {
        result0->set_friendly_name("res0");
        result1->set_friendly_name("res1");
    }

    model = make_shared<Model>(ResultVector{result0, result1}, ParameterVector{X, Y});
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> get_ref_model(bool insert_squeeze, bool use_friendly_names) {
    std::shared_ptr<ov::Model> model;
    // create ReadValue for X
    auto variable_x =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{Shape{32, 1, 10}, element::f32, "xres0"});
    auto read_val_x = make_shared<ReadValue>(variable_x);

    // create ReadValue for Y
    auto variable_y =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{Shape{32, 1, 10}, element::f32, "yres1"});
    auto read_val_y = make_shared<ReadValue>(variable_y);

    if (!use_friendly_names) {
        read_val_x->get_output_tensor(0).add_names({"x"});
        read_val_y->get_output_tensor(0).add_names({"y"});
    } else {
        read_val_x->set_friendly_name("x");
        read_val_y->set_friendly_name("y");
    }

    // -> Add  -> Squeeze -> Assign
    //         -> Assign
    // or
    // -> Add -> Assign
    //        -> Assign
    shared_ptr<ov::Node> node;
    node = make_shared<Add>(read_val_x, read_val_y);
    auto assign_x = make_shared<Assign>(node, variable_x);

    if (!use_friendly_names) {
        node->get_output_tensor(0).add_names({"res0"});
    } else {
        node->set_friendly_name("res0");
    }

    auto assign_y = make_shared<Assign>(node, variable_y);
    if (!use_friendly_names) {
        node->get_output_tensor(0).add_names({"res1"});
    } else {
        node->set_friendly_name("res1");
    }

    assign_x->add_control_dependency(read_val_x);
    assign_y->add_control_dependency(read_val_y);

    model = make_shared<Model>(ResultVector{}, SinkVector{assign_x, assign_y}, ParameterVector{});
    model->validate_nodes_and_infer_types();
    return model;
}

TEST(TransformationTests, make_stateful_by_tensor_name) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = get_test_model(true, false);
        std::map<std::string, std::string> tensor_names = {{"x", "res0"}, {"y", "res1"}};

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(tensor_names);

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_ref_model(true, false); }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, make_stateful_by_param_res) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = get_test_model(true, true);
        auto pairs = ov::pass::MakeStateful::ParamResPairs{{f->get_parameters()[0], f->get_results()[0]},
                                                           {f->get_parameters()[1], f->get_results()[1]}};

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pairs);
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_ref_model(true, true); }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, make_stateful_dynamic_shapes) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        // dynamic shapes are not supported
        auto X = make_shared<Parameter>(element::f32, PartialShape::dynamic());
        auto Y = make_shared<Parameter>(element::f32, PartialShape::dynamic());
        X->get_output_tensor(0).add_names({"x"});
        Y->get_output_tensor(0).add_names({"y"});

        auto add = make_shared<Add>(X, Y);
        auto result0 = make_shared<Result>(add);
        auto result1 = make_shared<Result>(add);
        result0->get_input_tensor(0).add_names({"res0"});
        result1->get_input_tensor(0).add_names({"res1"});

        f = make_shared<Model>(ResultVector{result0, result1}, ParameterVector{X, Y});
        map<std::string, std::string> pair_names = {{"x", "res0"}, {"y", "res1"}};
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pair_names);

        try {
            manager.run_passes(f);
        } catch (::ov::AssertFailure& ex) {
            EXPECT_STR_CONTAINS(ex.what(), "MakeStateful transformation doesn't support dynamic shapes.");
        } catch (...) {
            FAIL() << "Expected ::ov::AssertFailure";
        }
    }
}

TEST(TransformationTests, make_stateful_one_out_to_several_results_by_tensor_names) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = get_test_model(false, false);
        std::map<std::string, std::string> tensor_names = {{"x", "res0"}, {"y", "res1"}};

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(tensor_names);

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_ref_model(false, false); }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, make_stateful_one_out_to_several_results_by_param_res) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = get_test_model(false, true);
        auto pairs = ov::pass::MakeStateful::ParamResPairs{{f->get_parameters()[0], f->get_results()[0]},
                                                           {f->get_parameters()[1], f->get_results()[1]}};

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::MakeStateful>(pairs);
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_ref_model(false, true); }
    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}
