// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/loop_inputs.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/pass/visualize_tree.hpp" // DEBUG

#include "openvino/op/loop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/sinh.hpp"

using namespace testing;
using namespace ov;
using namespace ov::op;

TEST_F(TransformationTestsF, LoopInputsInvariantAndOutput) {
    {
        auto main_param0 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto main_const1 = std::make_shared<v0::Constant>(element::i32, Shape{1}, 1);

        auto trip_count = std::make_shared<v0::Constant>(element::i32, Shape{}, 10);
        auto condition = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<v5::Loop>(trip_count, condition);

        auto param0 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto param1 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto add = std::make_shared<v1::Add>(param0, param1);
        auto result0 = std::make_shared<v0::Result>(add);
        auto result1 = std::make_shared<v0::Result>(param1);

        auto body = std::make_shared<Model>(OutputVector{condition, result0, result1}, ParameterVector{param0, param1});
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);

        loop->set_invariant_input(param0, main_param0);
        loop->set_merged_input(param1, main_const1, result1);

        auto tanh = std::make_shared<v0::Tanh>(loop->get_iter_value(result0));
        auto sinh = std::make_shared<v0::Sinh>(loop->get_iter_value(result1));

        auto main_result0 = std::make_shared<v0::Result>(tanh);
        auto main_result1 = std::make_shared<v0::Result>(sinh);

        model = std::make_shared<Model>(OutputVector{main_result0, main_result1}, ParameterVector{main_param0});

        manager.register_pass<ov::pass::VisualizeTree>("./before.png"); // DEBUG
        manager.register_pass<ov::pass::LoopInputs>();
        manager.register_pass<ov::pass::VisualizeTree>("./after.png"); // DEBUG
    }

    {
        auto main_param0 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto main_const1 = std::make_shared<v0::Constant>(element::i32, Shape{1}, 1);

        auto trip_count = std::make_shared<v0::Constant>(element::i32, Shape{}, 10);
        auto condition = std::make_shared<v0::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<v5::Loop>(trip_count, condition);

        auto param0 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto param1 = std::make_shared<v0::Parameter>(element::i32, Shape{1});
        auto add = std::make_shared<v1::Add>(param0, param1);
        auto result0 = std::make_shared<v0::Result>(add);
        auto result1 = std::make_shared<v0::Result>(param1);

        auto body = std::make_shared<Model>(OutputVector{condition, result0, result1}, ParameterVector{param0, param1});
        loop->set_special_body_ports({-1, 0});
        loop->set_function(body);

        loop->set_invariant_input(param0, main_param0);
        loop->set_invariant_input(param1, main_const1);

        auto tanh = std::make_shared<v0::Tanh>(loop->get_iter_value(result0));
        auto sinh = std::make_shared<v0::Sinh>(main_const1);

        auto main_result0 = std::make_shared<v0::Result>(tanh);
        auto main_result1 = std::make_shared<v0::Result>(sinh);

        model_ref = std::make_shared<Model>(OutputVector{main_result0, main_result1}, ParameterVector{main_param0});
    }
}
