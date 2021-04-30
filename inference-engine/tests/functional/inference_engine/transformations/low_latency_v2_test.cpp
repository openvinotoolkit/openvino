// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>

#include <transformations/control_flow/unroll_tensor_iterator.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/low_latency.hpp>
#include <transformations/serialize.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace opset7;
using namespace std;

Output<Node> create_init_subgraph(const Output<Node>& in_node) {
    auto const_zero = make_shared<Constant>(in_node.get_element_type(), Shape{1}, 0);
    auto shape_of = make_shared<ShapeOf>(in_node);
    auto broadcast = make_shared<Broadcast>(const_zero, shape_of);
    return broadcast->output(0);
}

TEST(TransformationTests, LowLatency_v2_LSTM) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3}, ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("LSTMTensorIterator");

        tensor_iterator->set_merged_input(C_t, C_init, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H_init, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2},
                                               ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C_t), variable_C);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, read_value_H, read_value_C, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(lstm_cell->output(0), variable_H);
        auto assign_C = std::make_shared<Assign>(lstm_cell->output(1), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        f_ref = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_GRU) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell = std::make_shared<GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(gru_cell);
        auto unsqueeze = std::make_shared<Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        f = std::make_shared<Function>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>();

        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("GRUTensorIterator/variable0");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{384}, b_val);

        auto rnn_cell = std::make_shared<GRUCell>(squeeze, read_value_H, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(rnn_cell->output(0), variable_H);
        auto res_1 = std::make_shared<Result>(assign_H);
        auto unsqueeze = std::make_shared<Unsqueeze>(rnn_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        f_ref = std::make_shared<Function>(OutputVector{unsqueeze}, ParameterVector{Xi, H_t});
        f_ref->add_sinks({assign_H});
        assign_H->add_control_dependency(read_value_H);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_RNN) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(rnn_cell);
        auto unsqueeze = std::make_shared<Unsqueeze>(rnn_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi,
                                                                                                   Yi});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        f = std::make_shared<Function>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>();

        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("RNNTensorIterator/variable0");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<RNNCell>(squeeze, read_value_H, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(rnn_cell->output(0), variable_H);
        auto res_1 = std::make_shared<Result>(assign_H);
        auto unsqueeze = std::make_shared<Unsqueeze>(rnn_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        f_ref = std::make_shared<Function>(OutputVector{unsqueeze}, ParameterVector{Xi, H_t});
        f_ref->add_sinks({assign_H});
        assign_H->add_control_dependency(read_value_H);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_LSTMReshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3},
                                                       ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(C_t, C, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H,
                                                                                                               C});

        // Reshape
        // change the number of iteration of TI. 2 -> 1
        auto new_X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        f->replace_parameter(0, new_X);
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>();

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C_t), variable_C);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, read_value_H, read_value_C, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(lstm_cell->output(0), variable_H);
        auto assign_C = std::make_shared<Assign>(lstm_cell->output(1), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        f_ref = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_LSTM_Loop) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body_condition = std::make_shared<Constant>(
                element::boolean, Shape{1}, true);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3, body_condition},
                                                       ParameterVector{Xi, H_t, C_t});

        auto trip_count =
                std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition =
                std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, res_3);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2},
                                               ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>();

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C_t), variable_C);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, read_value_H, read_value_C, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(lstm_cell->output(0), variable_H);
        auto assign_C = std::make_shared<Assign>(lstm_cell->output(1), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        f_ref = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_LSTM_several_iterations) {
    constexpr int ITER_CNT = 5;
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{ITER_CNT, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3},
                                               ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(C_t, C, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H,
                                                                                       C});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>(ITER_CNT);

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    // TensorIterator not unrolled.
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{ITER_CNT, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C), variable_C);

        // Body

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3},
                                               ParameterVector{Xi, H_t, C_t});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_merged_input(C_t, read_value_C, res_3);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, read_value_H, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);
        auto out2 = tensor_iterator->get_iter_value(res_3, -1);

        auto assign_H = std::make_shared<Assign>(out0, variable_H);
        auto assign_C = std::make_shared<Assign>(out2, variable_C);
        auto outer_res_2 = std::make_shared<Result>(out1);
        auto outer_res_1 = std::make_shared<Result>(out0);
        f_ref = std::make_shared<Function>(OutputVector{outer_res_1, outer_res_2}, ParameterVector{X, H, C});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency_v2_LSTM_Loop_Reshape) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body_condition = std::make_shared<Constant>(
                element::boolean, Shape{1}, true);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3, body_condition},
                                               ParameterVector{Xi, H_t, C_t});

        auto trip_count =
                std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition =
                std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, res_3);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2},
                                       ParameterVector{X, H_init, C_init});

        // Reshape
        // change the number of iteration of Loop. 10 -> 1
        auto new_X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        f->replace_parameter(0, new_X);
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>(1);

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H_t), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C_t), variable_C);
        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, read_value_H, read_value_C, W, R, B, 128);
        auto assign_H = std::make_shared<Assign>(lstm_cell->output(0), variable_H);
        auto assign_C = std::make_shared<Assign>(lstm_cell->output(1), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        f_ref = std::make_shared<Function>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, LowLatency_v2_LSTM_Loop_several_iterations) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body_condition = std::make_shared<Constant>(
                element::boolean, Shape{1}, true);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3, body_condition},
                                               ParameterVector{Xi, H_t, C_t});

        auto trip_count =
                std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition =
                std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, res_3);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Function>(NodeVector{res_ti_1, res_ti_2},
                                       ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::LowLatency_v2>(10);

        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/variable0");
        const std::string variable_name_C("LSTMTensorIterator/variable1");
        auto variable_H = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_H});
        auto variable_C = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, variable_name_C});
        auto read_value_H = std::make_shared<ReadValue>(create_init_subgraph(H), variable_H);
        auto read_value_C = std::make_shared<ReadValue>(create_init_subgraph(C), variable_C);

        // Body
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, H_t, C_t, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell->output(0));
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto res_3 = std::make_shared<Result>(lstm_cell->output(1));
        auto body_condition = std::make_shared<Constant>(
                element::boolean, Shape{1}, true);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2, res_3, body_condition},
                                               ParameterVector{Xi, H_t, C_t});

        auto trip_count =
                std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition =
                std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, read_value_C, res_3);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, read_value_H, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);
        auto out3 = loop->get_iter_value(res_3, -1);

        auto assign_H = std::make_shared<Assign>(out0, variable_H);
        auto assign_C = std::make_shared<Assign>(out3, variable_C);
        auto outer_res_2 = std::make_shared<Result>(out1);
        auto outer_res_1 = std::make_shared<Result>(out0);
        f_ref = std::make_shared<Function>(OutputVector{outer_res_1, outer_res_2}, ParameterVector{X, H, C});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
