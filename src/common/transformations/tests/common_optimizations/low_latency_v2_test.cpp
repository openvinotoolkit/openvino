// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;
using namespace opset7;
using namespace std;

Output<Node> create_init_subgraph(const Output<Node>& in_node) {
    auto const_zero = make_shared<Constant>(in_node.get_element_type(), Shape{1}, 0);
    auto shape_of = make_shared<ShapeOf>(in_node);
    auto broadcast = make_shared<Broadcast>(const_zero, shape_of);
    return broadcast->output(0);
}

Output<Node> insert_identity(const Output<Node>& in_node) {
    auto axis_1 = Constant::create(element::i64, Shape{1}, {1});
    auto identity_1 = std::make_shared<Unsqueeze>(in_node, axis_1);
    return std::make_shared<Squeeze>(identity_1, axis_1);
}

std::shared_ptr<Model> createLSTMBody(const std::shared_ptr<Parameter>& Xi,
                                      const std::shared_ptr<Parameter>& H_t,
                                      const std::shared_ptr<Parameter>& C_t,
                                      bool is_loop = false) {
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

    auto func = std::make_shared<Model>(OutputVector{res_1, res_2, res_3}, ParameterVector{Xi, H_t, C_t});
    if (is_loop) {
        auto body_condition = std::make_shared<Constant>(element::boolean, Shape{1}, true);
        auto cond_res = std::make_shared<Result>(body_condition);
        func->add_results({cond_res});
    }
    return func;
}

TEST(TransformationTests, LowLatency2_LSTM) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t);
        auto results = body->get_results();

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("LSTMTensorIterator");

        tensor_iterator->set_merged_input(C_t, C_init, results[2]);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H_init, results[0]);

        tensor_iterator->get_iter_value(results[0], -1);
        tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/H_t/variable_2");
        const std::string variable_name_C("LSTMTensorIterator/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C_t->get_shape(), C_t->get_element_type(), variable_name_C});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(lstm_cell->output(0)), variable_H);
        auto assign_C = std::make_shared<Assign>(insert_identity(lstm_cell->output(1)), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        auto res_1 = std::make_shared<Result>(insert_identity(lstm_cell->output(0)));
        f_ref = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_GRU) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        Xi->set_friendly_name("Xi");
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        Yi->set_friendly_name("Yi");

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
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("GRUTensorIterator");

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("GRUTensorIterator/Yi/variable");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(rnn_cell->output(0)), variable_H);
        auto res_1 = std::make_shared<Result>(assign_H);
        auto unsqueeze = std::make_shared<Unsqueeze>(rnn_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        f_ref = std::make_shared<Model>(ResultVector{res_2}, ParameterVector{Xi, H_t});
        f_ref->add_sinks({assign_H});
        assign_H->add_control_dependency(read_value_H);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_RNN) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        Xi->set_friendly_name("Xi");
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        Yi->set_friendly_name("Yi");

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
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("RNNTensorIterator");

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("RNNTensorIterator/Yi/variable");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(rnn_cell->output(0)), variable_H);
        auto res_1 = std::make_shared<Result>(assign_H);
        auto unsqueeze = std::make_shared<Unsqueeze>(rnn_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        f_ref = std::make_shared<Model>(ResultVector{res_2}, ParameterVector{Xi, H_t});
        f_ref->add_sinks({assign_H});
        assign_H->add_control_dependency(read_value_H);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_LSTMReshape) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t);
        auto results = body->get_results();

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("LSTMTensorIterator");

        tensor_iterator->set_merged_input(C_t, C, results[2]);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H, results[0]);

        auto out0 = tensor_iterator->get_iter_value(results[0], -1);
        auto out1 = tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H, C});

        // Reshape
        // change the number of iteration of TI. 2 -> 1
        auto new_X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        f->replace_parameter(0, new_X);
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/H_t/variable_2");
        const std::string variable_name_C("LSTMTensorIterator/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C_t->get_shape(), C_t->get_element_type(), variable_name_C});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(lstm_cell->output(0)), variable_H);
        auto assign_C = std::make_shared<Assign>(insert_identity(lstm_cell->output(1)), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        auto res_1 = std::make_shared<Result>(insert_identity(lstm_cell->output(0)));
        f_ref = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_LSTM_Loop) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t, true);
        auto results = body->get_results();

        auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 1);
        auto exec_condition = std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, results[2]);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, results[0]);

        auto out0 = loop->get_iter_value(results[0], -1);
        auto out1 = loop->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMLoop/H_t/variable_2");
        const std::string variable_name_C("LSTMLoop/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C_t->get_shape(), C_t->get_element_type(), variable_name_C});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(lstm_cell->output(0)), variable_H);
        auto assign_C = std::make_shared<Assign>(insert_identity(lstm_cell->output(1)), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        auto res_1 = std::make_shared<Result>(insert_identity(lstm_cell->output(0)));
        f_ref = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_LSTM_several_iterations) {
    constexpr int ITER_CNT = 5;
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{ITER_CNT, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t);
        auto results = body->get_results();

        auto tensor_iterator = std::make_shared<TensorIterator>();
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("LSTMTensorIterator");

        tensor_iterator->set_merged_input(C_t, C, results[2]);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(H_t, H, results[0]);

        auto out0 = tensor_iterator->get_iter_value(results[0], -1);
        auto out1 = tensor_iterator->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<Result>(tensor_iterator->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H, C});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    // TensorIterator not unrolled.
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{ITER_CNT, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMTensorIterator/H_t/variable_2");
        const std::string variable_name_C("LSTMTensorIterator/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H->get_shape(), H->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C->get_shape(), C->get_element_type(), variable_name_C});
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
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, res_3}, ParameterVector{Xi, H_t, C_t});

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
        f_ref = std::make_shared<Model>(OutputVector{outer_res_1, outer_res_2}, ParameterVector{X, H, C});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_LSTM_Loop_Reshape) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t, true);
        auto results = body->get_results();

        auto shape_of = std::make_shared<ShapeOf>(X);
        const auto trip_count = std::make_shared<ov::op::v8::Gather>(shape_of,
                                                                     Constant::create(element::i64, {1}, {0}),
                                                                     Constant::create(element::i64, {1}, {0}));
        auto exec_condition = std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, results[2]);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, results[0]);

        auto out0 = loop->get_iter_value(results[0], -1);
        auto out1 = loop->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H_init, C_init});

        // Reshape
        // change the number of iteration of Loop. 10 -> 1
        auto new_X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        f->replace_parameter(0, new_X);
        f->validate_nodes_and_infer_types();

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>();

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMLoop/H_t/variable_2");
        const std::string variable_name_C("LSTMLoop/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H_t->get_shape(), H_t->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C_t->get_shape(), C_t->get_element_type(), variable_name_C});
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
        auto assign_H = std::make_shared<Assign>(insert_identity(lstm_cell->output(0)), variable_H);
        auto assign_C = std::make_shared<Assign>(insert_identity(lstm_cell->output(1)), variable_C);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell->output(0), axis);
        auto res_2 = std::make_shared<Result>(insert_identity(unsqueeze));
        auto res_1 = std::make_shared<Result>(insert_identity(lstm_cell->output(0)));
        f_ref = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, H_t, C_t});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LowLatency2_LSTM_Loop_several_iterations) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C_init = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto H_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        H_t->set_friendly_name("H_t");
        auto C_t = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        C_t->set_friendly_name("C_t");

        // Body
        auto body = createLSTMBody(Xi, H_t, C_t, true);
        auto results = body->get_results();

        auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition = std::make_shared<Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 3});
        loop->set_function(body);
        loop->set_friendly_name("LSTMLoop");

        loop->set_merged_input(C_t, C_init, results[2]);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(H_t, H_init, results[0]);

        auto out0 = loop->get_iter_value(results[0], -1);
        auto out1 = loop->get_concatenated_slices(results[1], 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1, res_ti_2}, ParameterVector{X, H_init, C_init});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::LowLatency2>(true);

        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{10, 1, 16});
        auto H = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto C = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        const std::string variable_name_H("LSTMLoop/H_t/variable_2");
        const std::string variable_name_C("LSTMLoop/C_t/variable_0");
        auto variable_H = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H->get_shape(), H->get_element_type(), variable_name_H});
        auto variable_C = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C->get_shape(), C->get_element_type(), variable_name_C});
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
        auto body_condition = std::make_shared<Constant>(element::boolean, Shape{1}, true);
        auto body =
            std::make_shared<Model>(OutputVector{res_1, res_2, res_3, body_condition}, ParameterVector{Xi, H_t, C_t});

        auto trip_count = std::make_shared<Constant>(element::i64, Shape{}, 10);
        auto exec_condition = std::make_shared<Constant>(element::boolean, Shape{}, true);
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
        f_ref = std::make_shared<Model>(OutputVector{outer_res_1, outer_res_2}, ParameterVector{X, H, C});
        f_ref->add_sinks({assign_C, assign_H});
        assign_H->add_control_dependency(read_value_H);
        assign_C->add_control_dependency(read_value_C);
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

namespace {
using OutPtr = Output<Node>;
enum class RNNType : size_t {
    RNN = 1,
    GRU = 3,
    LSTM = 4,
};

struct LLT2Params {
    RNNType rnn_type;
    size_t seq_len;
};

struct RNNAttributes {
    size_t batch;
    size_t num_dir;
    size_t hidden_size;
    size_t input_size;
    size_t seq_len;
};

OutputVector create_sequence(RNNType rnn_type, RNNAttributes attrs, const OutPtr& X, const OutPtr& H, const OutPtr& C) {
    auto gate = static_cast<size_t>(rnn_type);
    auto shape_of = std::make_shared<ShapeOf>(X);
    auto batch_dimension = std::make_shared<Gather>(shape_of,
                                                    Constant::create(element::i64, {1}, {0}),
                                                    Constant::create(element::i64, {}, {0}));
    auto seq_len_dim = std::make_shared<Gather>(shape_of,
                                                Constant::create(element::i64, {1}, {1}),
                                                Constant::create(element::i64, {}, {0}));
    auto seq_lengths = std::make_shared<Broadcast>(seq_len_dim, batch_dimension);
    auto W = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size, attrs.input_size}, 0);
    auto R = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size, attrs.hidden_size}, 0);
    auto B = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size}, 0);

    shared_ptr<Node> sequence;
    if (rnn_type == RNNType::LSTM) {
        sequence = make_shared<LSTMSequence>(X,
                                             H,
                                             C,
                                             seq_lengths,
                                             W,
                                             R,
                                             B,
                                             attrs.hidden_size,
                                             LSTMSequence::direction::FORWARD);
    } else if (rnn_type == RNNType::RNN) {
        sequence = make_shared<RNNSequence>(X,
                                            H,
                                            seq_lengths,
                                            W,
                                            R,
                                            B,
                                            attrs.hidden_size,
                                            op::RecurrentSequenceDirection::FORWARD);
    } else if (rnn_type == RNNType::GRU) {
        sequence = make_shared<GRUSequence>(X,
                                            H,
                                            seq_lengths,
                                            W,
                                            R,
                                            B,
                                            attrs.hidden_size,
                                            op::RecurrentSequenceDirection::FORWARD);
    }
    return sequence->outputs();
}

shared_ptr<ReadValue> create_read_value(const shared_ptr<Parameter>& param,
                                        const shared_ptr<ov::op::util::Variable>& variable) {
    auto const_zero = make_shared<Constant>(param->get_element_type(), ov::Shape{1}, 0);
    auto shape_of = make_shared<ShapeOf>(param);
    auto broadcast = make_shared<Broadcast>(const_zero, shape_of);
    auto read_value = make_shared<ReadValue>(broadcast, variable);
    return read_value;
}

OutputVector create_cell_reference(RNNType rnn_type,
                                   RNNAttributes attrs,
                                   const OutPtr& X,
                                   const OutPtr& H,
                                   const OutPtr& C) {
    auto gate = static_cast<size_t>(rnn_type);
    auto axis_0 = Constant::create(element::i32, Shape{1}, {0});
    auto axis_1 = Constant::create(element::i32, Shape{1}, {1});

    shared_ptr<Node> squeeze_C;
    if (rnn_type == RNNType::LSTM) {
        squeeze_C = make_shared<Squeeze>(C, axis_1);
    }

    auto squeeze_X = make_shared<Squeeze>(X, axis_1);
    auto squeeze_H = make_shared<Squeeze>(H, axis_1);

    auto W = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size, attrs.input_size}, 0);
    auto R = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size, attrs.hidden_size}, 0);
    auto B = make_shared<Constant>(element::f32, Shape{attrs.num_dir, gate * attrs.hidden_size}, 0);

    auto squeeze_W = std::make_shared<Squeeze>(W, axis_0);
    auto squeeze_R = std::make_shared<Squeeze>(R, axis_0);
    auto squeeze_B = std::make_shared<Squeeze>(B, axis_0);

    shared_ptr<Node> cell;
    if (rnn_type == RNNType::LSTM) {
        cell =
            make_shared<LSTMCell>(squeeze_X, squeeze_H, squeeze_C, squeeze_W, squeeze_R, squeeze_B, attrs.hidden_size);
    } else if (rnn_type == RNNType::RNN) {
        cell = make_shared<RNNCell>(squeeze_X, squeeze_H, squeeze_W, squeeze_R, squeeze_B, attrs.hidden_size);
    } else if (rnn_type == RNNType::GRU) {
        cell = make_shared<GRUCell>(squeeze_X, squeeze_H, squeeze_W, squeeze_R, squeeze_B, attrs.hidden_size);
    }

    auto axis_12 = Constant::create(element::i32, Shape{2}, {1, 2});
    auto unsqueeze_Y = make_shared<Unsqueeze>(cell->output(0), axis_12);
    auto unsqueeze_H = make_shared<Unsqueeze>(cell->output(0), axis_1);
    OutputVector outputs = {unsqueeze_Y, unsqueeze_H};

    if (rnn_type == RNNType::LSTM) {
        auto unsqueeze_C = make_shared<Unsqueeze>(cell->output(1), axis_1);
        outputs.push_back(unsqueeze_C);
    }
    return outputs;
}
}  // namespace

class LLT2Sequence : public WithParamInterface<LLT2Params>, public TransformationTestsF {};

TEST_P(LLT2Sequence, RNNLowLatency_v2) {
    const auto& p = GetParam();
    RNNAttributes attrs = {1, 1, 10, 4, p.seq_len};
    {
        auto X = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.seq_len, attrs.input_size});
        auto H = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.num_dir, attrs.hidden_size});
        auto C = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.num_dir, attrs.hidden_size});
        auto outputs = create_sequence(p.rnn_type, attrs, X, H, C);
        ParameterVector params{X, H};
        if (p.rnn_type == RNNType::LSTM) {
            params.push_back(C);
        }
        ResultVector results;
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto result = std::make_shared<Result>(outputs[i]);
            auto result_name = "output_" + std::to_string(i);
            result->set_friendly_name(result_name);
            results.push_back(result);
        }
        model = make_shared<ov::Model>(results, params);
        manager.register_pass<pass::LowLatency2>();
    }

    {
        auto X = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.seq_len, attrs.input_size});
        auto H = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.num_dir, attrs.hidden_size});
        auto C = make_shared<Parameter>(element::f32, Shape{attrs.batch, attrs.num_dir, attrs.hidden_size});
        auto variable_h = make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{H->get_shape(), H->get_element_type(), "node_28/variable_0"});
        auto variable_c = make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{C->get_shape(), C->get_element_type(), "node_28/variable_1"});
        auto read_val_H = create_read_value(H, variable_h);
        auto read_val_C = create_read_value(C, variable_c);

        OutputVector outputs;
        if (p.seq_len == 1) {
            outputs = create_cell_reference(p.rnn_type, attrs, X, read_val_H, read_val_C);
        } else {
            outputs = create_sequence(p.rnn_type, attrs, X, read_val_H, read_val_C);
        }

        SinkVector sinks;
        auto assign_h = make_shared<Assign>(outputs[1], variable_h);
        sinks.push_back(assign_h);
        assign_h->add_control_dependency(read_val_H);

        ParameterVector params = {X, H};
        if (p.rnn_type == RNNType::LSTM) {
            params.push_back(C);
            auto assign_c = make_shared<Assign>(outputs[2], variable_c);
            assign_c->add_control_dependency(read_val_C);
            sinks.push_back(assign_c);
        }

        ResultVector results;
        for (size_t i = 0; i < outputs.size(); ++i) {
            auto result = std::make_shared<Result>(outputs[i]);
            auto result_name = "output_" + std::to_string(i);
            result->set_friendly_name(result_name);
            results.push_back(result);
        }
        model_ref = make_shared<ov::Model>(results, sinks, params);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<LLT2Params> llt_v2_params = {
    LLT2Params{RNNType::RNN, 1},
    LLT2Params{RNNType::GRU, 1},
    LLT2Params{RNNType::LSTM, 1},
    LLT2Params{RNNType::RNN, 10},
    LLT2Params{RNNType::GRU, 10},
    LLT2Params{RNNType::LSTM, 10},
};

INSTANTIATE_TEST_SUITE_P(LLT2Sequence, LLT2Sequence, ValuesIn(llt_v2_params));
