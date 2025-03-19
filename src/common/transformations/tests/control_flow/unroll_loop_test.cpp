// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace opset6;

TEST(TransformationTests, UnrollLoopGRUCell) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
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
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 2);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis_split = Constant::create(element::i64, Shape{}, {0});
        auto split = std::make_shared<Split>(X, axis_split, 2);
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell_1 = std::make_shared<GRUCell>(squeeze_1, Y, W, R, B, 128);
        auto gru_cell_2 = std::make_shared<GRUCell>(squeeze_2, gru_cell_1, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(gru_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<Unsqueeze>(gru_cell_2, axis);
        auto concat = std::make_shared<Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<Result>(concat);
        // auto res_ti_2 = std::make_shared<Result>(unsqueeze_2);
        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollLoopRNNCell) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
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
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 2);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis_split = Constant::create(element::i64, Shape{}, {0});
        auto split = std::make_shared<Split>(X, axis_split, 2);
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell_1 = std::make_shared<RNNCell>(squeeze_1, Y, W, R, B, 128);
        auto rnn_cell_2 = std::make_shared<RNNCell>(squeeze_2, rnn_cell_1, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(rnn_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<Unsqueeze>(rnn_cell_2, axis);
        auto concat = std::make_shared<Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<Result>(concat);
        // auto res_ti_2 = std::make_shared<Result>(unsqueeze_2);
        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollLoopLSTMCell) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi, Zi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 2);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_invariant_input(Zi, Z);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis_split = Constant::create(element::i64, Shape{}, {0});
        auto split = std::make_shared<Split>(X, axis_split, 2);
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(split->output(0), axis);
        auto squeeze_2 = std::make_shared<Squeeze>(split->output(1), axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell_1 = std::make_shared<LSTMCell>(squeeze_1, Y, Z, W, R, B, 128);
        auto lstm_cell_2 = std::make_shared<LSTMCell>(squeeze_2, lstm_cell_1, Z, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(lstm_cell_1, axis);
        auto unsqueeze_2 = std::make_shared<Unsqueeze>(lstm_cell_2, axis);
        auto concat = std::make_shared<Concat>(OutputVector{unsqueeze_1, unsqueeze_2}, 0);

        auto res_ti_1 = std::make_shared<Result>(concat);
        // auto res_ti_2 = std::make_shared<Result>(unsqueeze_2);
        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollLoopGRUCellSingleIteration) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
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
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 1);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(X, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell_1 = std::make_shared<GRUCell>(squeeze_1, Y, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(gru_cell_1, axis);

        auto res_ti_1 = std::make_shared<Result>(unsqueeze_1);
        // auto res_ti_2 = std::make_shared<Result>(unsqueeze_2);
        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollLoopRNNCellSingleIteration) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
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
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 1);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(X, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell_1 = std::make_shared<RNNCell>(squeeze_1, Y, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(rnn_cell_1, axis);
        auto res_ti_1 = std::make_shared<Result>(unsqueeze_1);

        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollLoopLSTMCellSingleIteration) {
    std::shared_ptr<Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze = std::make_shared<Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<Result>(lstm_cell);
        auto unsqueeze = std::make_shared<Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<Result>(unsqueeze);
        auto body_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{1}, true);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2, body_condition}, ParameterVector{Xi, Yi, Zi});

        auto trip_count = std::make_shared<opset6::Constant>(element::i64, Shape{}, 1);
        auto exec_condition = std::make_shared<opset6::Constant>(element::boolean, Shape{}, true);
        auto loop = std::make_shared<Loop>(trip_count, exec_condition);
        loop->set_special_body_ports({-1, 2});
        loop->set_function(body);

        loop->set_invariant_input(Zi, Z);
        loop->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        loop->set_merged_input(Yi, Y, res_1);

        auto out0 = loop->get_iter_value(res_1, -1);
        auto out1 = loop->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<Result>(loop->output(1));
        // auto res_ti_2 = std::make_shared<Result>(loop->output(0));
        f = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});

        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<Parameter>(element::f32, Shape{1, 128});

        auto axis = Constant::create(element::i64, Shape{}, {0});
        auto squeeze_1 = std::make_shared<Squeeze>(X, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell_1 = std::make_shared<LSTMCell>(squeeze_1, Y, Z, W, R, B, 128);

        auto unsqueeze_1 = std::make_shared<Unsqueeze>(lstm_cell_1, axis);
        auto res_ti_1 = std::make_shared<Result>(unsqueeze_1);
        // auto res_ti_2 = std::make_shared<Result>(unsqueeze_2);
        f_ref = std::make_shared<Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
