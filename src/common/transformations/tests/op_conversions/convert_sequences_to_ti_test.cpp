// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, ConvertLSTMSequenceToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X,
                                                                   Y,
                                                                   Z,
                                                                   seq_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho, Co}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho, Co}, ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H, res_ti_C}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertLSTMSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X,
                                                                   Y,
                                                                   Z,
                                                                   seq_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho, Co}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho, Co}, ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H, res_ti_C}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertQuantizedLSTMSequenceToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto input_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{}, {20});
        auto X_fq = std::make_shared<opset5::FakeQuantize>(X, input_low, input_high, input_low, input_high, 255);
        auto H = opset5::Constant::create(element::f32, Shape{1, 1, 128}, {1});
        auto C = opset5::Constant::create(element::f32, Shape{1, 1, 128}, {2});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, {1});
        auto W_fq = std::make_shared<opset5::FakeQuantize>(W, input_low, input_high, input_low, input_high, 256);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, {2});
        auto R_fq = std::make_shared<opset5::FakeQuantize>(R, input_low, input_high, input_low, input_high, 256);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, {3});
        auto B_abs = std::make_shared<opset5::Abs>(B);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X_fq,
                                                                   H,
                                                                   C,
                                                                   seq_lengths,
                                                                   W_fq,
                                                                   R_fq,
                                                                   B_abs,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<Model>(NodeVector{Y, Ho, Co}, ParameterVector{X});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto input_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{}, {20});
        auto X_fq = std::make_shared<opset5::FakeQuantize>(X, input_low, input_high, input_low, input_high, 255);

        auto H = opset5::Constant::create(element::f32, Shape{1, 128}, {1});
        auto C = opset5::Constant::create(element::f32, Shape{1, 128}, {2});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto first_axis = opset5::Constant::create(element::i64, Shape{1}, {0});

        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, {1});
        auto W_fq = std::make_shared<opset5::FakeQuantize>(W, input_low, input_high, input_low, input_high, 256);
        auto W_squeezed = std::make_shared<opset5::Squeeze>(W_fq, first_axis);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, {2});
        auto R_fq = std::make_shared<opset5::FakeQuantize>(R, input_low, input_high, input_low, input_high, 256);
        auto R_squeezed = std::make_shared<opset5::Squeeze>(R_fq, first_axis);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, {3});
        auto B_abs = std::make_shared<opset5::Abs>(B);
        auto B_squeezed = std::make_shared<opset5::Squeeze>(B_abs, first_axis);

        // Body
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        auto second_axis = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, second_axis);

        auto Hi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Ci = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Hi, Ci, W_squeezed, R_squeezed, B_squeezed, 128);

        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));
        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));
        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), second_axis);
        auto Y = std::make_shared<opset5::Result>(unsqueeze_y);

        auto body = std::make_shared<Model>(OutputVector{Y, Ho, Co}, ParameterVector{Xi, Hi, Ci, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X_fq, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Hi, H, Ho);
        tensor_iterator->set_merged_input(Ci, C, Co);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), second_axis));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), second_axis));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), second_axis));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        f_ref = std::make_shared<Model>(NodeVector{res_ti_Y, res_ti_H, res_ti_C}, ParameterVector{X});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 128}, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 128}, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, axis_1);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, axis_1);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, axis_1);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y =
            std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), axis_1));
        auto res_ti_H =
            std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{384}, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  Y,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{384}, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Yi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertQuantizedGRUSequenceToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto input_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{}, {20});
        auto X_fq = std::make_shared<opset5::FakeQuantize>(X, input_low, input_high, input_low, input_high, 255);

        auto H = opset5::Constant::create(element::f32, Shape{1, 1, 128}, {1});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, {2});
        auto W_fq = std::make_shared<opset5::FakeQuantize>(W, input_low, input_high, input_low, input_high, 256);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, {3});
        auto R_fq = std::make_shared<opset5::FakeQuantize>(R, input_low, input_high, input_low, input_high, 256);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, {4});
        auto B_abs = std::make_shared<opset5::Abs>(B);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X_fq,
                                                                  H,
                                                                  seq_lengths,
                                                                  W_fq,
                                                                  R_fq,
                                                                  B_abs,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto Y = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<Model>(NodeVector{Y, Ho}, ParameterVector{X});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto input_low = opset5::Constant::create(element::f32, Shape{}, {0});
        auto input_high = opset5::Constant::create(element::f32, Shape{}, {20});
        auto X_fq = std::make_shared<opset5::FakeQuantize>(X, input_low, input_high, input_low, input_high, 255);

        auto H = opset5::Constant::create(element::f32, Shape{1, 128}, {1});
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{1}, {2});

        auto first_axis = opset5::Constant::create(element::i64, Shape{1}, {0});

        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, {2});
        auto W_fq = std::make_shared<opset5::FakeQuantize>(W, input_low, input_high, input_low, input_high, 256);
        auto W_squeezed = std::make_shared<opset5::Squeeze>(W_fq, first_axis);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, {3});
        auto R_fq = std::make_shared<opset5::FakeQuantize>(R, input_low, input_high, input_low, input_high, 256);
        auto R_squeezed = std::make_shared<opset5::Squeeze>(R_fq, first_axis);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, {4});
        auto B_abs = std::make_shared<opset5::Abs>(B);
        auto B_squeezed = std::make_shared<opset5::Squeeze>(B_abs, first_axis);

        // Body
        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Hi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        auto second_axis = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, second_axis);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Hi, W_squeezed, R_squeezed, B_squeezed, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, second_axis);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho}, ParameterVector{Xi, Hi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X_fq, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Hi, H, Ho);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), second_axis));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), second_axis));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<Model>(NodeVector{res_ti_Y, res_ti_H}, ParameterVector{X});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, convert_lstm_seq_to_ti_with_enabled_mask) {
    auto param_x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto param_init_cell_state =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 256});
    auto param_hidden_cell_state =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 256});
    auto param_seq_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto param_w = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1024, 40});
    auto param_r = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1024, 256});
    auto param_b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1024});
    std::int64_t hidden_size = 256;
    ov::op::v5::LSTMSequence::direction lstm_direction = ov::op::v5::LSTMSequence::direction::FORWARD;
    auto lstm_cell = std::make_shared<ov::op::v5::LSTMSequence>(param_x,
                                                                param_init_cell_state,
                                                                param_hidden_cell_state,
                                                                param_seq_len,
                                                                param_w,
                                                                param_r,
                                                                param_b,
                                                                hidden_size,
                                                                lstm_direction);
    auto model = std::make_shared<ov::Model>(lstm_cell->outputs(),
                                             ov::ParameterVector{param_x,
                                                                 param_init_cell_state,
                                                                 param_hidden_cell_state,
                                                                 param_seq_len,
                                                                 param_w,
                                                                 param_r,
                                                                 param_b});
    pass::Manager m;
    m.register_pass<ov::pass::ConvertSequenceToTensorIterator>();
    m.run_passes(model);
}

TEST(TransformationTests, ConvertLSTMSequenceWithDynSeqLenToTensorIterator) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, -1, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto shape_of = std::make_shared<opset5::ShapeOf>(X);
        auto indices = opset5::Constant::create(element::i32, {1}, {1});
        auto axis = opset5::Constant::create(element::i32, {}, {0});
        auto seq_lengths = std::make_shared<opset5::Gather>(shape_of, indices, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X,
                                                                   Y,
                                                                   Z,
                                                                   seq_lengths,
                                                                   W,
                                                                   R,
                                                                   B,
                                                                   128,
                                                                   op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ov::Model>(NodeVector{Y_out, Ho, Co}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{1, -1, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto squeeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{1});

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto unsqueeze_pattern = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto body = std::make_shared<Model>(OutputVector{Y_out, Ho, Co}, ParameterVector{Xi, Yi, Zi, seq_body_param});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto shape_of = std::make_shared<opset5::ShapeOf>(X);
        auto indices = opset5::Constant::create(element::i32, {1}, {1});
        auto axis = opset5::Constant::create(element::i32, {}, {0});
        auto seq_lengths = std::make_shared<opset5::Gather>(shape_of, indices, axis);
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(
            std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_Y, res_ti_H, res_ti_C}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
