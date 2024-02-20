// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_ti_to_sequences.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

namespace {

std::shared_ptr<ov::Node> create_seq_len(const std::shared_ptr<ov::Node>& X) {
    auto shape_of = std::make_shared<opset5::ShapeOf>(X);
    auto batch_dimension = std::make_shared<opset5::Gather>(shape_of,
                                                            opset5::Constant::create(element::i64, {1}, {0}),
                                                            opset5::Constant::create(element::i64, {}, {0}));
    auto seq_len_dim = std::make_shared<opset5::Gather>(shape_of,
                                                        opset5::Constant::create(element::i64, {1}, {1}),
                                                        opset5::Constant::create(element::i64, {}, {0}));
    auto seq_lengths = std::make_shared<opset5::Broadcast>(seq_len_dim, batch_dimension);
    return seq_lengths;
}

}  // namespace

TEST(TransformationTests, ConvertTensorIteratorToLSTMSequence) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto lstm_res_1 = std::make_shared<opset5::Result>(lstm_cell->output(0));
        auto lstm_res_2 = std::make_shared<opset5::Result>(lstm_cell->output(1));
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell->output(0), reshape_pattern_2, false);
        auto lstm_res1_unsqueeze = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{lstm_res_1, lstm_res1_unsqueeze, lstm_res_2},
                                            ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, lstm_res_1);

        auto out0 = tensor_iterator->get_concatenated_slices(lstm_res1_unsqueeze, 0, 1, 1, -1, 1);
        auto out1 = tensor_iterator->get_iter_value(lstm_res_1, -1);
        auto out2 = tensor_iterator->get_iter_value(lstm_res_2, -1);

        auto res_ti_0 = std::make_shared<opset5::Result>(tensor_iterator->output(0));
        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<opset5::Result>(tensor_iterator->output(2));

        res_ti_0->set_friendly_name("Result1");
        res_ti_1->set_friendly_name("Result2");
        res_ti_2->set_friendly_name("Result3");
        f = std::make_shared<ov::Model>(NodeVector{res_ti_0, res_ti_1, res_ti_2}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToLSTMSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);
        auto in_2 = std::make_shared<opset5::Unsqueeze>(Z, axis_1);

        auto seq_lengths = create_seq_len(X);
        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(X,
                                                               in_1,
                                                               in_2,
                                                               seq_lengths,
                                                               W,
                                                               R,
                                                               B,
                                                               128,
                                                               op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(lstm_seq->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(lstm_seq->output(1), axis_out);
        auto out_2 = std::make_shared<opset5::Squeeze>(lstm_seq->output(2), axis_out);

        auto res_ti_0 = std::make_shared<opset5::Result>(out_0);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_1);
        auto res_ti_2 = std::make_shared<opset5::Result>(out_2);
        res_ti_0->set_friendly_name("Result1");
        res_ti_1->set_friendly_name("Result2");
        res_ti_2->set_friendly_name("Result3");

        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_0, res_ti_1, res_ti_2}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToLSTMSequenceDynamicReshapeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, -1});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);

        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToLSTMSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);
        auto in_2 = std::make_shared<opset5::Unsqueeze>(Z, axis_1);

        auto seq_lengths = create_seq_len(X);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(X,
                                                               in_1,
                                                               in_2,
                                                               seq_lengths,
                                                               W,
                                                               R,
                                                               B,
                                                               128,
                                                               op::RecurrentSequenceDirection::FORWARD);

        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(lstm_seq->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(lstm_seq->output(1), axis_out);
        auto out_2 = std::make_shared<opset5::Squeeze>(lstm_seq->output(2), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToLSTMSequenceDynamicSqueezeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto axis = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze = std::make_shared<opset5::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);

        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToLSTMSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);
        auto in_2 = std::make_shared<opset5::Unsqueeze>(Z, axis_1);

        auto seq_lengths = create_seq_len(X);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 512}, b_val);

        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(X,
                                                               in_1,
                                                               in_2,
                                                               seq_lengths,
                                                               W,
                                                               R,
                                                               B,
                                                               128,
                                                               op::RecurrentSequenceDirection::FORWARD);

        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(lstm_seq->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(lstm_seq->output(1), axis_out);
        auto out_2 = std::make_shared<opset5::Squeeze>(lstm_seq->output(2), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToRNNSequence) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(rnn_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(rnn_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToRNNSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 128}, b_val);

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto seq_lengths = create_seq_len(X);
        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToRNNSequenceDynamicReshapeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, -1});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(rnn_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, -1});
        auto unsqueeze = std::make_shared<opset5::Reshape>(rnn_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToRNNSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 128}, b_val);

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto seq_lengths = create_seq_len(X);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToRNNSequenceDynamicSqueezeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto axis = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze = std::make_shared<opset5::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(rnn_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, -1});
        auto unsqueeze = std::make_shared<opset5::Reshape>(rnn_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToRNNSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 128, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 128, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 128}, b_val);

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto seq_lengths = create_seq_len(X);
        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(rnn_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToGRUSequence) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset5::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(gru_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(gru_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToGRUSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, b_val);

        auto seq_lengths = create_seq_len(X);
        auto gru_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(gru_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(gru_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToGRUSequenceDynamicReshapeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, -1});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset5::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(gru_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(gru_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToGRUSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, b_val);

        auto seq_lengths = create_seq_len(X);

        auto gru_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(gru_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(gru_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTensorIteratorToGRUSequenceDynamicSqueezeCase) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 1, -1});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        // Body
        auto axis = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto squeeze = std::make_shared<opset5::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset5::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(gru_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(gru_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::ConvertTensorIteratorToGRUSequence>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, 2, -1});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{-1, -1});

        auto axis_1 = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto in_1 = std::make_shared<opset5::Unsqueeze>(Y, axis_1);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = opset5::Constant::create(element::f32, Shape{1, 384, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{1, 384, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{1, 384}, b_val);

        auto seq_lengths = create_seq_len(X);

        auto gru_sequence = std::make_shared<opset5::GRUSequence>(X,
                                                                  in_1,
                                                                  seq_lengths,
                                                                  W,
                                                                  R,
                                                                  B,
                                                                  128,
                                                                  op::RecurrentSequenceDirection::FORWARD);
        auto axis_out = opset5::Constant::create(element::i64, Shape{1}, {1});
        auto out_0 = std::make_shared<opset5::Squeeze>(gru_sequence->output(0), axis_out);
        auto out_1 = std::make_shared<opset5::Squeeze>(gru_sequence->output(1), axis_out);
        auto res_ti_1 = std::make_shared<opset5::Result>(out_0);
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

using ConvertLoopToLSTMSequenceTestParams = std::tuple<bool,   // with_input_transpose
                                                       bool>;  // with_gather_reshape

class ConvertLoopToLSTMSequenceTest : public testing::WithParamInterface<ConvertLoopToLSTMSequenceTestParams>,
                                      public TransformationTestsF {};

TEST_P(ConvertLoopToLSTMSequenceTest, FusionTest) {
    const auto& params = GetParam();
    bool with_input_transpose = std::get<0>(params);
    bool with_gather_reshape = std::get<1>(params);

    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t num_sequences = 5;
    size_t batch_size = 1;

    {
        auto trip_count = op::v0::Constant::create(element::i32, Shape{}, {-1});
        auto condition = op::v0::Constant::create(element::boolean, Shape{}, {true});
        auto iteration_counter = op::v0::Constant::create(element::i32, Shape{}, {0});
        auto sequence_index = op::v0::Constant::create(element::i32, Shape{}, {0});
        std::shared_ptr<op::v0::Parameter> X;
        std::shared_ptr<Node> scatter_updates;
        if (with_input_transpose) {
            X = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            scatter_updates =
                std::make_shared<op::v1::Transpose>(X, op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2}));
        } else {
            X = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_sequences, batch_size, input_size});
            scatter_updates = X;
        }
        auto scatter_input = op::v0::Constant::create(element::f32, Shape{num_sequences, batch_size, input_size}, {0});
        std::vector<int> indexes_values(num_sequences);
        std::iota(indexes_values.begin(), indexes_values.end(), 0);
        auto scatter_indexes = op::v0::Constant::create(element::i32, Shape{num_sequences, 1}, indexes_values);
        auto scatter = std::make_shared<op::v3::ScatterNDUpdate>(scatter_input, scatter_indexes, scatter_updates);
        auto H = op::v0::Constant::create(element::f32, Shape{batch_size, hidden_size}, {0});
        auto C = op::v0::Constant::create(element::f32, Shape{batch_size, hidden_size}, {0});
        auto Y = op::v0::Constant::create(element::f32, Shape{num_sequences, batch_size, hidden_size}, {0});

        auto loop = std::make_shared<op::v5::Loop>(trip_count, condition);

        auto X_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_sequences, batch_size, input_size});
        auto Y_body = std::make_shared<op::v0::Parameter>(
            element::f32,
            PartialShape{static_cast<int64_t>(num_sequences), static_cast<int64_t>(batch_size), -1});
        auto C_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
        auto sequence_index_body = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
        auto H_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, hidden_size});
        auto iteration_counter_body = std::make_shared<op::v0::Parameter>(element::i32, Shape{});
        auto iteration_counter_step = op::v0::Constant::create(element::i32, Shape{}, {1});
        auto iteration_counter_incremented =
            std::make_shared<op::v1::Add>(iteration_counter_body, iteration_counter_step);
        auto iteration_counter_limit = op::v0::Constant::create(element::i32, Shape{}, {num_sequences});
        auto iteration_counter_less_than_limit =
            std::make_shared<op::v1::Less>(iteration_counter_incremented, iteration_counter_limit);
        auto sequence_index_step = op::v0::Constant::create(element::i32, Shape{}, {1});
        auto sequence_index_incremented = std::make_shared<op::v1::Add>(sequence_index_body, sequence_index_step);
        auto sequence_index_limit = op::v0::Constant::create(element::i32, Shape{}, {num_sequences});
        auto sequence_index_less_than_limit =
            std::make_shared<op::v1::Less>(sequence_index_incremented, sequence_index_limit);
        auto output_condition =
            std::make_shared<op::v1::LogicalAnd>(iteration_counter_less_than_limit, sequence_index_less_than_limit);
        auto condition_result = std::make_shared<op::v0::Result>(output_condition);
        auto Y_shape = std::make_shared<op::v3::ShapeOf>(Y_body);
        auto zero = op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto max_sequence_length = std::make_shared<op::v8::Gather>(Y_shape, zero, zero);
        auto sequence_index_new_shape = op::v0::Constant::create(element::i32, Shape{0}, {});
        std::shared_ptr<Node> gather_index;
        if (with_gather_reshape) {
            gather_index = std::make_shared<op::v1::Reshape>(sequence_index_body, sequence_index_new_shape, false);
        } else {
            gather_index = sequence_index_body;
        }
        auto gather_axis = op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto X_slice = std::make_shared<op::v8::Gather>(X_body, gather_index, gather_axis);
        std::vector<float> W_values(4 * hidden_size * input_size);
        std::iota(W_values.begin(), W_values.end(), 0.0f);
        auto W = op::v0::Constant::create(element::f32, Shape{4 * hidden_size, input_size}, W_values);
        std::vector<float> R_values(4 * hidden_size * hidden_size);
        std::iota(R_values.begin(), R_values.end(), 0.0f);
        auto R = op::v0::Constant::create(element::f32, Shape{4 * hidden_size, hidden_size}, R_values);
        std::vector<float> B_values(4 * hidden_size);
        std::iota(B_values.begin(), B_values.end(), 0.0f);
        auto B = op::v0::Constant::create(element::f32, Shape{4 * hidden_size}, B_values);
        auto lstm_cell = std::make_shared<op::v4::LSTMCell>(X_slice,
                                                            H_body,
                                                            C_body,
                                                            W,
                                                            R,
                                                            B,
                                                            hidden_size,
                                                            std::vector<std::string>{"sigmoid", "tanh", "tanh"});
        auto Y_new_shape2 = std::make_shared<op::v3::ShapeOf>(lstm_cell->output(0));
        auto Y_new_shape = std::make_shared<op::v0::Concat>(OutputVector{max_sequence_length, Y_new_shape2}, 0);
        auto Y_broadcasted = std::make_shared<op::v3::Broadcast>(Y_body, Y_new_shape);
        auto sequence_index_new_shape2 = op::v0::Constant::create(element::i64, Shape{1}, {-1});
        auto scatter_update_index =
            std::make_shared<op::v1::Reshape>(sequence_index_body, sequence_index_new_shape2, false);
        auto H_unsqueezed = std::make_shared<op::v0::Unsqueeze>(lstm_cell->output(0), zero);
        auto scatter_update_body =
            std::make_shared<op::v3::ScatterUpdate>(Y_broadcasted, scatter_update_index, H_unsqueezed, zero);
        auto Y_result = std::make_shared<op::v0::Result>(scatter_update_body);
        auto Co_result = std::make_shared<op::v0::Result>(lstm_cell->output(1));
        auto sequence_index_result = std::make_shared<op::v0::Result>(sequence_index_incremented);
        auto Ho_result = std::make_shared<op::v0::Result>(lstm_cell->output(0));
        auto iteration_counter_result = std::make_shared<op::v0::Result>(iteration_counter_incremented);

        ParameterVector params{X_body, H_body, C_body, Y_body, sequence_index_body, iteration_counter_body};
        ResultVector results{Y_result,
                             Ho_result,
                             Co_result,
                             sequence_index_result,
                             iteration_counter_result,
                             condition_result};
        auto body = std::make_shared<Model>(results, params);
        loop->set_function(body);

        loop->set_invariant_input(Y_body, Y);
        loop->get_iter_value(Y_result, -1);
        loop->set_merged_input(iteration_counter_body, iteration_counter, iteration_counter_result);
        loop->set_merged_input(H_body, H, Ho_result);
        loop->set_merged_input(sequence_index_body, sequence_index, sequence_index_result);
        loop->set_merged_input(C_body, C, Co_result);
        loop->set_invariant_input(X_body, scatter);
        loop->set_special_body_ports({-1, 5});
        auto transpose =
            std::make_shared<op::v1::Transpose>(loop->output(0),
                                                op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2}));

        model = std::make_shared<Model>(transpose, ParameterVector{X});

        manager.register_pass<pass::ConvertTensorIteratorToSequence>();
    }

    {
        auto perm = op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
        std::shared_ptr<op::v0::Parameter> X;
        std::shared_ptr<Node> X_lstm;
        if (with_input_transpose) {
            // fused subgraph doesn't have Transpose
            X = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            X_lstm = X;
        } else {
            X = std::make_shared<op::v0::Parameter>(element::f32, Shape{num_sequences, batch_size, input_size});
            X_lstm = std::make_shared<op::v1::Transpose>(X, perm);
        }
        auto one = op::v0::Constant::create(element::i32, Shape{1}, {1});
        auto zero = op::v0::Constant::create(element::i32, Shape{1}, {0});
        auto shapeof_X = std::make_shared<op::v3::ShapeOf>(X_lstm);
        auto batch_size_node = std::make_shared<op::v8::Gather>(shapeof_X, zero, zero);
        auto H = op::v0::Constant::create(element::f32, Shape{batch_size, hidden_size}, {0});
        auto new_H_shape =
            std::make_shared<op::v0::Concat>(OutputVector{batch_size_node, std::make_shared<op::v3::ShapeOf>(H)}, 0);
        auto H_broadcasted = std::make_shared<op::v3::Broadcast>(H, new_H_shape);
        auto C = op::v0::Constant::create(element::f32, Shape{batch_size, hidden_size}, {0});
        auto new_C_shape =
            std::make_shared<op::v0::Concat>(OutputVector{batch_size_node, std::make_shared<op::v3::ShapeOf>(C)}, 0);
        auto C_broadcasted = std::make_shared<op::v3::Broadcast>(C, new_C_shape);

        std::vector<float> W_values(4 * hidden_size * input_size);
        std::iota(W_values.begin(), W_values.end(), 0.0f);
        std::vector<float> R_values(4 * hidden_size * hidden_size);
        std::iota(R_values.begin(), R_values.end(), 0.0f);
        std::vector<float> B_values(4 * hidden_size);
        std::iota(B_values.begin(), B_values.end(), 0.0f);
        auto W = std::make_shared<op::v0::Unsqueeze>(
            op::v0::Constant::create(element::f32, Shape{4 * hidden_size, input_size}, W_values),
            zero);
        auto R = std::make_shared<op::v0::Unsqueeze>(
            op::v0::Constant::create(element::f32, Shape{4 * hidden_size, hidden_size}, R_values),
            zero);
        auto B = std::make_shared<op::v0::Unsqueeze>(
            op::v0::Constant::create(element::f32, Shape{4 * hidden_size}, B_values),
            zero);

        auto sequence_lengths = op::v0::Constant::create(element::i32, Shape{1}, {num_sequences});
        auto lstm = std::make_shared<op::v5::LSTMSequence>(X_lstm,
                                                           H_broadcasted,
                                                           C_broadcasted,
                                                           sequence_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           hidden_size,
                                                           op::v5::LSTMSequence::direction::FORWARD,
                                                           std::vector<float>{},
                                                           std::vector<float>{},
                                                           std::vector<std::string>{"sigmoid", "tanh", "tanh"});
        auto Ho_squeezed = std::make_shared<op::v0::Squeeze>(lstm->output(0), one);

        model_ref = std::make_shared<Model>(Ho_squeezed, ParameterVector{X});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(ConvertLoopToLSTMSequence,
                         ConvertLoopToLSTMSequenceTest,
                         testing::Combine(testing::Values(false, true), testing::Values(false, true)));

class FuseReverseLSTMSequenceTest : public TransformationTestsF, public testing::WithParamInterface<bool> {};

TEST_P(FuseReverseLSTMSequenceTest, FusionTest) {
    const auto with_input_transpose = GetParam();

    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t num_sequences = 5;
    size_t batch_size = 1;

    {
        std::shared_ptr<op::v0::Parameter> input;
        std::shared_ptr<op::v1::Transpose> second_transpose;
        if (with_input_transpose) {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            auto input_transpose =
                std::make_shared<op::v1::Transpose>(input, op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2}));
            auto input_reverse = std::make_shared<op::v0::ReverseSequence>(
                input_transpose,
                op::v0::Constant::create(element::i32, Shape{1}, {num_sequences}),
                1,
                0);
            second_transpose =
                std::make_shared<op::v1::Transpose>(input_reverse,
                                                    op::v0::Constant::create(element::i32, Shape{3}, {1, 0, 2}));
        } else {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size, num_sequences});
            auto input_reverse = std::make_shared<op::v0::ReverseSequence>(
                input,
                op::v0::Constant::create(element::i32, Shape{1}, {num_sequences}),
                0,
                2);
            second_transpose =
                std::make_shared<op::v1::Transpose>(input_reverse,
                                                    op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
        }
        auto H = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {1.0f});
        auto C = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {2.0f});
        auto sequence_lengths = op::v0::Constant::create(element::i32, Shape{1}, {num_sequences});
        auto W = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {3.0f});
        auto R = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {4.0f});
        auto B = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {5.0f});
        auto lstm = std::make_shared<op::v5::LSTMSequence>(second_transpose,
                                                           H,
                                                           C,
                                                           sequence_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           hidden_size,
                                                           op::v5::LSTMSequence::direction::FORWARD);
        auto squeeze =
            std::make_shared<op::v0::Squeeze>(lstm->output(0), op::v0::Constant::create(element::i32, Shape{}, {1}));
        auto output_reverse =
            std::make_shared<op::v0::ReverseSequence>(squeeze,
                                                      op::v0::Constant::create(element::i32, Shape{1}, {num_sequences}),
                                                      0,
                                                      1);

        model = std::make_shared<Model>(output_reverse, ParameterVector{input});

        manager.register_pass<ov::pass::FuseReverseLSTMSequence>();
    }

    {
        std::shared_ptr<op::v0::Parameter> input;
        std::shared_ptr<Node> lstm_input;
        if (with_input_transpose) {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            lstm_input = input;
        } else {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size, num_sequences});
            lstm_input =
                std::make_shared<op::v1::Transpose>(input, op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
        }
        auto H = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {1.0f});
        auto C = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {2.0f});
        auto sequence_lengths = op::v0::Constant::create(element::i32, Shape{1}, {num_sequences});
        auto W = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {3.0f});
        auto R = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {4.0f});
        auto B = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {5.0f});
        auto lstm = std::make_shared<op::v5::LSTMSequence>(lstm_input,
                                                           H,
                                                           C,
                                                           sequence_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           hidden_size,
                                                           op::v5::LSTMSequence::direction::REVERSE);
        auto squeeze =
            std::make_shared<op::v0::Squeeze>(lstm->output(0), op::v0::Constant::create(element::i32, Shape{}, {1}));

        model_ref = std::make_shared<Model>(squeeze, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(FuseReverseLSTMSequence, FuseReverseLSTMSequenceTest, testing::Values(false, true));

class FuseLSTMSequencesToBidirectionalLSTMSequenceTest : public TransformationTestsF,
                                                         public testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(FuseLSTMSequencesToBidirectionalLSTMSequenceTest, FusionTest) {
    const auto& params = GetParam();
    bool with_input_transpose = std::get<0>(params);
    bool const_sequence_lengths = std::get<1>(params);

    size_t input_size = 3;
    size_t hidden_size = 2;
    size_t num_sequences = 5;
    size_t batch_size = 1;

    {
        std::shared_ptr<op::v0::Parameter> input;
        std::shared_ptr<Node> forward_lstm_input;
        std::shared_ptr<Node> reverse_lstm_input;
        std::shared_ptr<Node> forward_sequence_lengths;
        std::shared_ptr<Node> reverse_sequence_lengths;
        if (with_input_transpose) {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size, num_sequences});
            forward_lstm_input =
                std::make_shared<op::v1::Transpose>(input, op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
            reverse_lstm_input =
                std::make_shared<op::v1::Transpose>(input, op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
        } else {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            forward_lstm_input = input;
            reverse_lstm_input = input;
        }
        if (const_sequence_lengths) {
            forward_sequence_lengths = op::v0::Constant::create(element::i32, Shape{batch_size}, {num_sequences});
            reverse_sequence_lengths = op::v0::Constant::create(element::i32, Shape{batch_size}, {num_sequences});
        } else {
            auto shapeof_forward = std::make_shared<op::v3::ShapeOf>(forward_lstm_input);
            auto gather_forward =
                std::make_shared<op::v8::Gather>(shapeof_forward,
                                                 op::v0::Constant::create(element::i32, Shape{1}, {0}),
                                                 op::v0::Constant::create(element::i32, Shape{}, {0}));
            forward_sequence_lengths =
                std::make_shared<op::v3::Broadcast>(op::v0::Constant::create(element::i32, Shape{}, {num_sequences}),
                                                    gather_forward);
            auto shapeof_reverse = std::make_shared<op::v3::ShapeOf>(reverse_lstm_input);
            auto gather_reverse =
                std::make_shared<op::v8::Gather>(shapeof_reverse,
                                                 op::v0::Constant::create(element::i32, Shape{1}, {0}),
                                                 op::v0::Constant::create(element::i32, Shape{}, {0}));
            reverse_sequence_lengths =
                std::make_shared<op::v3::Broadcast>(op::v0::Constant::create(element::i32, Shape{}, {num_sequences}),
                                                    gather_reverse);
        }
        auto H_forward = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {1.0f});
        auto C_forward = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {2.0f});
        auto W_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {3.0f});
        auto R_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {4.0f});
        auto B_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {5.0f});
        auto lstm_forward = std::make_shared<op::v5::LSTMSequence>(forward_lstm_input,
                                                                   H_forward,
                                                                   C_forward,
                                                                   forward_sequence_lengths,
                                                                   W_forward,
                                                                   R_forward,
                                                                   B_forward,
                                                                   hidden_size,
                                                                   op::v5::LSTMSequence::direction::FORWARD);
        auto squeeze_forward = std::make_shared<op::v0::Squeeze>(lstm_forward->output(0),
                                                                 op::v0::Constant::create(element::i32, Shape{}, {1}));

        auto H_reverse = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {6.0f});
        auto C_reverse = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {7.0f});
        auto W_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {8.0f});
        auto R_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {9.0f});
        auto B_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {10.0f});
        auto lstm_reverse = std::make_shared<op::v5::LSTMSequence>(reverse_lstm_input,
                                                                   H_reverse,
                                                                   C_reverse,
                                                                   reverse_sequence_lengths,
                                                                   W_reverse,
                                                                   R_reverse,
                                                                   B_reverse,
                                                                   hidden_size,
                                                                   op::v5::LSTMSequence::direction::REVERSE);
        auto squeeze_reverse = std::make_shared<op::v0::Squeeze>(lstm_reverse->output(0),
                                                                 op::v0::Constant::create(element::i32, Shape{}, {1}));

        auto concat = std::make_shared<op::v0::Concat>(OutputVector{squeeze_forward, squeeze_reverse}, 2);
        model = std::make_shared<Model>(concat, ParameterVector{input});

        manager.register_pass<ov::pass::FuseLSTMSequencesToBidirectionalLSTMSequence>();
    }

    {
        std::shared_ptr<op::v0::Parameter> input;
        std::shared_ptr<Node> lstm_input;
        std::shared_ptr<Node> sequence_lengths;
        if (with_input_transpose) {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, input_size, num_sequences});
            lstm_input =
                std::make_shared<op::v1::Transpose>(input, op::v0::Constant::create(element::i32, Shape{3}, {0, 2, 1}));
        } else {
            input = std::make_shared<op::v0::Parameter>(element::f32, Shape{batch_size, num_sequences, input_size});
            lstm_input = input;
        }
        if (const_sequence_lengths) {
            sequence_lengths = op::v0::Constant::create(element::i32, Shape{batch_size}, {num_sequences});
        } else {
            auto shapeof = std::make_shared<op::v3::ShapeOf>(lstm_input);
            auto gather = std::make_shared<op::v8::Gather>(shapeof,
                                                           op::v0::Constant::create(element::i32, Shape{1}, {0}),
                                                           op::v0::Constant::create(element::i32, Shape{}, {0}));
            sequence_lengths =
                std::make_shared<op::v3::Broadcast>(op::v0::Constant::create(element::i32, Shape{}, {num_sequences}),
                                                    gather);
        }
        auto H_forward = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {1.0f});
        auto H_reverse = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {6.0f});
        auto H = std::make_shared<op::v0::Concat>(OutputVector{H_forward, H_reverse}, 1);
        auto C_forward = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {2.0f});
        auto C_reverse = op::v0::Constant::create(element::f32, Shape{batch_size, 1, hidden_size}, {7.0f});
        auto C = std::make_shared<op::v0::Concat>(OutputVector{C_forward, C_reverse}, 1);
        auto W_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {3.0f});
        auto W_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, input_size}, {8.0f});
        auto W = std::make_shared<op::v0::Concat>(OutputVector{W_forward, W_reverse}, 0);
        auto R_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {4.0f});
        auto R_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size, hidden_size}, {9.0f});
        auto R = std::make_shared<op::v0::Concat>(OutputVector{R_forward, R_reverse}, 0);
        auto B_forward = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {5.0f});
        auto B_reverse = op::v0::Constant::create(element::f32, Shape{1, 4 * hidden_size}, {10.0f});
        auto B = std::make_shared<op::v0::Concat>(OutputVector{B_forward, B_reverse}, 0);

        auto lstm = std::make_shared<op::v5::LSTMSequence>(lstm_input,
                                                           H,
                                                           C,
                                                           sequence_lengths,
                                                           W,
                                                           R,
                                                           B,
                                                           hidden_size,
                                                           op::v5::LSTMSequence::direction::BIDIRECTIONAL);
        auto transpose =
            std::make_shared<op::v1::Transpose>(lstm->output(0),
                                                op::v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        auto reshape = std::make_shared<op::v1::Reshape>(transpose,
                                                         op::v0::Constant::create(element::i32, Shape{3}, {0, 0, -1}),
                                                         true);
        model_ref = std::make_shared<Model>(reshape, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

INSTANTIATE_TEST_SUITE_P(FuseLSTMSequencesToBidirectionalLSTMSequence,
                         FuseLSTMSequencesToBidirectionalLSTMSequenceTest,
                         testing::Combine(testing::Values(false, true), testing::Values(false, true)));
