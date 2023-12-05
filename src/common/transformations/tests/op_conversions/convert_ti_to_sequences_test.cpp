// Copyright (C) 2018-2023 Intel Corporation
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
