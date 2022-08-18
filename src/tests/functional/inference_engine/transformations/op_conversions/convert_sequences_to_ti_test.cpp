// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>
#include <iomanip>

#include <ngraph/pass/manager.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

#include "openvino/opsets/opset9.hpp"
#include "openvino/openvino.hpp"

#include "ngraph/type/bfloat16.hpp"

#include "ngraph_ops/augru_cell.hpp"
#include "ngraph_ops/augru_sequence.hpp"

using namespace testing;
using namespace ngraph;
using namespace std;

TEST(TransformationTests, ConvertLSTMSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 2, 16 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512 }, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X, Y, Z, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho, Co }, ngraph::ParameterVector{ X, Y, Z });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 2, 16 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 16 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 128 });
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512 }, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho, Co }, ParameterVector{ Xi, Yi, Zi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H, res_ti_C }, ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertLSTMSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 512 }, b_val);

        auto rnn_sequence = std::make_shared<opset5::LSTMSequence>(X, Y, Z, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        auto Co = std::make_shared<opset5::Result>(rnn_sequence->output(2));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho, Co }, ngraph::ParameterVector{ X, Y, Z });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertLSTMSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto Z = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);
        auto squeeze_z = std::make_shared<opset5::Squeeze>(Z, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 1, -1 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 128 });
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 512 }, b_val);

        auto rnn_cell = std::make_shared<opset5::LSTMCell>(squeeze_x, Yi, Zi, W, R, B, 128);

        auto Ho = std::make_shared<opset5::Result>(rnn_cell->output(0));

        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto unsqueeze_y = std::make_shared<opset5::Unsqueeze>(rnn_cell->output(0), unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze_y);

        auto Co = std::make_shared<opset5::Result>(rnn_cell->output(1));

        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho, Co }, ParameterVector{ Xi, Yi, Zi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        tensor_iterator->set_merged_input(Zi, squeeze_z, Co);

        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);
        tensor_iterator->get_iter_value(Co);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        auto res_ti_C = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(2), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");
        res_ti_C->set_friendly_name("Co");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H, res_ti_C }, ngraph::ParameterVector{ X, Y, Z });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 128});
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{1, 128}, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X, Y, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{X, Y});


        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 2, 16 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 16 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128 }, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho }, ParameterVector{ Xi, Yi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertRNNSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 128, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 128, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 128 }, b_val);

        auto rnn_sequence = std::make_shared<opset5::RNNSequence>(X, Y, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertRNNSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, axis_1);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 1, -1 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, axis_1);

        auto w_val = std::vector<float>(128 * 16, 0);
        auto r_val = std::vector<float>(128 * 128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 128 }, b_val);

        auto rnn_cell = std::make_shared<opset5::RNNCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, axis_1);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho }, ParameterVector{ Xi, Yi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), axis_1));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), axis_1));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIterator) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 2, 16 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384 }, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X, Y, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 2, 16 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 128 });
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 1, 16 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384 }, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho }, ParameterVector{ Xi, Yi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGRUSequenceToTensorIteratorDynamic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto seq_lengths = ngraph::opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 384 }, b_val);

        auto rnn_sequence = std::make_shared<opset5::GRUSequence>(X, Y, seq_lengths, W, R, B, 128, op::RecurrentSequenceDirection::FORWARD);
        auto Y_out = std::make_shared<opset5::Result>(rnn_sequence->output(0));
        auto Ho = std::make_shared<opset5::Result>(rnn_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertGRUSequenceToTensorIterator>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 2, -1 });
        auto Y = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 1, 128 });
        auto squeeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto squeeze_y = std::make_shared<opset5::Squeeze>(Y, squeeze_pattern);

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ -1, 1, -1 });
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, PartialShape{ 1, 128 });
        auto seq_body_param = std::make_shared<opset5::Parameter>(element::i32, PartialShape{ 1 });

        // Body
        auto squeeze_x = std::make_shared<opset5::Squeeze>(Xi, squeeze_pattern);

        auto w_val = std::vector<float>(384 * 16, 0);
        auto r_val = std::vector<float>(384 * 128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384, 16 }, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384, 128 }, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{ 384 }, b_val);

        auto rnn_cell = std::make_shared<opset5::GRUCell>(squeeze_x, Yi, W, R, B, 128);
        auto Ho = std::make_shared<opset5::Result>(rnn_cell);
        auto unsqueeze_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });
        auto unsqueeze = std::make_shared<opset5::Unsqueeze>(rnn_cell, unsqueeze_pattern);
        auto Y_out = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{ Y_out, Ho }, ParameterVector{ Xi, Yi, seq_body_param });

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->get_concatenated_slices(Y_out, 0, 1, 1, -1, 1);

        tensor_iterator->set_merged_input(Yi, squeeze_y, Ho);
        auto seq_lengths = opset5::Constant::create(element::i32, Shape{ 1 }, { 2 });
        tensor_iterator->set_invariant_input(seq_body_param, seq_lengths);

        tensor_iterator->get_iter_value(Ho);

        auto res_ti_Y = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(0), unsqueeze_pattern));
        auto res_ti_H = std::make_shared<opset5::Result>(std::make_shared<opset5::Unsqueeze>(tensor_iterator->output(1), unsqueeze_pattern));
        res_ti_Y->set_friendly_name("Y_out");
        res_ti_H->set_friendly_name("Ho");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ res_ti_Y, res_ti_H }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

template<typename T>
void print(const T* data, int size, std::string name) {
    std::cout << name << std::endl;
    std::cout << std::setprecision(6);
    std::string end;
    for (int i = 0; i < size; i++) {
        if (i != 0 && i%10 == 0) {
            end = ",\n";
        } else {
            end = ", ";
        }
        std::cout << data[i] << end;
    }
    std::cout << std::endl;
}

// TEST(TransformationTests, xxxxxxxx) {
//     std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//     {
//         uint64_t batch_size = 5;
//         uint64_t input_size = 10;
//         uint64_t hidden_size = 10;
//         uint64_t seq_length = 10;
//         uint64_t num_directions = 1;
//         auto et = element::f32;

//         auto X_val = std::vector<float>{
//                 1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
//                 8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
//                 8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
//                 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
//                 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
//                 5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
//                 3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
//                 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
//                 2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
//                 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
//                 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
//                 1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
//                 2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
//                 1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
//                 4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
//                 2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
//                 7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
//                 5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
//                 2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
//                 6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
//                 4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
//                 4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
//                 5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
//                 9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
//                 8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
//                 6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
//                 7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
//                 1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
//                 1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
//                 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 1.39976,
//                 3.53095, 8.78581, 1.65811, 6.94299, 2.68641, 5.70058, 9.13491, 5.27637, 8.6232, 8.54902,
//                 2.25352, 5.86274, 5.20377, 2.96815, 4.96745, 5.3225, 3.99956, 1.08021, 5.54918, 7.05833,
//                 1.49913, 2.41822, 6.44593, 3.87301, 9.01465, 8.11336, 2.95749, 2.80188, 7.12396, 2.40595,
//                 5.59325, 9.89258, 2.30223, 1.4347, 9.09158, 7.43797, 3.79295, 4.53646, 1.72705, 4.16909,
//                 1.00912, 6.62167, 2.80244, 6.626, 3.89307, 1.42586, 7.51028, 7.83327, 4.65674, 7.33902,
//                 6.26823, 9.72608, 3.73491, 3.8238, 3.03815, 7.05101, 8.0103, 5.61396, 6.53738, 1.41095,
//                 5.0149, 9.71211, 4.23604, 5.98629, 4.70219, 9.69442, 2.82752, 9.93544, 6.9328, 8.2817,
//                 5.12336, 8.98577, 5.80541, 6.19552, 9.25748, 3.82732, 7.53525, 8.24712, 5.32057, 5.38817,
//                 8.57269, 5.99975, 3.42893, 5.38068, 3.48261, 3.02851, 6.82079, 9.2902, 2.80427, 8.91868,
//                 5.19227, 7.52482, 3.72584, 5.40107, 2.83307, 1.79755, 2.49121, 5.52697, 8.08823, 9.13242,
//                 2.97572, 7.64318, 3.32023, 6.07788, 2.19187, 4.34879, 1.7457, 5.55154, 7.24966, 5.1128,
//                 4.25147, 8.34407, 1.4123, 4.49045, 5.12671, 7.62159, 9.18673, 3.49665, 8.35992, 6.90684,
//                 1.10152, 7.61818, 6.43145, 7.12017, 6.25564, 6.16169, 4.24916, 9.6283, 9.88249, 4.48422,
//                 8.52562, 9.83928, 6.26818, 7.03839, 1.77631, 9.92305, 8.0155, 9.94928, 6.88321, 1.33685,
//                 7.4718, 7.19305, 6.47932, 1.9559, 3.52616, 7.98593, 9.0115, 5.59539, 7.44137, 1.70001,
//                 6.53774, 8.54023, 7.26405, 5.99553, 8.75071, 7.70789, 3.38094, 9.99792, 6.16359, 6.75153,
//                 5.4073, 9.00437, 8.87059, 8.63011, 6.82951, 6.27021, 3.53425, 9.92489, 8.19695, 5.51473,
//                 7.95084, 2.11852, 9.28916, 1.40353, 3.05744, 8.58238, 3.75014, 5.35889, 6.85048, 2.29549,
//                 3.75218, 8.98228, 8.98158, 5.63695, 3.40379, 8.92309, 5.48185, 4.00095, 9.05227, 2.84035,
//                 8.37644, 8.54954, 5.70516, 2.45744, 9.54079, 1.53504, 8.9785, 6.1691, 4.40962, 10};


//         auto H_t_val = std::vector<float>{1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
//                 8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
//                 8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
//                 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
//                 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 10};


//         auto s_len_val = std::vector<int64_t>{10, 10, 10, 10, 10};


//         auto w_val = std::vector<float>{1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
//                 8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
//                 8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
//                 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
//                 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
//                 5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
//                 3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
//                 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
//                 2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
//                 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
//                 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
//                 1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
//                 2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
//                 1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
//                 4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
//                 2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
//                 7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
//                 5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
//                 2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
//                 6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
//                 4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
//                 4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
//                 5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
//                 9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
//                 8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
//                 6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
//                 7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
//                 1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
//                 1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
//                 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 10};


//         auto r_val = std::vector<float>{1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
//                 8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
//                 8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 1.53319,
//                 7.03475, 6.33759, 7.04489, 4.70609, 2.77796, 3.60667, 2.27908, 8.04983, 4.71285, 1.30754,
//                 6.61627, 6.94572, 3.68646, 5.01521, 2.99912, 1.66028, 5.22315, 1.86555, 9.13033, 2.07541,
//                 5.72319, 1.75261, 9.25175, 9.19404, 3.69037, 6.2595, 6.09321, 6.52544, 9.60882, 3.34881,
//                 3.07914, 5.80104, 9.54944, 5.43754, 5.8654, 7.88937, 1.40811, 2.2597, 8.13163, 1.26821,
//                 8.94813, 5.86709, 5.03182, 9.02922, 4.39826, 5.84582, 6.87069, 4.25135, 6.13908, 6.74053,
//                 2.13683, 7.21184, 6.82974, 4.18545, 7.8691, 4.20879, 7.77509, 8.93208, 1.10502, 5.48298,
//                 1.66413, 8.08256, 1.57661, 4.19779, 9.47653, 4.41823, 7.86628, 7.94436, 3.71224, 7.95465,
//                 2.37637, 6.20771, 1.08107, 7.38138, 5.23577, 7.88133, 5.20653, 3.42101, 8.48523, 5.96192,
//                 1.63073, 5.25228, 7.68488, 2.7276, 5.1788, 3.07327, 5.57423, 2.87711, 1.44374, 5.66976,
//                 2.55051, 4.56682, 1.96629, 5.58829, 1.91922, 3.59846, 3.08583, 9.70901, 3.50487, 3.1026,
//                 1.82401, 6.1306, 4.76134, 4.31059, 8.31695, 3.60784, 7.45652, 6.51653, 4.84219, 7.76686,
//                 4.85031, 4.85544, 4.25714, 2.38005, 9.43471, 9.24769, 8.03763, 6.54696, 1.32399, 6.88891,
//                 2.16793, 3.64924, 4.24733, 3.47181, 1.66572, 2.36923, 2.45457, 9.44841, 4.34021, 1.45016,
//                 7.6686, 3.68812, 2.83922, 9.83581, 9.03719, 7.83414, 6.86009, 1.35715, 8.32489, 7.86316,
//                 5.09754, 5.78644, 1.98402, 2.31429, 5.5791, 2.94085, 9.24799, 5.15937, 2.19041, 7.87817,
//                 2.9146, 1.66833, 1.85877, 2.45985, 4.20817, 1.85777, 2.28937, 9.37912, 6.18926, 8.55681,
//                 6.60963, 3.92066, 7.5521, 5.70463, 7.6313, 2.48866, 7.18352, 4.8413, 7.55702, 7.80702,
//                 4.5785, 9.3268, 2.83159, 1.07202, 9.33716, 3.6506, 2.50256, 1.21691, 5.06801, 8.27505,
//                 4.31539, 6.48286, 1.31363, 4.1912, 1.70668, 7.23867, 1.11441, 5.13591, 9.65186, 4.00767,
//                 5.24875, 1.94852, 5.52768, 8.97121, 5.8094, 3.53329, 4.19126, 9.06652, 3.1734, 1.21496,
//                 9.69154, 4.86971, 4.1166, 6.19361, 2.13874, 9.55039, 3.8225, 9.57548, 2.96554, 3.2383,
//                 8.77422, 3.11741, 8.3359, 5.89508, 2.72134, 6.29956, 1.43323, 1.14286, 1.4474, 4.59474,
//                 6.19214, 8.80766, 8.07546, 3.29232, 1.74029, 2.4198, 2.88544, 4.75644, 4.12921, 7.29896,
//                 7.27759, 1.67252, 1.32823, 8.1046, 9.10476, 1.04197, 3.37783, 5.2064, 4.23835, 3.16196,
//                 1.20852, 5.78501, 2.17175, 6.05313, 2.51048, 4.78967, 7.16219, 3.4651, 1.09, 2.9788,
//                 1.28761, 9.41385, 8.03883, 5.65835, 1.14816, 3.6892, 5.86648, 8.73895, 2.66603, 1.75192,
//                 1.39845, 4.99427, 1.17387, 1.60329, 8.30594, 6.72662, 7.95565, 7.35114, 3.1439, 10};


//         auto b_val = std::vector<float>{1, 9.97466, 9.39302, 2.15312, 9.99136, 3.1248, 4.56923, 4.4912, 7.02771, 9.41985,
//                 8.6168, 3.81946, 5.72093, 4.99108, 3.0662, 5.80973, 9.22566, 5.11484, 4.87629, 9.45215,
//                 8.0055, 7.44373, 8.22482, 1.83521, 5.66337, 8.78518, 8.46232, 8.46643, 3.45745, 10};


//         auto a_val = std::vector<float>{0.034518f, 0.673286f, 0.406714f, 0.252863f, 0.403579f, 0.985528f,
//                 0.195026f, 0.102555f, 0.862911f, 0.698157f, 0.159086f, 0.980120f,
//                 0.821253f, 0.041956f, 0.297002f, 0.154800f, 0.107107f, 0.457263f,
//                 0.512179f, 0.372368f, 0.672125f, 0.078459f, 0.593353f, 0.800342f,
//                 0.245871f, 0.675492f, 0.799024f, 0.432696f, 0.095204f, 0.187363f,
//                 0.715765f, 0.981477f, 0.806237f, 0.544187f, 0.450795f, 0.717673f,
//                 0.426730f, 0.601688f, 0.765923f, 0.675334f, 0.341641f, 0.958286f,
//                 0.137889f, 0.239074f, 0.015245f, 0.148769f, 0.793591f, 0.210904f,
//                 0.973056f, 0.646550f};


//         const auto X = make_shared<ov::opset9::Parameter>(et, Shape{batch_size, seq_length, input_size});
//         const auto H_t = ov::opset9::Constant::create(et, Shape{batch_size, num_directions, hidden_size}, H_t_val);
//         const auto seq_lengths = ov::opset9::Constant::create(element::i64, Shape{batch_size}, s_len_val);
//         const auto W = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, input_size}, w_val);
//         const auto R = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, hidden_size}, r_val);
//         const auto B = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3}, b_val);
//         const auto A = ov::opset9::Constant::create(et, Shape{batch_size, seq_length, 1}, a_val);

//         auto augru_sequence = std::make_shared<ngraph::op::internal::AUGRUSequence>(X, H_t, seq_lengths, W, R, B, A, hidden_size);
//         auto Y_out = std::make_shared<ov::opset9::Result>(augru_sequence->output(0));
//         auto Ho = std::make_shared<ov::opset9::Result>(augru_sequence->output(1));
//         Y_out->set_friendly_name("Y_out");
//         Ho->set_friendly_name("Ho");

//         f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{X});

//         ngraph::pass::Manager m;
//         m.register_pass<ngraph::pass::InitNodeInfo>();
//         m.register_pass<ngraph::pass::ConvertAUGRUSequenceToTensorIterator>();
//         m.run_passes(f);

//         ov::Core core;
//         auto compiled_model = core.compile_model(f, "TEMPLATE");
//         ov::InferRequest infer_request = compiled_model.create_infer_request();

//         ov::Tensor X_input = ov::Tensor(element::f32, Shape{batch_size, seq_length, input_size}, X_val.data());
//         infer_request.set_input_tensor(0, X_input);

//         infer_request.infer();

//         const ov::Tensor& Y_out_tensor = infer_request.get_output_tensor(0);
//         print(Y_out_tensor.data<const float>(), Y_out_tensor.get_size(), "Y_out");

//         const ov::Tensor& Ho_tensor = infer_request.get_output_tensor(1);
//         print(Ho_tensor.data<const float>(), Ho_tensor.get_size(), "Ho");
//     }
// }

TEST(TransformationTests, yyyyyy) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        uint64_t batch_size = 3;
        uint64_t input_size = 4;
        uint64_t hidden_size = 4;
        uint64_t seq_length = 4;
        uint64_t num_directions = 1;
        auto et = element::f32;

        auto X_val = std::vector<float>{
                6.333189f, 5.312533f, 3.092159f, 6.788004f, 6.543326f, 6.307378f,
                7.487966f, 3.300125f, 5.563141f, 5.551623f, 3.812029f, 0.132248f,
                6.767892f, 7.179934f, 4.832786f, 7.047989f, 7.439896f, 5.970488f,
                4.878981f, 7.562621f, 7.069236f, 6.484528f, 3.208064f, 5.561788f,
                3.716249f, 3.412311f, 2.414488f, 7.465214f, 6.444024f, 6.718469f,
                1.158724f, 0.874442f, 3.319976f, 3.639372f, 5.725887f, 6.594659f,
                2.427224f, 7.611652f, 4.804684f, 3.739928f, 1.892036f, 2.038615f,
                1.297785f, 2.687379f, 7.228432f, 3.015881f, 1.658817f, 6.812868f};


        auto H_t_val = std::vector<float>{1.905747f, 1.157613f, 7.828747f, 0.730339f, 0.177484f, 6.356011f,
                                          7.802150f, 0.351901f, 2.614103f, 5.295752f, 1.925256f, 5.925945f};


        auto s_len_val = std::vector<int64_t>{2, 4, 6};


        auto w_val = std::vector<float>{4.092708f, 3.566296f, 1.985790f, 6.146150f, 6.427968f, 7.096002f,
                                        7.786629f, 1.156184f, 2.091389f, 3.677408f, 0.800674f, 2.707813f,
                                        5.396107f, 5.135335f, 3.197786f, 2.983154f, 3.029478f, 1.616802f,
                                        2.162019f, 6.922834f, 5.473023f, 4.648859f, 7.976208f, 0.100666f,
                                        1.022222f, 6.531802f, 4.358081f, 2.619875f, 2.189079f, 4.086223f,
                                        7.818407f, 1.802662f, 7.781939f, 7.542665f, 0.512391f, 6.402292f,
                                        6.450581f, 5.147434f, 5.103659f, 0.894397f, 3.192224f, 3.798631f,
                                        4.647818f, 3.119819f, 6.469874f, 1.386456f, 2.030223f, 6.888086f};


        auto r_val = std::vector<float>{5.094750f, 0.455363f, 4.989539f, 0.998320f, 6.369301f, 6.971411f,
                                        1.183518f, 1.386444f, 7.572741f, 4.089349f, 0.116142f, 3.784642f,
                                        7.925501f, 6.721643f, 3.293674f, 5.754296f, 0.642623f, 2.969238f,
                                        3.536983f, 6.635469f, 1.634170f, 0.507635f, 0.205818f, 0.014995f,
                                        4.528792f, 7.128315f, 2.403315f, 7.728932f, 4.206394f, 6.285463f,
                                        5.314773f, 6.473219f, 2.506818f, 0.430820f, 3.337787f, 3.828441f,
                                        2.620897f, 5.545934f, 2.593279f, 3.601373f, 2.595556f, 0.687849f,
                                        5.277596f, 4.074282f, 1.844880f, 1.232070f, 4.970289f, 5.472321f};


        auto b_val = std::vector<float>{5.516398f, 5.621736f, 4.341954f, 6.820305f, 4.760572f, 7.643595f,
                                        1.733547f, 3.100034f, 7.009072f, 3.027971f, 5.044797f, 0.534897f};


        auto a_val = std::vector<float>{5.472119f, 1.116471f, 4.916850f, 1.217513f, 5.672895f, 4.827186f,
                                        2.210116f, 3.160670f, 4.265747f, 3.949529f, 4.224079f, 3.049556f};


        const auto X = make_shared<ov::opset9::Parameter>(et, Shape{batch_size, seq_length, input_size});
        const auto H_t = ov::opset9::Constant::create(et, Shape{batch_size, num_directions, hidden_size}, H_t_val);
        const auto seq_lengths = ov::opset9::Constant::create(element::i64, Shape{batch_size}, s_len_val);
        const auto W = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, input_size}, w_val);
        const auto R = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, hidden_size}, r_val);
        const auto B = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3}, b_val);
        const auto A = ov::opset9::Constant::create(et, Shape{batch_size, seq_length, 1}, a_val);

        auto augru_sequence = std::make_shared<ngraph::op::internal::AUGRUSequence>(X, H_t, seq_lengths, W, R, B, A, hidden_size);
        auto Y_out = std::make_shared<ov::opset9::Result>(augru_sequence->output(0));
        auto Ho = std::make_shared<ov::opset9::Result>(augru_sequence->output(1));
        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{X});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertAUGRUSequenceToTensorIterator>();
        m.run_passes(f);

        ov::Core core;
        auto compiled_model = core.compile_model(f, "CPU");
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        ov::Tensor X_input = ov::Tensor(element::f32, Shape{batch_size, seq_length, input_size}, X_val.data());
        infer_request.set_input_tensor(0, X_input);

        infer_request.infer();

        const ov::Tensor& Y_out_tensor = infer_request.get_output_tensor(0);
        print(Y_out_tensor.data<const float>(), Y_out_tensor.get_size(), "Y_out");

        const ov::Tensor& Ho_tensor = infer_request.get_output_tensor(1);
        print(Ho_tensor.data<const float>(), Ho_tensor.get_size(), "Ho");
    }
}

// TEST(TransformationTests, zzzzz) {
//     std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//     {
//         uint64_t batch_size = 3;
//         uint64_t input_size = 4;
//         uint64_t hidden_size = 5;
//         uint64_t seq_length = 6;
//         uint64_t num_directions = 1;
//         auto et = element::bf16;

//         auto X_val = std::vector<ov::bfloat16>{
//                 18.351103f, 13.520717f, 11.687705f, 16.378006f, 15.568568f, 2.611258f,
//                 8.754645f, 1.173500f, 19.354236f, 18.344900f, 16.889204f, 15.882624f,
//                 5.627136f, 18.532579f, 3.884619f, 5.119867f, 19.311458f, 13.728893f,
//                 14.391294f, 14.994748f, 1.131725f, 7.582668f, 14.131282f, 19.547387f,
//                 17.038887f, 9.964800f, 0.993434f, 11.316858f, 5.330938f, 17.378582f,
//                 14.467881f, 4.232232f, 1.326875f, 5.338556f, 6.427002f, 7.961141f,
//                 11.043235f, 13.890585f, 15.062217f, 11.441225f, 14.734964f, 17.898548f,
//                 3.364136f, 12.155336f, 13.873233f, 5.822778f, 14.347963f, 3.387873f,
//                 11.021886f, 15.638179f, 15.419064f, 3.743839f, 7.656481f, 9.581622f,
//                 11.851100f, 4.481464f, 2.963150f, 18.103926f, 1.524719f, 6.200507f,
//                 10.349086f, 1.085302f, 9.037059f, 8.240071f, 9.869549f, 13.749188f,
//                 6.019619f, 7.583932f, 9.870675f, 5.238130f, 3.383805f, 6.283896f};


//         auto H_t_val = std::vector<ov::bfloat16>{17.012305f, 0.280625f, 4.584126f, 5.899313f, 17.788775f, 19.869320f,
//                                           17.561679f, 1.209108f, 2.772207f, 15.623852f, 0.212452f, 8.111215f,
//                                           5.552669f, 10.256264f, 0.026503f};


//         auto s_len_val = std::vector<int64_t>{5, 6, 7};


//         auto w_val = std::vector<ov::bfloat16>{2.599543f, 1.017497f, 2.188785f, 9.186879f, 19.503938f, 9.249134f,
//                                         1.613915f, 16.139065f, 16.139263f, 6.162586f, 3.584053f, 13.815329f,
//                                         3.166837f, 9.710348f, 13.262320f, 5.637648f, 7.555024f, 4.282188f,
//                                         0.890288f, 9.009603f, 6.672247f, 10.287107f, 3.250285f, 17.159962f,
//                                         12.805244f, 7.266579f, 6.671443f, 4.756829f, 14.331350f, 4.955475f,
//                                         15.988379f, 1.444935f, 8.917654f, 13.265950f, 14.192596f, 6.813692f,
//                                         6.118729f, 9.446750f, 6.676908f, 2.862558f, 7.298118f, 14.044843f,
//                                         4.047191f, 18.313578f, 14.433683f, 10.046828f, 6.272552f, 5.960752f,
//                                         7.687321f, 17.799588f, 17.974497f, 3.092019f, 17.537077f, 5.673742f,
//                                         7.945506f, 16.820461f, 3.382466f, 14.536867f, 10.431873f, 17.666089f};


//         auto r_val = std::vector<ov::bfloat16>{19.883110f, 3.627721f, 12.381158f, 8.494785f, 0.744855f, 2.083576f,
//                                         3.628426f, 13.435420f, 8.021385f, 16.324589f, 9.171570f, 5.194103f,
//                                         9.534415f, 3.747218f, 14.976561f, 15.450245f, 0.690866f, 14.611735f,
//                                         12.058640f, 12.255162f, 11.922434f, 5.324248f, 16.084200f, 11.910202f,
//                                         8.972001f, 8.331036f, 5.271550f, 3.376512f, 17.947813f, 11.445078f,
//                                         19.813407f, 7.357054f, 18.609516f, 1.641881f, 15.659794f, 12.757155f,
//                                         19.407852f, 15.042537f, 5.609916f, 5.240596f, 16.863105f, 2.939757f,
//                                         7.419171f, 17.463287f, 15.461648f, 3.089222f, 0.349571f, 17.916363f,
//                                         17.127918f, 9.218649f, 16.540375f, 19.735329f, 14.503906f, 2.743288f,
//                                         18.953856f, 7.761779f, 15.795299f, 4.491015f, 16.724311f, 10.091499f,
//                                         8.233435f, 7.727666f, 17.705235f, 10.873172f, 6.631204f, 4.461747f,
//                                         0.832513f, 7.237117f, 12.597311f, 13.430508f, 11.884650f, 3.903417f,
//                                         8.122048f, 14.688200f, 3.733118f};


//         auto b_val = std::vector<ov::bfloat16>{14.387985f, 12.398350f, 12.775042f, 11.106416f, 18.022651f, 2.087257f,
//                                         17.773750f, 15.364595f, 9.153267f, 14.613577f, 11.195361f, 15.070744f,
//                                         1.611076f, 11.896046f, 10.134009f};


//         auto a_val = std::vector<ov::bfloat16>{9.273749f, 12.871767f, 16.940379f, 9.644809f, 8.121616f, 12.174741f,
//                                         11.509982f, 10.122257f, 4.616639f, 13.560133f, 3.175609f, 5.129174f,
//                                         2.827964f, 9.646965f, 9.387886f, 1.972116f, 17.192705f, 6.235506f};


//         const auto X = ov::opset9::Constant::create(et, Shape{batch_size, seq_length, input_size}, X_val.data());
//         const auto H_t = ov::opset9::Constant::create(et, Shape{batch_size, num_directions, hidden_size}, H_t_val);
//         const auto seq_lengths = ov::opset9::Constant::create(element::i64, Shape{batch_size}, s_len_val);
//         const auto W = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, input_size}, w_val);
//         const auto R = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3, hidden_size}, r_val);
//         const auto B = ov::opset9::Constant::create(et, Shape{num_directions, hidden_size * 3}, b_val);
//         const auto A = ov::opset9::Constant::create(et, Shape{batch_size, seq_length, 1}, a_val);

//         auto augru_sequence = std::make_shared<ngraph::op::internal::AUGRUSequence>(X, H_t, seq_lengths, W, R, B, A, hidden_size);
//         auto Y_out = std::make_shared<ov::opset9::Result>(augru_sequence->output(0));
//         auto Ho = std::make_shared<ov::opset9::Result>(augru_sequence->output(1));
//         Y_out->set_friendly_name("Y_out");
//         Ho->set_friendly_name("Ho");

//         f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ Y_out, Ho }, ngraph::ParameterVector{});

//         ngraph::pass::Manager m;
//         m.register_pass<ngraph::pass::InitNodeInfo>();
//         m.register_pass<ngraph::pass::ConvertAUGRUSequenceToTensorIterator>();
//         m.run_passes(f);

//         ov::Core core;
//         auto compiled_model = core.compile_model(f, "TEMPLATE");
//         ov::InferRequest infer_request = compiled_model.create_infer_request();

//         infer_request.infer();

//         const ov::Tensor& Y_out_tensor = infer_request.get_output_tensor(0);
//         print(Y_out_tensor.data<const ov::bfloat16>(), Y_out_tensor.get_size(), "Y_out");

//         const ov::Tensor& Ho_tensor = infer_request.get_output_tensor(1);
//         print(Ho_tensor.data<const ov::bfloat16>(), Ho_tensor.get_size(), "Ho");
//     }
// }