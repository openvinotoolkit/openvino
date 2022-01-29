// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <transformations/init_node_info.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_sequences_to_sequences_ie.hpp>

#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/function.hpp>
#include <legacy/ngraph_ops/gru_sequence_ie.hpp>
#include <legacy/ngraph_ops/rnn_sequence_ie.hpp>
#include <legacy/ngraph_ops/lstm_sequence_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, GRUSequenceConversionTest) {
    std::shared_ptr<ngraph::opset5::GRUSequence> sequence;

    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    const size_t num_directions = 1;
    {
        const auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 1, input_size});
        const auto W =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions, gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions, gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions, gates_count * hidden_size});

        const auto seq_len = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i32, ngraph::Shape{batch_size});
        sequence = std::make_shared<ngraph::opset5::GRUSequence>(X, H_t, seq_len, W, R, B, hidden_size,
                                                                 ngraph::op::RecurrentSequenceDirection::FORWARD);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence}, ngraph::ParameterVector{X, H_t});
        manager.register_pass<ngraph::pass::ConvertGRUSequenceMatcher>();
    }

    {
        const auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 1, input_size});
        const auto W =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions, gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions, gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions, gates_count * hidden_size});

        const auto seq_len = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i32, ngraph::Shape{batch_size}, 1);
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(H_t, axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector({W, R}), 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(B, axis_2);
        auto sequence_ie = std::make_shared<ngraph::op::GRUSequenceIE>(X,
                                                                       in_1,
                                                                       seq_len,   // this input is not supported
                                                                       in_3,
                                                                       in_4,
                                                                       sequence->get_hidden_size(),
                                                                       sequence->get_direction(),
                                                                       sequence->get_activations(),
                                                                       sequence->get_activations_alpha(),
                                                                       sequence->get_activations_beta(),
                                                                       sequence->get_clip(),
                                                                       sequence->get_linear_before_reset());

        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(1), unsqueeze_axis);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze_1}, ngraph::ParameterVector{X, H_t});
    }
}

TEST_F(TransformationTestsF, RNNSequenceConversionTest) {
    const size_t hidden_size = 3;
    const size_t num_directions = 1;
    const size_t batch_size = 2;
    std::shared_ptr<ngraph::opset5::RNNSequence> sequence;

    {
        auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, 1, 3});
        auto H = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, num_directions, 3});
        auto W = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3, 3});
        auto R = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3, 3});
        auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3});
        auto seq_len = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{2});
        sequence = std::make_shared<ngraph::opset5::RNNSequence>(X, H, seq_len, W, R, B, hidden_size,
                                                                 ngraph::op::RecurrentSequenceDirection::FORWARD);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence}, ngraph::ParameterVector{X, H});
        manager.register_pass<ngraph::pass::ConvertRNNSequenceMatcher>();
    }

    {
        auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, 1, 3});
        auto H = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{batch_size, num_directions, 3});
        auto W = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3, 3});
        auto R = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3, 3});
        auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{num_directions, 3});
        auto seq_len = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32, ngraph::Shape{batch_size}, 1);
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(H, axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector({W, R}), 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(B, axis_2);
        auto sequence_ie = std::make_shared<ngraph::op::RNNSequenceIE>(X,
                                                                       in_1,
                                                                       seq_len,
                                                                       in_3,
                                                                       in_4,
                                                                       sequence->get_hidden_size(),
                                                                       sequence->get_direction(),
                                                                       sequence->get_activations(),
                                                                       sequence->get_activations_alpha(),
                                                                       sequence->get_activations_beta(),
                                                                       sequence->get_clip());

        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(1), unsqueeze_axis);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze_1}, ngraph::ParameterVector{X, H});
    }
}

TEST_F(TransformationTestsF, LSTMSequenceConversionTest) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const size_t num_directions = 1;
    std::shared_ptr<ngraph::opset5::LSTMSequence> sequence;
    {
        const auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 10, input_size});
        const auto W =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions,
                                                                                   hidden_size});
        const auto C_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions,
                                                                                gates_count * hidden_size});

        const auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{batch_size});
        sequence = std::make_shared<ngraph::op::v5::LSTMSequence>(X, H_t, C_t, seq_len, W, R, B, hidden_size,
                                                                  ngraph::op::RecurrentSequenceDirection::FORWARD);
        sequence->set_friendly_name("test_sequence");

        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{sequence->output(0)}, ngraph::ParameterVector{X, H_t, C_t});
        manager.register_pass<ngraph::pass::ConvertLSTMSequenceMatcher>();
    }

    {
        const auto X = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 10, input_size});
        const auto W =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto seq_lengths = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size});
        const auto B = std::make_shared<ngraph::opset5::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions,
                                                                                gates_count * hidden_size});
        // const auto seq_len = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
        auto axis_1 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto in_1 = std::make_shared<ngraph::opset5::Squeeze>(H_t, axis_1);
        auto in_2 = std::make_shared<ngraph::opset5::Squeeze>(C_t, axis_1);
        auto concat = std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector({W, R}), 2);
        auto axis_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto in_3 = std::make_shared<ngraph::opset5::Squeeze>(concat->output(0), axis_2);
        auto in_4 = std::make_shared<ngraph::opset5::Squeeze>(B, axis_2);
        auto sequence_ie = std::make_shared<ngraph::op::LSTMSequenceIE>(X,
                                                                        in_1,
                                                                        in_2,
                                                                        seq_lengths,
                                                                        in_3,
                                                                        in_4,
                                                                        sequence->get_hidden_size(),
                                                                        sequence->get_direction(),
                                                                        sequence->get_activations(),
                                                                        sequence->get_activations_alpha(),
                                                                        sequence->get_activations_beta(),
                                                                        sequence->get_clip());
        auto unsqueeze_axis = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto unsqueeze_1 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(0), unsqueeze_axis);
        auto unsqueeze_2 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(1), unsqueeze_axis);
        auto unsqueeze_3 = std::make_shared<ngraph::opset5::Unsqueeze>(sequence_ie->output(2), unsqueeze_axis);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze_1},
                                                   ngraph::ParameterVector{X, H_t, C_t});
    }
}
