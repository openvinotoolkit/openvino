// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <transformations/init_node_info.hpp>
#include <transformations/convert_opset4_to_opset3/convert_sequences_to_sequences_ie.hpp>

#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/function.hpp>
#include <ngraph_ops/gru_sequence_ie.hpp>
#include <ngraph_ops/rnn_sequence_ie.hpp>
#include <ngraph_ops/lstm_sequence_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, GRUSequenceConversionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    std::shared_ptr<ngraph::opset4::GRUSequence> sequence;

    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    {
        const auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{gates_count * hidden_size});

        const auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
        sequence = std::make_shared<ngraph::opset4::GRUSequence>(X, H_t, seq_len, W, R, B, hidden_size,
                                                                 ngraph::op::RecurrentSequenceDirection::FORWARD);
        sequence->set_friendly_name("test_sequence");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence}, ngraph::ParameterVector{X, H_t});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertGRUSequenceMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        const auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, input_size});
        const auto W =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, hidden_size});
        const auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{gates_count * hidden_size});

        const auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 1);
        auto sequence_ie = std::make_shared<ngraph::op::GRUSequenceIE>(X,
                                                                       H_t,
                                                                       seq_len,
                                                                       concat,
                                                                       B,
                                                                       sequence->get_hidden_size(),
                                                                       sequence->get_direction(),
                                                                       sequence->get_activations(),
                                                                       sequence->get_activations_alpha(),
                                                                       sequence->get_activations_beta(),
                                                                       sequence->get_clip(),
                                                                       sequence->get_linear_before_reset());
        sequence_ie->set_friendly_name("test_sequence");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence_ie}, ngraph::ParameterVector{X, H_t});
    }
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto sequence_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(sequence_node->get_friendly_name() == "test_sequence")
                                << "Transformation ConvertGRUSequenceToGRUSequenceIE should keep output names.\n";
}

TEST(TransformationTests, RNNSequenceConversionTest) {
    const size_t hidden_size = 3;
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    std::shared_ptr<ngraph::opset4::RNNSequence> sequence;

    {
        auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto H = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto W = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto R = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3});
        auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{1}, 1);
        sequence = std::make_shared<ngraph::opset4::RNNSequence>(X, H, seq_len, W, R, B, hidden_size,
                                                                 ngraph::op::RecurrentSequenceDirection::FORWARD);
        sequence->set_friendly_name("test_sequence");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence}, ngraph::ParameterVector{X, H});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertRNNSequenceMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto H = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto W = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto R = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3, 3});
        auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{3});
        auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32, ngraph::Shape{1}, 1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 1);
        auto sequence_ie = std::make_shared<ngraph::op::RNNSequenceIE>(X,
                                                                       H,
                                                                       seq_len,
                                                                       concat,
                                                                       B,
                                                                       sequence->get_hidden_size(),
                                                                       sequence->get_direction(),
                                                                       sequence->get_activations(),
                                                                       sequence->get_activations_alpha(),
                                                                       sequence->get_activations_beta(),
                                                                       sequence->get_clip());

        sequence_ie->set_friendly_name("test_sequence");
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence_ie}, ngraph::ParameterVector{X, H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto sequence_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(sequence_node->get_friendly_name() == "test_sequence")
                                << "Transformation ConvertRNNSequenceToRNNSequenceIE should keep output names.\n";
}

TEST(TransformationTests, LSTMSequenceConversionTest) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const size_t num_directions = 1;
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    std::shared_ptr<ngraph::opset4::LSTMSequence> sequence;
    {
        const auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 10, input_size});
        const auto W =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions,
                                                                                   hidden_size});
        const auto C_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions,
                                                                                gates_count * hidden_size});
        const auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
        sequence = std::make_shared<ngraph::opset4::LSTMSequence>(X, H_t, C_t, seq_len, W, R, B, hidden_size,
                                                                  ngraph::op::RecurrentSequenceDirection::FORWARD);
        sequence->set_friendly_name("test_sequence");

        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{sequence->output(0)}, ngraph::ParameterVector{X, H_t, C_t});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertLSTMSequenceMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        const auto X = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                   ngraph::Shape{batch_size, 10, input_size});
        const auto W =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, input_size});
        const auto R =
                std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                           ngraph::Shape{num_directions,
                                                                         gates_count * hidden_size, hidden_size});
        const auto H_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto C_t = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32,
                                                                     ngraph::Shape{batch_size, num_directions, hidden_size});
        const auto B = std::make_shared<ngraph::opset4::Constant>(ngraph::element::f32,
                                                                  ngraph::Shape{num_directions,
                                                                                gates_count * hidden_size});
        const auto seq_len = std::make_shared<ngraph::opset4::Constant>(ngraph::element::i32, ngraph::Shape{1}, 1);
        auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::NodeVector({W, R}), 1);
        auto sequence_ie = std::make_shared<ngraph::op::LSTMSequenceIE>(X,
                                                                        H_t,
                                                                        C_t,
                                                                        seq_len,
                                                                        concat,
                                                                        B,
                                                                        sequence->get_hidden_size(),
                                                                        sequence->get_direction(),
                                                                        sequence->get_activations(),
                                                                        sequence->get_activations_alpha(),
                                                                        sequence->get_activations_beta(),
                                                                        sequence->get_clip_threshold());
        sequence_ie->set_friendly_name("test_sequence");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sequence_ie},
                                                   ngraph::ParameterVector{X, H_t, C_t});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto sequence_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(sequence_node->get_friendly_name() == "test_sequence")
                                << "Transformation ConvertLSTMSequenceToLSTMSequenceIE should keep output names.\n";
}