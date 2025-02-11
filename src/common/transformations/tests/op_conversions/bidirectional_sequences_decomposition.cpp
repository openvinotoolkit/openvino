// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

struct Inputs {
    unsigned long hidden_size;
    std::shared_ptr<opset5::Parameter> X;
    std::shared_ptr<opset5::Parameter> H;
    std::shared_ptr<opset5::Parameter> C;
    std::shared_ptr<opset5::Constant> S;
    std::shared_ptr<opset5::Constant> W;
    std::shared_ptr<opset5::Constant> R;
    std::shared_ptr<opset5::Constant> B;
};

Inputs getInputs(const unsigned long num_gates) {
    Inputs retn;

    retn.hidden_size = 1;
    const unsigned long batch_size = 8;
    const unsigned long input_size = 10;
    const unsigned long seq_len = 2;
    const unsigned long num_directions = 2;

    const auto seq_len_val = std::vector<int>(batch_size, seq_len);
    const auto w_val = std::vector<float>(num_directions * num_gates * retn.hidden_size * input_size, 0);
    const auto r_val = std::vector<float>(num_directions * num_gates * retn.hidden_size * retn.hidden_size, 0);
    const auto b_val = std::vector<float>(num_directions * num_gates * retn.hidden_size, 0);

    retn.X = std::make_shared<opset5::Parameter>(element::f32, Shape{batch_size, seq_len, input_size});
    retn.H = std::make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, retn.hidden_size});
    retn.C = std::make_shared<opset5::Parameter>(element::f32, Shape{batch_size, num_directions, retn.hidden_size});
    retn.S = std::make_shared<opset5::Constant>(element::i32, Shape{batch_size}, seq_len_val);
    retn.W = std::make_shared<opset5::Constant>(element::f32,
                                                Shape{num_directions, num_gates * retn.hidden_size, input_size},
                                                w_val);
    retn.R = std::make_shared<opset5::Constant>(element::f32,
                                                Shape{num_directions, num_gates * retn.hidden_size, retn.hidden_size},
                                                r_val);
    retn.B =
        std::make_shared<opset5::Constant>(element::f32, Shape{num_directions, num_gates * retn.hidden_size}, b_val);

    return retn;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionLSTM) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 4;
    auto ins = getInputs(num_gates);

    {
        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(ins.X,
                                                               ins.H,
                                                               ins.C,
                                                               ins.S,
                                                               ins.W,
                                                               ins.R,
                                                               ins.B,
                                                               ins.hidden_size,
                                                               op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                        ParameterVector{ins.X, ins.H, ins.C});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto axis_0 = opset5::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = opset5::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<opset5::Split>(ins.H, axis_1, 2);
        auto C_split = std::make_shared<opset5::Split>(ins.C, axis_1, 2);
        auto W_split = std::make_shared<opset5::Split>(ins.W, axis_0, 2);
        auto R_split = std::make_shared<opset5::Split>(ins.R, axis_0, 2);
        auto B_split = std::make_shared<opset5::Split>(ins.B, axis_0, 2);

        auto lstm_seq_forward = std::make_shared<opset5::LSTMSequence>(ins.X,
                                                                       H_split->output(0),
                                                                       C_split->output(0),
                                                                       ins.S,
                                                                       W_split->output(0),
                                                                       R_split->output(0),
                                                                       B_split->output(0),
                                                                       ins.hidden_size,
                                                                       op::RecurrentSequenceDirection::FORWARD);
        auto lstm_seq_reverse = std::make_shared<opset5::LSTMSequence>(ins.X,
                                                                       H_split->output(1),
                                                                       C_split->output(1),
                                                                       ins.S,
                                                                       W_split->output(1),
                                                                       R_split->output(1),
                                                                       B_split->output(1),
                                                                       ins.hidden_size,
                                                                       op::RecurrentSequenceDirection::REVERSE);

        auto concat_0 =
            std::make_shared<opset5::Concat>(OutputVector{lstm_seq_forward->output(0), lstm_seq_reverse->output(0)}, 1);
        auto concat_1 =
            std::make_shared<opset5::Concat>(OutputVector{lstm_seq_forward->output(1), lstm_seq_reverse->output(1)}, 1);
        auto concat_2 =
            std::make_shared<opset5::Concat>(OutputVector{lstm_seq_forward->output(2), lstm_seq_reverse->output(2)}, 1);

        f_ref = std::make_shared<ov::Model>(OutputVector{concat_0, concat_1, concat_2},
                                            ParameterVector{ins.X, ins.H, ins.C});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionGRU) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 3;
    auto ins = getInputs(num_gates);

    {
        auto gru_seq = std::make_shared<opset5::GRUSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{gru_seq->output(0), gru_seq->output(1)},
                                        ParameterVector{ins.X, ins.H});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto axis_0 = opset5::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = opset5::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<opset5::Split>(ins.H, axis_1, 2);
        auto W_split = std::make_shared<opset5::Split>(ins.W, axis_0, 2);
        auto R_split = std::make_shared<opset5::Split>(ins.R, axis_0, 2);
        auto B_split = std::make_shared<opset5::Split>(ins.B, axis_0, 2);

        auto gru_seq_forward = std::make_shared<opset5::GRUSequence>(ins.X,
                                                                     H_split->output(0),
                                                                     ins.S,
                                                                     W_split->output(0),
                                                                     R_split->output(0),
                                                                     B_split->output(0),
                                                                     ins.hidden_size,
                                                                     op::RecurrentSequenceDirection::FORWARD);
        auto gru_seq_reverse = std::make_shared<opset5::GRUSequence>(ins.X,
                                                                     H_split->output(1),
                                                                     ins.S,
                                                                     W_split->output(1),
                                                                     R_split->output(1),
                                                                     B_split->output(1),
                                                                     ins.hidden_size,
                                                                     op::RecurrentSequenceDirection::REVERSE);

        auto concat_0 =
            std::make_shared<opset5::Concat>(OutputVector{gru_seq_forward->output(0), gru_seq_reverse->output(0)}, 1);
        auto concat_1 =
            std::make_shared<opset5::Concat>(OutputVector{gru_seq_forward->output(1), gru_seq_reverse->output(1)}, 1);

        f_ref = std::make_shared<ov::Model>(OutputVector{concat_0, concat_1}, ParameterVector{ins.X, ins.H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionRNN) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 1;
    auto ins = getInputs(num_gates);

    {
        auto rnn_seq = std::make_shared<opset5::RNNSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{rnn_seq->output(0), rnn_seq->output(1)},
                                        ParameterVector{ins.X, ins.H});

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto axis_0 = opset5::Constant::create(element::i64, Shape{}, {0});
        auto axis_1 = opset5::Constant::create(element::i64, Shape{}, {1});
        auto H_split = std::make_shared<opset5::Split>(ins.H, axis_1, 2);
        auto W_split = std::make_shared<opset5::Split>(ins.W, axis_0, 2);
        auto R_split = std::make_shared<opset5::Split>(ins.R, axis_0, 2);
        auto B_split = std::make_shared<opset5::Split>(ins.B, axis_0, 2);

        auto rnn_seq_forward = std::make_shared<opset5::RNNSequence>(ins.X,
                                                                     H_split->output(0),
                                                                     ins.S,
                                                                     W_split->output(0),
                                                                     R_split->output(0),
                                                                     B_split->output(0),
                                                                     ins.hidden_size,
                                                                     op::RecurrentSequenceDirection::FORWARD);
        auto rnn_seq_reverse = std::make_shared<opset5::RNNSequence>(ins.X,
                                                                     H_split->output(1),
                                                                     ins.S,
                                                                     W_split->output(1),
                                                                     R_split->output(1),
                                                                     B_split->output(1),
                                                                     ins.hidden_size,
                                                                     op::RecurrentSequenceDirection::REVERSE);

        auto concat_0 =
            std::make_shared<opset5::Concat>(OutputVector{rnn_seq_forward->output(0), rnn_seq_reverse->output(0)}, 1);
        auto concat_1 =
            std::make_shared<opset5::Concat>(OutputVector{rnn_seq_forward->output(1), rnn_seq_reverse->output(1)}, 1);

        f_ref = std::make_shared<ov::Model>(OutputVector{concat_0, concat_1}, ParameterVector{ins.X, ins.H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionLSTMDisabled) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 4;
    auto ins = getInputs(num_gates);

    {
        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(ins.X,
                                                               ins.H,
                                                               ins.C,
                                                               ins.S,
                                                               ins.W,
                                                               ins.R,
                                                               ins.B,
                                                               ins.hidden_size,
                                                               op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                        ParameterVector{ins.X, ins.H, ins.C});

        const auto transformations_callback = [](const std::shared_ptr<const ::Node>& node) -> bool {
            if (as_type<const opset5::LSTMSequence>(node.get())) {
                return true;
            }
            return false;
        };

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.get_pass_config()->set_callback(transformations_callback);
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto lstm_seq = std::make_shared<opset5::LSTMSequence>(ins.X,
                                                               ins.H,
                                                               ins.C,
                                                               ins.S,
                                                               ins.W,
                                                               ins.R,
                                                               ins.B,
                                                               ins.hidden_size,
                                                               op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f_ref = std::make_shared<ov::Model>(OutputVector{lstm_seq->output(0), lstm_seq->output(1), lstm_seq->output(2)},
                                            ParameterVector{ins.X, ins.H, ins.C});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionGRUDisabled) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 3;
    auto ins = getInputs(num_gates);

    {
        auto gru_seq = std::make_shared<opset5::GRUSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{gru_seq->output(0), gru_seq->output(1)},
                                        ParameterVector{ins.X, ins.H});

        const auto transformations_callback = [](const std::shared_ptr<const ::Node>& node) -> bool {
            if (as_type<const opset5::GRUSequence>(node.get())) {
                return true;
            }
            return false;
        };

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.get_pass_config()->set_callback(transformations_callback);
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto gru_seq = std::make_shared<opset5::GRUSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f_ref = std::make_shared<ov::Model>(OutputVector{gru_seq->output(0), gru_seq->output(1)},
                                            ParameterVector{ins.X, ins.H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, BidirectionalSequenceDecompositionRNNDisabled) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    const unsigned long num_gates = 1;
    auto ins = getInputs(num_gates);

    {
        auto rnn_seq = std::make_shared<opset5::RNNSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f = std::make_shared<ov::Model>(OutputVector{rnn_seq->output(0), rnn_seq->output(1)},
                                        ParameterVector{ins.X, ins.H});

        const auto transformations_callback = [](const std::shared_ptr<const ::Node>& node) -> bool {
            if (as_type<const opset5::RNNSequence>(node.get())) {
                return true;
            }
            return false;
        };

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::BidirectionalSequenceDecomposition>();
        m.get_pass_config()->set_callback(transformations_callback);
        m.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto rnn_seq = std::make_shared<opset5::RNNSequence>(ins.X,
                                                             ins.H,
                                                             ins.S,
                                                             ins.W,
                                                             ins.R,
                                                             ins.B,
                                                             ins.hidden_size,
                                                             op::RecurrentSequenceDirection::BIDIRECTIONAL);
        f_ref = std::make_shared<ov::Model>(OutputVector{rnn_seq->output(0), rnn_seq->output(1)},
                                            ParameterVector{ins.X, ins.H});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
