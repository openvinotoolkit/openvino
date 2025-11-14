// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/gru_sequence.hpp"

namespace tests {
inline std::shared_ptr<ov::Model> makeLBRGRUSequence(ov::element::Type_t model_type, ov::PartialShape initShape,
    size_t N, size_t I, size_t H, size_t sequence_axis, ov::op::RecurrentSequenceDirection seq_direction) {
    auto X = std::make_shared<ov::op::v0::Parameter>(model_type, initShape);
    auto Y = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{N, 1, H});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
    auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

    auto w_val = std::vector<float>(3 * H * I, 0);
    auto r_val = std::vector<float>(3 * H * H, 0);
    auto b_val = std::vector<float>(4 * H, 0);
    auto W = ov::op::v0::Constant::create(model_type, ov::Shape{N, 3 * H, I}, w_val);
    auto R = ov::op::v0::Constant::create(model_type, ov::Shape{N, 3 * H, H}, r_val);
    auto B = ov::op::v0::Constant::create(model_type, ov::Shape{N, 4 * H}, b_val);
    auto default_activations = std::vector<std::string>{"sigmoid", "tanh"};
    std::vector<float> empty_activations;
    auto rnn_sequence = std::make_shared<ov::op::v5::GRUSequence>(X,
        Y,
        seq_lengths,
        W,
        R,
        B,
        128,
        seq_direction,
        default_activations,
        empty_activations,
        empty_activations,
        0.f,
        true);

    auto Y_out = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(0));
    auto Ho = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(1));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");

    auto fn_ptr = std::make_shared<ov::Model>(ov::NodeVector{Y_out, Ho}, ov::ParameterVector{X, Y});
    fn_ptr->set_friendly_name("GRUSequence");
    return fn_ptr;
}


/*
*   Generate LSTMSequence
*   @param model_type precision of model
*   @param initShape initial shape {N, L(sequence length), I}
*   @param N batch size
*   @param I input size
*   @param H hidden layer
*/
inline std::shared_ptr<ov::Model> makeLSTMSequence(ov::element::Type_t model_type, ov::PartialShape initShape,
    size_t N, size_t I, size_t H, size_t sequence_axis,
    ov::op::RecurrentSequenceDirection seq_direction) {
    auto X = std::make_shared<ov::op::v0::Parameter>(model_type, initShape);
    auto Y = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{N, 1, H});
    auto Z = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape{N, 1, H});
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
    auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
    auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

    auto w_val = std::vector<float>(4 * H * I, 0);
    auto r_val = std::vector<float>(4 * H * H, 0);
    auto b_val = std::vector<float>(4 * H, 0);
    auto W = ov::op::v0::Constant::create(model_type, ov::Shape{N, 4 * H, I}, w_val);
    auto R = ov::op::v0::Constant::create(model_type, ov::Shape{N, 4 * H, H}, r_val);
    auto B = ov::op::v0::Constant::create(model_type, ov::Shape{N, 4 * H}, b_val);

    auto rnn_sequence = std::make_shared<ov::op::v5::LSTMSequence>(X,
                Y,
                Z,
                seq_lengths,
                W,
                R,
                B,
                128,
                seq_direction);
    auto Y_out = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(0));
    auto Ho = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(1));
    auto Co = std::make_shared<ov::op::v0::Result>(rnn_sequence->output(2));
    Y_out->set_friendly_name("Y_out");
    Ho->set_friendly_name("Ho");
    Co->set_friendly_name("Co");

    auto fn_ptr = std::make_shared<ov::Model>(ov::OutputVector{Y_out, Ho, Co}, ov::ParameterVector{X, Y, Z});
    fn_ptr->set_friendly_name("LSTMSequence");
    return fn_ptr;
}

} // namespace tests
