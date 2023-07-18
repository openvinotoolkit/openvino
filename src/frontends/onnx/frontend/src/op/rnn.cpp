// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/rnn.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "utils/recurrent.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
namespace {
struct RNNInputMap : public recurrent::OpInputMap {
    RNNInputMap(const onnx_import::Node& node, std::size_t gates_count) : OpInputMap(node, gates_count) {}

    virtual ~RNNInputMap() = default;
};

struct RNNAttributes : public recurrent::OpAttributes {
    RNNAttributes(const Node& node) : OpAttributes(node) {}

    virtual ~RNNAttributes() = default;
};
}  // namespace

OutputVector rnn(const Node& node) {
    constexpr std::size_t gates_count = 1;
    RNNInputMap input_map{node, gates_count};
    RNNAttributes attributes{node};

    auto rnn_sequence = std::make_shared<default_opset::RNNSequence>(input_map.at(recurrent::OpInput::X),
                                                                     input_map.at(recurrent::OpInput::INIT_H),
                                                                     input_map.at(recurrent::OpInput::SEQ_LENGTHS),
                                                                     input_map.at(recurrent::OpInput::W),
                                                                     input_map.at(recurrent::OpInput::R),
                                                                     input_map.at(recurrent::OpInput::B),
                                                                     attributes.m_hidden_size,
                                                                     attributes.m_direction,
                                                                     attributes.m_activations,
                                                                     attributes.m_activations_alpha,
                                                                     attributes.m_activations_beta,
                                                                     attributes.m_clip_threshold);

    const auto Y = rnn_sequence->output(0);
    const auto Y_h = rnn_sequence->output(1);

    return {builder::opset1::reorder_axes(Y, {2, 1, 0, 3}), builder::opset1::reorder_axes(Y_h, {1, 0, 2})};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
