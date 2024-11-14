// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "utils/recurrent.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
namespace {
struct RNNInputMap : public recurrent::OpInputMap {
    RNNInputMap(const ov::frontend::onnx::Node& node, std::size_t gates_count) : OpInputMap(node, gates_count) {}

    virtual ~RNNInputMap() = default;
};

struct RNNAttributes : public recurrent::OpAttributes {
    RNNAttributes(const Node& node) : OpAttributes(node) {}

    virtual ~RNNAttributes() = default;
};
}  // namespace

ov::OutputVector rnn(const ov::frontend::onnx::Node& node) {
    constexpr std::size_t gates_count = 1;
    RNNInputMap input_map{node, gates_count};
    RNNAttributes attributes{node};

    auto rnn_sequence = std::make_shared<v5::RNNSequence>(input_map.at(recurrent::OpInput::X),
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

    return {ov::op::util::reorder_axes(Y, {2, 1, 0, 3}), ov::op::util::reorder_axes(Y_h, {1, 0, 2})};
}
ONNX_OP("RNN", OPSET_SINCE(1), ai_onnx::opset_1::rnn);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
