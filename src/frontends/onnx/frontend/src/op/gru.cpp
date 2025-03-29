// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "utils/recurrent.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
namespace {
struct GRUInputMap : public recurrent::OpInputMap {
    GRUInputMap(const Node& node, std::size_t gates_count) : OpInputMap(node, gates_count) {
        bool linear_before_reset = static_cast<bool>(node.get_attribute_value<std::int64_t>("linear_before_reset", 0));

        // Override bias, since we need separated W and R biases for `h` gate.
        if (linear_before_reset) {
            const auto& ng_inputs = node.get_ov_inputs();
            const auto el_type = ng_inputs.at(0).get_element_type();

            if (ng_inputs.size() > 3 && !ov::op::util::is_null(ng_inputs.at(3))) {
                auto bias = ng_inputs.at(3);
                // gates_count * 2 since B is: [Wb, Rb]
                const int split_parts = 2 * 3;
                const auto split_bias = ov::op::util::make_split(bias, split_parts, 1);
                const auto wr_z_bias = std::make_shared<v1::Add>(split_bias.at(0), split_bias.at(3));
                const auto wr_r_bias = std::make_shared<v1::Add>(split_bias.at(1), split_bias.at(4));
                // The result has shape: [num_directions, 4 * hidden_size]
                // and data layout:
                //       [
                //          [Wb_z + Rb_z],
                //          [Wb_r + Rb_r],
                //          [Wb_h],
                //          [Rb_h],
                //          // num_directions times
                //       ]
                m_map[recurrent::OpInput::B] = std::make_shared<v0::Concat>(
                    ov::OutputVector{wr_z_bias, wr_r_bias, split_bias.at(2), split_bias.at(5)},
                    1);
            } else {
                const std::size_t hidden_size = m_map[recurrent::OpInput::R].get_shape().back();
                const std::size_t num_directions = m_map[recurrent::OpInput::W].get_shape().front();

                m_map[recurrent::OpInput::B] =
                    std::make_shared<v0::Constant>(el_type,
                                                   ov::Shape{num_directions, (gates_count + 1) * hidden_size},
                                                   0.f);
            }
        }
    }

    virtual ~GRUInputMap() = default;
};

struct GRUAttributes : public recurrent::OpAttributes {
    GRUAttributes(const Node& node)
        : OpAttributes(node),
          m_linear_before_reset{static_cast<bool>(node.get_attribute_value<std::int64_t>("linear_before_reset", 0))} {
        m_activations = node.get_attribute_value<std::vector<std::string>>("activations", {"sigmoid", "tanh"});
    }

    virtual ~GRUAttributes() = default;

    bool m_linear_before_reset;
};
}  // namespace

ov::OutputVector gru(const ov::frontend::onnx::Node& node) {
    constexpr std::size_t gates_count = 3;
    GRUInputMap input_map{node, gates_count};
    GRUAttributes attributes{node};

    auto gru_sequence = std::make_shared<v5::GRUSequence>(input_map.at(recurrent::OpInput::X),
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
                                                          attributes.m_clip_threshold,
                                                          attributes.m_linear_before_reset);

    const auto Y = gru_sequence->output(0);
    const auto Y_h = gru_sequence->output(1);

    return {ov::op::util::reorder_axes(Y, {2, 1, 0, 3}), ov::op::util::reorder_axes(Y_h, {1, 0, 2})};
}
ONNX_OP("GRU", OPSET_SINCE(1), ai_onnx::opset_1::gru);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
