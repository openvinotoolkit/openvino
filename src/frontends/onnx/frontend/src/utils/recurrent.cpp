// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/recurrent.hpp"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "core/null_node.hpp"
#include "openvino/core/enum_names.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace recurrent {
OpInputMap::OpInputMap(const ov::frontend::onnx::Node& node, std::size_t gates_count) {
    const auto& ng_inputs = node.get_ov_inputs();

    m_map[OpInput::X] = ov::op::util::reorder_axes(ng_inputs.at(0), {1, 0, 2});
    m_map[OpInput::W] = ng_inputs.at(1);
    m_map[OpInput::R] = ng_inputs.at(2);

    const auto x_pshape = m_map[OpInput::X].get_partial_shape();
    const auto w_pshape = m_map[OpInput::W].get_partial_shape();
    const auto r_pshape = m_map[OpInput::R].get_partial_shape();

    // Get dimensions needed for default inputs creation
    auto shape_of_x = std::make_shared<v3::ShapeOf>(m_map[OpInput::X]);
    auto axes = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto batch_size_node =
        std::make_shared<v8::Gather>(shape_of_x, v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}), axes);
    auto seq_length_node =
        std::make_shared<v8::Gather>(shape_of_x, v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}), axes);

    auto shape_of_r = std::make_shared<v3::ShapeOf>(m_map[OpInput::R]);
    auto num_directions_node =
        std::make_shared<v8::Gather>(shape_of_r, v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}), axes);
    auto hidden_size_node =
        std::make_shared<v8::Gather>(shape_of_r, v0::Constant::create(ov::element::i32, ov::Shape{1}, {2}), axes);

    // ------ Optional inputs ------
    if (ng_inputs.size() > 3 && !ov::op::util::is_null(ng_inputs.at(3))) {
        auto bias = ng_inputs.at(3);
        auto split_bias = ov::op::util::make_split(bias, 2, 1);
        m_map[OpInput::B] = std::make_shared<v1::Add>(split_bias.at(0), split_bias.at(1));
    } else {
        auto b_shape = std::make_shared<v0::Concat>(
            ov::OutputVector{num_directions_node,
                             std::make_shared<v1::Multiply>(
                                 v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {gates_count}),
                                 hidden_size_node)},
            0);
        m_map[OpInput::B] = std::make_shared<v3::Broadcast>(
            v0::Constant::create(m_map[OpInput::X].get_element_type(), ov::Shape{}, {0}),
            b_shape);
    }
    if (ng_inputs.size() > 4 && !ov::op::util::is_null(ng_inputs.at(4))) {
        m_map[OpInput::SEQ_LENGTHS] = ng_inputs.at(4);
    } else {
        m_map[OpInput::SEQ_LENGTHS] = std::make_shared<v3::Broadcast>(seq_length_node, batch_size_node);
    }
    // The initial value of the hidden.
    if (ng_inputs.size() > 5 && !ov::op::util::is_null(ng_inputs.at(5))) {
        m_map[OpInput::INIT_H] = ov::op::util::reorder_axes(ng_inputs.at(5), {1, 0, 2});
    } else {
        auto init_h_shape =
            std::make_shared<v0::Concat>(ov::OutputVector{batch_size_node, num_directions_node, hidden_size_node}, 0);
        m_map[OpInput::INIT_H] = std::make_shared<v3::Broadcast>(
            v0::Constant::create(m_map[OpInput::X].get_element_type(), ov::Shape{}, {0}),
            init_h_shape);
    }
}

OpInputMap::OpInputMap(container_type&& map) : m_map(std::move(map)) {}

ov::Output<ov::Node>& OpInputMap::at(const OpInput& key) {
    return m_map.at(key);
}
const ov::Output<ov::Node>& OpInputMap::at(const OpInput& key) const {
    return m_map.at(key);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpAttributes::OpAttributes(const Node& node)
    : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")},
      m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)}
      // Recurrent Operators which have more activation functions should override
      // this value in constructor of respective Attributes' struct.
      ,
      m_activations{node.get_attribute_value<std::vector<std::string>>("activations", {"tanh"})}
      // Default values for activation functions are same
      // as for corresponding ONNX operator.
      ,
      m_activations_alpha{node.get_attribute_value<std::vector<float>>("activation_alpha", std::vector<float>{})},
      m_activations_beta{node.get_attribute_value<std::vector<float>>("activation_beta", std::vector<float>{})} {
    m_clip_threshold = std::abs(m_clip_threshold);
    std::string direction = ov::util::to_lower(node.get_attribute_value<std::string>("direction", "forward"));
    m_direction = ov::as_enum<ov::op::RecurrentSequenceDirection>(direction);
}

}  // namespace recurrent
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
