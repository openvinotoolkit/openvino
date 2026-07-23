// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/recurrent.hpp"

#include <cstdint>
#include <cstdlib>
#include <numeric>
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
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace recurrent {

ov::Output<ov::Node> normalize_tensor_rank(const ov::Output<ov::Node>& input,
                                           int64_t target_rank,
                                           const std::string& op_name,
                                           const std::string& input_name) {
    const auto& input_rank = input.get_partial_shape().rank();

    if (input_rank.is_dynamic() || input_rank.get_length() == target_rank) {
        return input;
    }

    if (input_rank.get_length() > target_rank) {
        // Squeeze leading dimensions to reduce rank to target_rank.
        const auto dims_to_squeeze = input_rank.get_length() - target_rank;

        // Static validation: leading dimensions that are statically known and != 1 cannot be squeezed.
        const auto& input_shape = input.get_partial_shape();
        for (int64_t i = 0; i < dims_to_squeeze; ++i) {
            if (input_shape[i].is_static() && input_shape[i].get_length() != 1) {
                OPENVINO_THROW(op_name,
                               " input '",
                               input_name,
                               "' has rank ",
                               input_rank.get_length(),
                               " but expected ",
                               target_rank,
                               ". Leading dimension [",
                               i,
                               "] is ",
                               input_shape[i].get_length(),
                               " but must be 1 to squeeze.");
            }
        }

        std::vector<int64_t> axes_to_squeeze(dims_to_squeeze);
        std::iota(axes_to_squeeze.begin(), axes_to_squeeze.end(), 0);
        auto axes_const = v0::Constant::create(ov::element::i64, Shape{axes_to_squeeze.size()}, axes_to_squeeze);
        return std::make_shared<v0::Squeeze>(input, axes_const);
    }

    // input_rank < target_rank: only allow exactly 1 missing dimension (the num_directions dim).
    // For unidirectional operators (forward/reverse), num_directions=1, so some models omit it.
    // A larger rank deficiency indicates a genuinely malformed model.
    const auto dims_to_unsqueeze = target_rank - input_rank.get_length();
    if (dims_to_unsqueeze != 1) {
        OPENVINO_THROW(op_name,
                       " input '",
                       input_name,
                       "' has rank ",
                       input_rank.get_length(),
                       " but expected ",
                       target_rank,
                       ". Rank difference is ",
                       dims_to_unsqueeze,
                       " but only 1 (missing num_directions) is supported.");
    }
    auto axes_const = v0::Constant::create(ov::element::i64, Shape{1}, std::vector<int64_t>{0});
    return std::make_shared<v0::Unsqueeze>(input, axes_const);
}

namespace {
// Gather a single dimension (as a rank-1 i32 tensor) from a tensor's runtime shape.
ov::Output<ov::Node> gather_dim(const ov::Output<ov::Node>& shape_of, int64_t index) {
    const auto axes = v0::Constant::create(ov::element::i32, Shape{1}, {0});
    return std::make_shared<v8::Gather>(shape_of, v0::Constant::create(ov::element::i32, Shape{1}, {index}), axes);
}
}  // namespace

LSTMDimensions::LSTMDimensions(const ov::Output<ov::Node>& x_ov_layout, const ov::Output<ov::Node>& r_ov_layout) {
    // X (OpenVINO layout): [batch_size, seq_length, input_size]
    auto shape_of_x = std::make_shared<v3::ShapeOf>(x_ov_layout);
    batch_size = gather_dim(shape_of_x, 0);
    seq_length = gather_dim(shape_of_x, 1);

    // R: [num_directions, gates*hidden_size, hidden_size]
    auto shape_of_r = std::make_shared<v3::ShapeOf>(r_ov_layout);
    num_directions = gather_dim(shape_of_r, 0);
    hidden_size = gather_dim(shape_of_r, 2);
}

ov::Output<ov::Node> default_bias(const LSTMDimensions& dims,
                                  const ov::element::Type& element_type,
                                  int64_t gates_count) {
    auto b_shape = std::make_shared<v0::Concat>(
        ov::OutputVector{dims.num_directions,
                         std::make_shared<v1::Multiply>(v0::Constant::create(ov::element::i64, Shape{1}, {gates_count}),
                                                        dims.hidden_size)},
        0);
    return std::make_shared<v3::Broadcast>(v0::Constant::create(element_type, Shape{}, {0}), b_shape);
}

ov::Output<ov::Node> default_sequence_lens(const LSTMDimensions& dims) {
    return std::make_shared<v3::Broadcast>(dims.seq_length, dims.batch_size);
}

ov::Output<ov::Node> default_initial_state(const LSTMDimensions& dims, const ov::element::Type& element_type) {
    auto state_shape =
        std::make_shared<v0::Concat>(ov::OutputVector{dims.batch_size, dims.num_directions, dims.hidden_size}, 0);
    return std::make_shared<v3::Broadcast>(v0::Constant::create(element_type, Shape{}, {0}), state_shape);
}

OpInputMap::OpInputMap(const ov::frontend::onnx::Node& node, std::size_t gates_count) {
    const auto& ng_inputs = node.get_ov_inputs();

    m_map[OpInput::X] = ov::op::util::reorder_axes(ng_inputs.at(0), {1, 0, 2});
    m_map[OpInput::W] = ng_inputs.at(1);
    m_map[OpInput::R] = ng_inputs.at(2);

    // X must be in OV layout [batch, seq, input] (reorder_axes applied above) before LSTMDimensions
    // is constructed; constructing it earlier would swap batch/seq in all derived default tensors.
    const LSTMDimensions dims{m_map[OpInput::X], m_map[OpInput::R]};
    const auto x_type = m_map[OpInput::X].get_element_type();

    // ------ Optional inputs ------
    if (ng_inputs.size() > 3 && !ov::op::util::is_null(ng_inputs.at(3))) {
        auto bias = ng_inputs.at(3);
        auto split_bias = ov::op::util::make_split(bias, 2, 1);
        m_map[OpInput::B] = std::make_shared<v1::Add>(split_bias.at(0), split_bias.at(1));
    } else {
        m_map[OpInput::B] = default_bias(dims, x_type, gates_count);
    }
    if (ng_inputs.size() > 4 && !ov::op::util::is_null(ng_inputs.at(4))) {
        m_map[OpInput::SEQ_LENGTHS] = ng_inputs.at(4);
    } else {
        m_map[OpInput::SEQ_LENGTHS] = default_sequence_lens(dims);
    }
    // The initial value of the hidden.
    if (ng_inputs.size() > 5 && !ov::op::util::is_null(ng_inputs.at(5))) {
        m_map[OpInput::INIT_H] = ov::op::util::reorder_axes(ng_inputs.at(5), {1, 0, 2});
    } else {
        m_map[OpInput::INIT_H] = default_initial_state(dims, x_type);
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
