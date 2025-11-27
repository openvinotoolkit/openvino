// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/reshape.hpp"
#include "utils/split.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace {
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum class LSTMInput {
    LSTM_INPUT_X,
    LSTM_INPUT_W,
    LSTM_INPUT_R,
    LSTM_INPUT_B,
    LSTM_INPUT_SEQ_LENGTHS,
    LSTM_INPUT_INIT_H,
    LSTM_INPUT_INIT_C,
    LSTM_INPUT_P
};

// Helper function to reduce tensor rank to target_rank by squeezing or reshaping
std::shared_ptr<ov::Node> reduce_tensor_rank(const ov::Output<ov::Node>& input, int64_t target_rank) {
    const auto& input_shape = input.get_partial_shape();

    if (!input_shape.rank().is_static()) {
        return input.get_node_shared_ptr();
    }

    const auto input_rank = input_shape.rank().get_length();

    if (input_rank <= target_rank) {
        return input.get_node_shared_ptr();
    }

    // Strategy: Try to squeeze all leading dimensions that are equal to 1
    std::vector<int64_t> axes_to_squeeze;
    for (int64_t i = 0; i < input_rank - target_rank; ++i) {
        if (input_shape[i].is_static() && input_shape[i].get_length() == 1) {
            axes_to_squeeze.push_back(i);
        }
    }

    if (axes_to_squeeze.size() == static_cast<size_t>(input_rank - target_rank)) {
        // All extra dimensions are 1, we can squeeze to get target rank
        auto axes_const = v0::Constant::create(ov::element::i64, Shape{axes_to_squeeze.size()}, axes_to_squeeze);
        return std::make_shared<v0::Squeeze>(input, axes_const);
    } else {
        // Some dimensions are not 1 or dynamic, need to reshape
        auto shape_of_input = std::make_shared<v3::ShapeOf>(input);
        auto start_idx = v0::Constant::create(ov::element::i64, Shape{1}, {input_rank - target_rank});
        auto stop_idx = v0::Constant::create(ov::element::i64, Shape{1}, {input_rank});
        auto step = v0::Constant::create(ov::element::i64, Shape{1}, {1});

        // Get last target_rank dimensions: shape[-target_rank:]
        auto last_dims = std::make_shared<v8::Slice>(shape_of_input, start_idx, stop_idx, step);

        // Reshape to extract last target_rank dimensions
        return std::make_shared<v1::Reshape>(input, last_dims, false);
    }
}

struct LSTMNgInputMap {
    explicit LSTMNgInputMap(const Node& node) {
        const auto& ng_inputs = node.get_ov_inputs();
        // We have input, output, forget and cell gates
        constexpr std::size_t gates_count{4};
        constexpr std::size_t P_gates_count{3};

        // ----- Mandatory inputs ------
        // Packed input sequences.
        // ONNX Shape: [seq_length, batch_size, input_size]
        // OpenVino Shape: [batch_size, seq_length, input_size]

        // First reduce rank if needed, THEN reorder axes
        // This is important because Squeeze changes dimension indices
        auto input_x = ng_inputs.at(0);
        input_x = reduce_tensor_rank(input_x, 3);
        input_x = ov::op::util::reorder_axes(input_x, {1, 0, 2});

        m_input_map[LSTMInput::LSTM_INPUT_X] = input_x;

        // Weight tensor for the gates.
        // Shape: [num_directions, 4*hidden_size, input_size]
        m_input_map[LSTMInput::LSTM_INPUT_W] =
            ov::op::util::convert_lstm_node_format(ng_inputs.at(1),
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        // The recurrence weight tensor.
        // Shape: [num_directions, 4*hidden_size, hidden_size]
        m_input_map[LSTMInput::LSTM_INPUT_R] =
            ov::op::util::convert_lstm_node_format(ng_inputs.at(2),
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        // Get dimensions needed for default inputs creation
        auto shape_of_x = std::make_shared<v3::ShapeOf>(m_input_map[LSTMInput::LSTM_INPUT_X]);
        auto axes = v0::Constant::create(ov::element::Type_t::i32, ov::Shape{1}, {0});
        auto batch_size_node =
            std::make_shared<v8::Gather>(shape_of_x,
                                         v0::Constant::create(ov::element::Type_t::i32, ov::Shape{1}, {0}),
                                         axes);
        auto seq_length_node =
            std::make_shared<v8::Gather>(shape_of_x,
                                         v0::Constant::create(ov::element::Type_t::i32, ov::Shape{1}, {1}),
                                         axes);

        auto shape_of_r = std::make_shared<v3::ShapeOf>(m_input_map[LSTMInput::LSTM_INPUT_R]);
        auto num_directions_node =
            std::make_shared<v8::Gather>(shape_of_r,
                                         v0::Constant::create(ov::element::Type_t::i32, ov::Shape{1}, {0}),
                                         axes);
        auto hidden_size_node =
            std::make_shared<v8::Gather>(shape_of_r,
                                         v0::Constant::create(ov::element::Type_t::i32, ov::Shape{1}, {2}),
                                         axes);

        // ------ Optional inputs ------
        // `B` - The bias tensor for input gate.
        // ONNX Shape: [num_directions, 8*hidden_size]
        // OpenVino Shape: [num_directions, 4*hidden_size]
        if (ng_inputs.size() > 3 && !ov::op::util::is_null(ng_inputs.at(3))) {
            auto bias = ng_inputs.at(3);
            auto split_bias = ov::op::util::make_split(bias, 2, 1);
            m_input_map[LSTMInput::LSTM_INPUT_B] = std::make_shared<v1::Add>(split_bias.at(0), split_bias.at(1));
            m_input_map[LSTMInput::LSTM_INPUT_B] =
                ov::op::util::convert_lstm_node_format(m_input_map[LSTMInput::LSTM_INPUT_B],
                                                       ov::op::util::LSTMWeightsFormat::IOFC,
                                                       ov::op::util::LSTMWeightsFormat::FICO,
                                                       1);
        } else {
            auto b_shape = std::make_shared<v0::Concat>(
                ov::OutputVector{num_directions_node,
                                 std::make_shared<v1::Multiply>(
                                     v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {gates_count}),
                                     hidden_size_node)},
                0);
            m_input_map[LSTMInput::LSTM_INPUT_B] = std::make_shared<v3::Broadcast>(
                v0::Constant::create(m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(), ov::Shape{}, {0}),
                b_shape);
        }
        // `sequence_lens`- The lengths of the sequences in a batch.
        // Shape: [batch_size]
        if (ng_inputs.size() > 4 && !ov::op::util::is_null(ng_inputs.at(4))) {
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ng_inputs.at(4);
        } else {
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] =
                std::make_shared<v3::Broadcast>(seq_length_node, batch_size_node);
        }
        // `initial_h` - The initial value of the hidden.
        // ONNX Shape: [num_directions, batch_size, hidden_size]
        // OpenVino Shape: [batch_size, num_directions, hidden_size]
        if (ng_inputs.size() > 5 && !ov::op::util::is_null(ng_inputs.at(5))) {
            auto init_h = ng_inputs.at(5);
            // First reduce rank, THEN reorder axes
            init_h = reduce_tensor_rank(init_h, 3);
            init_h = ov::op::util::reorder_axes(init_h, {1, 0, 2});

            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] = init_h;
        } else {
            auto init_h_shape =
                std::make_shared<v0::Concat>(ov::OutputVector{batch_size_node, num_directions_node, hidden_size_node},
                                             0);
            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] = std::make_shared<v3::Broadcast>(
                v0::Constant::create(m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(), ov::Shape{}, {0}),
                init_h_shape);
        }
        // `initial_c` - The initial value of the cell.
        // ONNX Shape: [num_directions, batch_size, hidden_size]
        // OpenVino Shape: [batch_size, num_directions, hidden_size]
        if (ng_inputs.size() > 6 && !ov::op::util::is_null(ng_inputs.at(6))) {
            auto init_c = ng_inputs.at(6);
            // First reduce rank, THEN reorder axes
            init_c = reduce_tensor_rank(init_c, 3);
            init_c = ov::op::util::reorder_axes(init_c, {1, 0, 2});

            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] = init_c;
        } else {
            auto init_c_shape =
                std::make_shared<v0::Concat>(ov::OutputVector{batch_size_node, num_directions_node, hidden_size_node},
                                             0);
            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] = std::make_shared<v3::Broadcast>(
                v0::Constant::create(m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(), ov::Shape{}, {0}),
                init_c_shape);
        }
        // `P` - The weight tensor for peepholes.
        // ONNX Shape: [num_directions, 3*hidden_size]
        // OpenVino Shape: [num_directions, 4*hidden_size]
        if (ng_inputs.size() > 7 && !ov::op::util::is_null(ng_inputs.at(7))) {
            m_input_map[LSTMInput::LSTM_INPUT_P] =
                ov::op::util::convert_lstm_peepholes_format(ng_inputs.at(7),
                                                            ov::op::util::LSTMPeepholesFormat::IOF,
                                                            ov::op::util::LSTMPeepholesFormat::FIO,
                                                            1);
        } else {
            auto p_shape = std::make_shared<v0::Concat>(
                ov::OutputVector{num_directions_node,
                                 std::make_shared<v1::Multiply>(
                                     v0::Constant::create(ov::element::Type_t::i64, ov::Shape{1}, {P_gates_count}),
                                     hidden_size_node)},
                0);
            m_input_map[LSTMInput::LSTM_INPUT_P] = std::make_shared<v3::Broadcast>(
                v0::Constant::create(m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type(), ov::Shape{}, {0}),
                p_shape);
            m_input_map[LSTMInput::LSTM_INPUT_P].set_names({"P_blank"});
        }
    }

    ov::Output<ov::Node>& at(const LSTMInput& key) {
        return m_input_map.at(key);
    }
    std::map<LSTMInput, ov::Output<ov::Node>> m_input_map;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct LSTMAttributes {
    explicit LSTMAttributes(const Node& node)
        : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")},
          m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)},
          m_activations{node.get_attribute_value<std::vector<std::string>>("activations", {"sigmoid", "tanh", "tanh"})}
          // Default values for activation functions are same as for corresponding
          // ONNX operator.
          ,
          m_activation_alpha{node.get_attribute_value<std::vector<float>>("activation_alpha", std::vector<float>{})},
          m_activation_beta{node.get_attribute_value<std::vector<float>>("activation_beta", std::vector<float>{})},
          m_input_forget{static_cast<bool>(node.get_attribute_value<std::int64_t>("input_forget", 0))} {
        m_clip_threshold = std::abs(m_clip_threshold);

        std::string direction = ov::util::to_lower(node.get_attribute_value<std::string>("direction", "forward"));

        m_direction = ov::as_enum<ov::op::RecurrentSequenceDirection>(direction);
    }

    ov::op::RecurrentSequenceDirection m_direction;
    std::int64_t m_hidden_size;
    float m_clip_threshold;
    std::vector<std::string> m_activations;
    std::vector<float> m_activation_alpha;
    std::vector<float> m_activation_beta;
    bool m_input_forget;
};

}  // anonymous namespace

namespace opset_1 {
ov::OutputVector lstm(const ov::frontend::onnx::Node& node) {
    LSTMNgInputMap input_map{node};
    LSTMAttributes attributes{node};
    std::shared_ptr<ov::Node> lstm_sequence;

    lstm_sequence = std::make_shared<v5::LSTMSequence>(input_map.at(LSTMInput::LSTM_INPUT_X),
                                                       input_map.at(LSTMInput::LSTM_INPUT_INIT_H),
                                                       input_map.at(LSTMInput::LSTM_INPUT_INIT_C),
                                                       input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                                       input_map.at(LSTMInput::LSTM_INPUT_W),
                                                       input_map.at(LSTMInput::LSTM_INPUT_R),
                                                       input_map.at(LSTMInput::LSTM_INPUT_B),
                                                       attributes.m_hidden_size,
                                                       attributes.m_direction,
                                                       attributes.m_activation_alpha,
                                                       attributes.m_activation_beta,
                                                       attributes.m_activations,
                                                       attributes.m_clip_threshold);

    const auto Y = lstm_sequence->output(0);
    const auto Y_h = lstm_sequence->output(1);
    const auto Y_c = lstm_sequence->output(2);

    return {ov::op::util::reorder_axes(Y, {2, 1, 0, 3}),
            ov::op::util::reorder_axes(Y_h, {1, 0, 2}),
            ov::op::util::reorder_axes(Y_c, {1, 0, 2})};
}
ONNX_OP("LSTM", OPSET_SINCE(1), ai_onnx::opset_1::lstm);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
