// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/recurrent.hpp"
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

using ov::frontend::onnx::recurrent::normalize_tensor_rank;

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

        // First normalize rank if needed, THEN reorder axes
        // This is important because Squeeze/Unsqueeze changes dimension indices
        auto input_x = ng_inputs.at(0);
        input_x = normalize_tensor_rank(input_x, 3, "LSTM", "X");
        input_x = ov::op::util::reorder_axes(input_x, {1, 0, 2});

        m_input_map[LSTMInput::LSTM_INPUT_X] = input_x;

        // Detect if num_directions dimension is missing from W.
        // Some models omit the leading num_directions dimension when it equals 1.
        // normalize_tensor_rank will unsqueeze it, but we need to squeeze the
        // corresponding dimension from outputs to match the original model's expectations.
        const auto& w_rank = ng_inputs.at(1).get_partial_shape().rank();
        if (w_rank.is_static() && w_rank.get_length() < 3) {
            // Unsqueezing adds num_directions=1, which is only valid for forward/reverse.
            // For bidirectional LSTMs, num_directions=2 and cannot be inferred.
            const std::string direction =
                ov::util::to_lower(node.get_attribute_value<std::string>("direction", "forward"));
            OPENVINO_ASSERT(direction != "bidirectional",
                            "LSTM input 'W' has rank ",
                            w_rank.get_length(),
                            " but expected 3. Cannot add num_directions dimension for bidirectional LSTM "
                            "because num_directions=2 cannot be inferred from the data.");
            m_num_directions_unsqueezed = true;
        }

        // Weight tensor for the gates.
        // ONNX Shape: [num_directions, 4*hidden_size, input_size]
        auto input_w = normalize_tensor_rank(ng_inputs.at(1), 3, "LSTM", "W");
        m_input_map[LSTMInput::LSTM_INPUT_W] =
            ov::op::util::convert_lstm_node_format(input_w,
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        // The recurrence weight tensor.
        // ONNX Shape: [num_directions, 4*hidden_size, hidden_size]
        auto input_r = normalize_tensor_rank(ng_inputs.at(2), 3, "LSTM", "R");
        m_input_map[LSTMInput::LSTM_INPUT_R] =
            ov::op::util::convert_lstm_node_format(input_r,
                                                   ov::op::util::LSTMWeightsFormat::IOFC,
                                                   ov::op::util::LSTMWeightsFormat::FICO,
                                                   1);

        // Get dimensions needed for default inputs creation
        const ov::frontend::onnx::recurrent::LSTMDimensions dims{m_input_map[LSTMInput::LSTM_INPUT_X],
                                                                 m_input_map[LSTMInput::LSTM_INPUT_R]};
        const auto x_type = m_input_map[LSTMInput::LSTM_INPUT_X].get_element_type();

        // ------ Optional inputs ------
        // `B` - The bias tensor for input gate.
        // ONNX Shape: [num_directions, 8*hidden_size]
        // OpenVino Shape: [num_directions, 4*hidden_size]
        if (ng_inputs.size() > 3 && !ov::op::util::is_null(ng_inputs.at(3))) {
            auto bias = normalize_tensor_rank(ng_inputs.at(3), 2, "LSTM", "B");
            auto split_bias = ov::op::util::make_split(bias, 2, 1);
            m_input_map[LSTMInput::LSTM_INPUT_B] = std::make_shared<v1::Add>(split_bias.at(0), split_bias.at(1));
            m_input_map[LSTMInput::LSTM_INPUT_B] =
                ov::op::util::convert_lstm_node_format(m_input_map[LSTMInput::LSTM_INPUT_B],
                                                       ov::op::util::LSTMWeightsFormat::IOFC,
                                                       ov::op::util::LSTMWeightsFormat::FICO,
                                                       1);
        } else {
            m_input_map[LSTMInput::LSTM_INPUT_B] =
                ov::frontend::onnx::recurrent::default_bias(dims, x_type, gates_count);
        }
        // `sequence_lens`- The lengths of the sequences in a batch.
        // Shape: [batch_size]
        if (ng_inputs.size() > 4 && !ov::op::util::is_null(ng_inputs.at(4))) {
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ng_inputs.at(4);
        } else {
            m_input_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ov::frontend::onnx::recurrent::default_sequence_lens(dims);
        }
        // `initial_h` - The initial value of the hidden.
        // ONNX Shape: [num_directions, batch_size, hidden_size]
        // OpenVino Shape: [batch_size, num_directions, hidden_size]
        if (ng_inputs.size() > 5 && !ov::op::util::is_null(ng_inputs.at(5))) {
            auto init_h = ng_inputs.at(5);
            // First normalize rank, THEN reorder axes
            init_h = normalize_tensor_rank(init_h, 3, "LSTM", "initial_h");
            init_h = ov::op::util::reorder_axes(init_h, {1, 0, 2});

            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] = init_h;
        } else {
            m_input_map[LSTMInput::LSTM_INPUT_INIT_H] =
                ov::frontend::onnx::recurrent::default_initial_state(dims, x_type);
        }
        // `initial_c` - The initial value of the cell.
        // ONNX Shape: [num_directions, batch_size, hidden_size]
        // OpenVino Shape: [batch_size, num_directions, hidden_size]
        if (ng_inputs.size() > 6 && !ov::op::util::is_null(ng_inputs.at(6))) {
            auto init_c = ng_inputs.at(6);
            // First normalize rank, THEN reorder axes
            init_c = normalize_tensor_rank(init_c, 3, "LSTM", "initial_c");
            init_c = ov::op::util::reorder_axes(init_c, {1, 0, 2});

            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] = init_c;
        } else {
            m_input_map[LSTMInput::LSTM_INPUT_INIT_C] =
                ov::frontend::onnx::recurrent::default_initial_state(dims, x_type);
        }
        // `P` - The weight tensor for peepholes.
        // ONNX Shape: [num_directions, 3*hidden_size]
        // OpenVino Shape: [num_directions, 4*hidden_size]
        if (ng_inputs.size() > 7 && !ov::op::util::is_null(ng_inputs.at(7))) {
            auto peepholes = normalize_tensor_rank(ng_inputs.at(7), 2, "LSTM", "P");
            m_input_map[LSTMInput::LSTM_INPUT_P] =
                ov::op::util::convert_lstm_peepholes_format(peepholes,
                                                            ov::op::util::LSTMPeepholesFormat::IOF,
                                                            ov::op::util::LSTMPeepholesFormat::FIO,
                                                            1);
        } else {
            // A blank peephole tensor of zeros: [num_directions, 3*hidden_size].
            m_input_map[LSTMInput::LSTM_INPUT_P] =
                ov::frontend::onnx::recurrent::default_bias(dims, x_type, P_gates_count);
            m_input_map[LSTMInput::LSTM_INPUT_P].set_names({"P_blank"});
        }
    }

    ov::Output<ov::Node>& at(const LSTMInput& key) {
        return m_input_map.at(key);
    }
    std::map<LSTMInput, ov::Output<ov::Node>> m_input_map;
    // True when num_directions dimension was missing from inputs and was added via Unsqueeze.
    // In this case, outputs need the num_directions dimension squeezed to match the original model.
    bool m_num_directions_unsqueezed = false;
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

    if (input_map.m_num_directions_unsqueezed) {
        // The num_directions dimension was added to inputs via Unsqueeze.
        // Squeeze it from outputs so downstream consumers see the original ranks.
        // Y: OV [batch_size, num_directions(1), seq_length, hidden_size] -> ONNX [seq_length, batch_size, hidden_size]
        // Y_h: OV [batch_size, num_directions(1), hidden_size] -> ONNX [batch_size, hidden_size]
        // Y_c: OV [batch_size, num_directions(1), hidden_size] -> ONNX [batch_size, hidden_size]
        auto num_dir_axis = v0::Constant::create(ov::element::i64, Shape{1}, {1});
        auto Y_squeezed = std::make_shared<v0::Squeeze>(Y, num_dir_axis);
        auto Y_h_squeezed = std::make_shared<v0::Squeeze>(Y_h, num_dir_axis);
        auto Y_c_squeezed = std::make_shared<v0::Squeeze>(Y_c, num_dir_axis);

        // Y: [batch_size, seq_length, hidden_size] -> [seq_length, batch_size, hidden_size]
        return {ov::op::util::reorder_axes(Y_squeezed, {1, 0, 2}), Y_h_squeezed, Y_c_squeezed};
    }

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
