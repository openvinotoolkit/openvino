// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/rnn_cell_base.hpp"

namespace ov {
namespace op {
namespace v0 {

///
/// \brief      Class for lstm sequence node.
///
/// \note       It follows notation and equations defined as in ONNX standard:
///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
///
/// \sa         LSTMCell, RNNCell, GRUCell
///
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API OPENVINO_DEPRECATED(
    "The class ov::op::v0::LSTMSequence is deprecated. It will be removed in 2025.0 release. Use "
    "ov::op::v5::LSTMSequence instead.") LSTMSequence : public util::RNNCellBase {
public:
    OPENVINO_OP("LSTMSequence", "opset1", util::RNNCellBase);
    LSTMSequence() = default;

    using direction = RecurrentSequenceDirection;

    size_t get_default_output_index() const override {
        return no_default_index();
    }
    explicit LSTMSequence(const Output<Node>& X,
                          const Output<Node>& initial_hidden_state,
                          const Output<Node>& initial_cell_state,
                          const Output<Node>& sequence_lengths,
                          const Output<Node>& W,
                          const Output<Node>& R,
                          const Output<Node>& B,
                          const Output<Node>& P,
                          const std::int64_t hidden_size,
                          const direction lstm_direction,
                          LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                          const std::vector<float> activations_alpha = {},
                          const std::vector<float> activations_beta = {},
                          const std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"},
                          const float clip_threshold = 0,
                          const bool input_forget = false);

    explicit LSTMSequence(const Output<Node>& X,
                          const Output<Node>& initial_hidden_state,
                          const Output<Node>& initial_cell_state,
                          const Output<Node>& sequence_lengths,
                          const Output<Node>& W,
                          const Output<Node>& R,
                          const Output<Node>& B,
                          const std::int64_t hidden_size,
                          const direction lstm_direction,
                          LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                          const std::vector<float>& activations_alpha = {},
                          const std::vector<float>& activations_beta = {},
                          const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                          const float clip_threshold = 0,
                          const bool input_forget = false);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    std::vector<float> get_activations_alpha() const {
        return m_activations_alpha;
    }
    std::vector<float> get_activations_beta() const {
        return m_activations_beta;
    }
    std::vector<std::string> get_activations() const {
        return m_activations;
    }
    float get_clip_threshold() const {
        return m_clip;
    }
    direction get_direction() const {
        return m_direction;
    }
    void set_direction(const direction& dir) {
        m_direction = dir;
    }
    std::int64_t get_hidden_size() const {
        return m_hidden_size;
    }
    bool get_input_forget() const {
        return m_input_forget;
    }
    LSTMWeightsFormat get_weights_format() const {
        return m_weights_format;
    }

private:
    direction m_direction;
    bool m_input_forget;
    LSTMWeightsFormat m_weights_format;
};
}  // namespace v0

namespace v5 {
///
/// \brief      Class for lstm sequence node.
///
/// \note       It follows notation and equations defined as in ONNX standard:
///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
///
/// \sa         LSTMCell, RNNCell, GRUCell
///
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API LSTMSequence : public util::RNNCellBase {
public:
    OPENVINO_OP("LSTMSequence", "opset5", util::RNNCellBase);
    LSTMSequence() = default;

    using direction = RecurrentSequenceDirection;

    size_t get_default_output_index() const override {
        return no_default_index();
    }
    explicit LSTMSequence(const Output<Node>& X,
                          const Output<Node>& initial_hidden_state,
                          const Output<Node>& initial_cell_state,
                          const Output<Node>& sequence_lengths,
                          const Output<Node>& W,
                          const Output<Node>& R,
                          const Output<Node>& B,
                          const std::int64_t hidden_size,
                          const direction lstm_direction,
                          const std::vector<float>& activations_alpha = {},
                          const std::vector<float>& activations_beta = {},
                          const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                          const float clip = 0.f)
        : RNNCellBase({X, initial_hidden_state, initial_cell_state, sequence_lengths, W, R, B},
                      hidden_size,
                      clip,
                      activations,
                      activations_alpha,
                      activations_beta),
          m_direction(lstm_direction) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    direction get_direction() const {
        return m_direction;
    }
    void set_direction(const direction& dir) {
        m_direction = dir;
    }

private:
    direction m_direction{direction::FORWARD};
};
}  // namespace v5
}  // namespace op
}  // namespace ov
