// Copyright (C) 2018-2023 Intel Corporation
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
namespace v13 {
///
/// \brief      Class for lstm sequence node.
///
///
/// \sa         LSTMCell, RNNCell, GRUCell
///
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MultiLSTMSequence : public util::RNNMultiCellBase {
public:
    OPENVINO_OP("MultiLSTMSequence", "opset13", util::RNNMultiCellBase);
    MultiLSTMSequence() = default;

    using direction = RecurrentSequenceDirection;

    size_t get_default_output_index() const override {
        return no_default_index();
    }
    explicit MultiLSTMSequence(const Output<Node>& X,
                          const Output<Node>& initial_hidden_state,
                          const Output<Node>& initial_cell_state,
                          const Output<Node>& W,
                          const Output<Node>& R,
                          const Output<Node>& B,
                          const std::int64_t hidden_size,
                          const direction lstm_direction,
                          LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                          const std::vector<float>& activations_alpha = {},
                          const std::vector<float>& activations_beta = {},
                          const std::vector<std::string>& activations = {"sigmoid", "tanh", "tanh"},
                          const float clip_threshold = 0.f,
                          const bool input_forget = false)
        : RNNMultiCellBase({X, initial_hidden_state, initial_cell_state, W, R, B},
                      hidden_size,
                      clip_threshold,
                      activations,
                      activations_alpha,
                      activations_beta),
          m_direction(lstm_direction),
          m_weights_format(weights_format) {
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
    LSTMWeightsFormat m_weights_format;
};
}  // namespace v13
}  // namespace op
}  // namespace ov
