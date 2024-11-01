// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "activation.hpp"
#include <vector>
#include <algorithm>
#include "intel_gpu/graph/serialization/activation_serializer.hpp"
#include "rnn.hpp"


namespace cldnn {

struct lstm_elt : public RNNParams<lstm_elt> {
    CLDNN_DECLARE_PRIMITIVE(lstm_elt)
    using vec_activation = std::vector<activation_func>;
    using vec_activation_param = std::vector<activation_additional_params>;
    using RNNParams::RNNParams;
    lstm_elt() : RNNParams() {}
    lstm_elt(const lstm_elt&) = default;
    /// @brief Constructs lstm layer.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input cell Primitive id containing cell data. Provide empty string if using lstm without cell values.
    /// @param clip Clip threshold. Provide 0 if using lstm without activations clip threshold.
    /// @param input_forget Provide 0 if using lstm without coupled input-forget gates.
    /// @param offset_order. Order of the concatenated weights, recurrent, and bias. ONNX default is iofz [input, output, forget, block].
    /// @param direction default = 0, bidirectional = 1.
    lstm_elt(const primitive_id& id,
             const input_info& x,
             const primitive_id& cell = "",
             const float clip = 0,
             const bool input_forget = 0,
             const std::vector<activation_func> activations = {activation_func::logistic,
                                                               activation_func::hyperbolic_tan,
                                                               activation_func::hyperbolic_tan},
             const std::vector<activation_additional_params> activation_params = {},
             const lstm_weights_order offset_order = lstm_weights_order::iofz,
             const uint32_t direction = 0)
        : RNNParams(id, x, {}, cell, {}, {}, {}, {}, "", "", clip, input_forget, activations, activation_params, offset_order, \
          direction == 0 ? ov::op::RecurrentSequenceDirection::FORWARD : ov::op::RecurrentSequenceDirection::REVERSE) {
        if (!cell.empty())
            input.pop_back();
        }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret;
        if (!initial_cell_state.pid.empty())
            ret.push_back(initial_cell_state);
        return ret;
    }
};
} // namespace cldnn
