// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "internal_operation.hpp"
#include "openvino/frontend/exception.hpp"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for Switch
// Forwards data to the output port determined by predicate
// It has two outputs:
// 1. output_false - forward data to this output in case predicate to be false
// 2. output_true - forward data to this output in case predicate to be true
class Switch : public InternalOperation {
public:
    OPENVINO_OP("Switch", "ov::frontend::tensorflow", InternalOperation);

    Switch(const Output<Node>& data,
           const Output<Node>& pred,
           uint32_t switch_marker,
           const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{data, pred}, 2, "Switch"),
          m_switch_marker(switch_marker) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto data_type = get_input_element_type(0);
        auto data_shape = get_input_partial_shape(0);

        set_output_type(0, data_type, data_shape);
        set_output_type(1, data_type, data_shape);
    }

    bool is_cond_flow_marker_set() const {
        const auto this_node = this->shared_from_this();
        if (!cf_marker_exists(this_node)) {
            return false;
        }

        auto cf_marker = get_cf_marker(this_node);
        return cf_marker.new_markers.size() == 1;
    }

    uint32_t get_cond_flow_marker() const {
        const auto this_node = this->shared_from_this();
        FRONT_END_GENERAL_CHECK(
            cf_marker_exists(this_node),
            "[TensorFlow Frontend] internal error: Switch node has not set conditional flow marker");
        auto cf_marker = get_cf_marker(this_node);
        FRONT_END_GENERAL_CHECK(
            cf_marker.new_markers.size() == 1,
            "[TensorFlow Frontend] internal error: Switch node has incorrect value of conditional flow marker");
        return cf_marker.new_markers.begin()->first;
    }

    uint32_t get_switch_marker() const {
        return m_switch_marker;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 2,
                                      "[TensorFlow Frontend] internal error: Switch expects two inputs");
        auto switch_node = std::make_shared<Switch>(inputs[0], inputs[1], m_switch_marker, m_decoder);
        switch_node->set_attrs(get_attrs());
        return switch_node;
    }

private:
    uint32_t m_switch_marker;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
