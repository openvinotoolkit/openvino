// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "helper_ops/switch.hpp"
#include "internal_operation.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tf_utils.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Forwards the value of an available tensor from inputs to output
// It has two outputs:
// 1. output that is available by one of the inputs
// 2. value_index - index of available input
class Merge : public InternalOperation {
public:
    OPENVINO_OP("Merge", "ov::frontend::tensorflow", InternalOperation);

    Merge(OutputVector inputs, const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, inputs, 2, "Merge") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        ov::element::Type output_data_type = ov::element::dynamic;
        ov::PartialShape output_data_shape = ov::PartialShape::dynamic();

        auto input_size = get_input_size();
        for (size_t input_ind = 0; input_ind < input_size; ++input_ind) {
            auto input_type = get_input_element_type(input_ind);
            if (input_type.is_static()) {
                output_data_type = input_type;
            }

            auto input_shape = get_input_partial_shape(input_ind);
            if (input_shape.rank().is_dynamic()) {
                continue;
            }

            if (output_data_shape.rank().is_dynamic()) {
                // firstly met shape of static rank
                // immediately use this shape of static rank
                output_data_shape = input_shape;
            } else if (output_data_shape.rank().is_static() &&
                       output_data_shape.rank().get_length() != input_shape.rank().get_length()) {
                // different inputs have different rank means output must be of a dynamic rank
                output_data_shape = ov::PartialShape::dynamic();
                break;
            } else {
                auto output_rank = output_data_shape.rank().get_length();
                for (int64_t dim_ind = 0; dim_ind < output_rank; ++dim_ind) {
                    if (input_shape[dim_ind] != output_data_shape[dim_ind]) {
                        // different inputs can have different dimensions so it must combine them
                        output_data_shape[dim_ind] = ov::Dimension::dynamic();
                    }
                }
            }
        }

        set_output_type(0, output_data_type, output_data_shape);
        set_output_type(1, ov::element::i32, {});
    }

    bool is_cond_flow_eliminated() const {
        const auto this_node = this->shared_from_this();
        if (!cf_marker_exists(this_node)) {
            return false;
        }
        auto cf_marker = get_cf_marker(this_node);
        return cf_marker.merge_eliminated_markers.size() > 0;
    }

    std::vector<uint32_t> get_eliminated_cond_flow_marker() const {
        std::vector<uint32_t> resulted_indices;
        const auto this_node = this->shared_from_this();
        FRONT_END_GENERAL_CHECK(
            cf_marker_exists(this_node),
            "[TensorFlow Frontend] internal error: Switch node has not set conditional flow marker");
        auto cf_marker = get_cf_marker(this_node);
        for (const auto& eliminated_marker : cf_marker.merge_eliminated_markers) {
            resulted_indices.push_back(eliminated_marker.first);
        }
        return resulted_indices;
    }

    std::unordered_set<std::shared_ptr<Switch>> get_switch_nodes_set_by_cond_index(uint32_t switch_marker) {
        const auto this_node = this->shared_from_this();
        FRONT_END_GENERAL_CHECK(
            cf_marker_exists(this_node),
            "[TensorFlow Frontend] internal error: Switch node has not set conditional flow marker");
        auto cf_marker = get_cf_marker(this_node);
        FRONT_END_GENERAL_CHECK(
            cf_marker.merge_eliminated_markers.count(switch_marker) > 0,
            "[TensorFlow Frontend] internal error: no Switch node with requiested conditional flow marker");
        return cf_marker.merge_eliminated_markers[switch_marker];
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto merge_node = std::make_shared<Merge>(inputs, m_decoder);
        merge_node->set_attrs(get_attrs());
        return merge_node;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
