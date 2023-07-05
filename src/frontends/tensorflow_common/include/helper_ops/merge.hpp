// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

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
    OPENVINO_OP("Merge", "ov::frontend::tensorflow::util", InternalOperation);

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
                break;
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
        return cf_marker.eliminated_markers.size() > 0;
    }

    std::vector<uint32_t> get_eliminated_cond_flow_marker() const {
        const auto this_node = this->shared_from_this();
        FRONT_END_GENERAL_CHECK(
            cf_marker_exists(this_node),
            "[TensorFlow Frontend] internal error: Switch node has not set conditional flow marker");
        auto cf_marker = get_cf_marker(this_node);
        return cf_marker.eliminated_markers;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
