// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/bucketize.hpp"

#include <array>

#include "bucketize_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
op::v3::Bucketize::Bucketize(const Output<Node>& data,
                             const Output<Node>& buckets,
                             const element::Type output_type,
                             const bool with_right_bound)
    : Op({data, buckets}),
      m_output_type(output_type),
      m_with_right_bound(with_right_bound) {
    constructor_validate_and_infer_types();
}

bool op::v3::Bucketize::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_Bucketize_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("with_right_bound", m_with_right_bound);
    return true;
}

void op::v3::Bucketize::validate_and_infer_types() {
    OV_OP_SCOPE(v3_Bucketize_validate_and_infer_types);
    static constexpr std::array<const char*, 2> input_names{"Data", "Buckets"};

    for (size_t i = 0; i < input_names.size(); ++i) {
        const auto& in_et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              in_et.is_real() || in_et.is_integral_number(),
                              input_names[i],
                              " input type must be numeric. Got: ",
                              in_et);
    }

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64. Got: ",
                          m_output_type);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    if (get_input_partial_shape(0).is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
    set_output_type(0, m_output_type, output_shapes[0]);
}

std::shared_ptr<Node> op::v3::Bucketize::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v3_Bucketize_clone_with_new_inputs);
    check_new_args_count(this, inputs);

    return std::make_shared<v3::Bucketize>(inputs.at(0), inputs.at(1), m_output_type, m_with_right_bound);
}
}  // namespace ov
