// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reorg_yolo.hpp"

#include "itt.hpp"
#include "reorg_yolo_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
ReorgYolo::ReorgYolo(const Output<Node>& input, const Strides& strides) : Op({input}), m_strides(strides) {
    constructor_validate_and_infer_types();
}

ReorgYolo::ReorgYolo(const Output<Node>& input, const size_t stride) : Op({input}), m_strides(2, stride) {
    constructor_validate_and_infer_types();
}

void ReorgYolo::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ReorgYolo_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, !m_strides.empty(), "Stride attribute is required.");

    const auto input_shapes = std::vector<PartialShape>{get_input_partial_shape(0)};
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> ReorgYolo::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ReorgYolo_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReorgYolo>(new_args.at(0), m_strides);
}

bool ReorgYolo::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ReorgYolo_visit_attributes);
    visitor.on_attribute("stride", m_strides);
    return true;
}

void ReorgYolo::set_strides(const size_t stride) {
    m_strides.resize(2);
    std::fill_n(m_strides.begin(), 2, stride);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
