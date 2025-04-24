// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_max_pool.hpp"

#include "adaptive_max_pool_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

op::v8::AdaptiveMaxPool::AdaptiveMaxPool(const Output<Node>& data,
                                         const Output<Node>& output_shape,
                                         const element::Type& index_element_type)
    : Op({data, output_shape}),
      m_index_element_type{index_element_type} {
    constructor_validate_and_infer_types();
}

bool op::v8::AdaptiveMaxPool::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_AdaptiveMaxPool_visit_attributes);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}

void op::v8::AdaptiveMaxPool::validate_and_infer_types() {
    OV_OP_SCOPE(v8_AdaptiveMaxPool_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i64 || m_index_element_type == element::i32,
                          "Index element type must be i32 or i64");

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_index_element_type, output_shapes[1]);
}

std::shared_ptr<Node> op::v8::AdaptiveMaxPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_AdaptiveMaxPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v8::AdaptiveMaxPool>(new_args.at(0), new_args.at(1), m_index_element_type);
}

void op::v8::AdaptiveMaxPool::set_index_element_type(const element::Type& type) {
    m_index_element_type = type;
}
}  // namespace ov
