// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/reshape.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace op {
Reshape::Reshape(const Output<Node>& arg, ov::PartialShape target_shape)
    : Op({arg}), m_target_shape(std::move(target_shape)) {
    constructor_validate_and_infer_types();
}

void Reshape::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), m_target_shape);
}

std::shared_ptr<Node> Reshape::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Reshape);
    check_new_args_count(this, new_args);
    return std::make_shared<Reshape>(new_args.at(0), m_target_shape);
}

bool Reshape::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("target_shape", m_target_shape);
    return true;
}

const ov::PartialShape& Reshape::get_target_shape() const {
    return m_target_shape;
}

void Reshape::set_target_shape(ov::PartialShape shape) {
    m_target_shape = std::move(shape);
}
}// namespace op
}// namespace snippets
}// namespace ov