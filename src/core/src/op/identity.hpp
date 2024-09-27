// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/identity.hpp"

namespace ov {
namespace op {
namespace v15 {

Identity::Identity(const Output<Node>& data, const bool copy) : Op({data}), m_copy(copy) {
    constructor_validate_and_infer_types();
}

bool Identity::Identity::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_Identity_visit_attributes);
    visitor.on_attribute("copy", m_copy);
    return true;
}

void Identity::Identity::validate_and_infer_types() {
    OV_OP_SCOPE(v15_Identity_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    set_output_type(0, get_input_element_type(0), input_shapes[0]);
}

std::shared_ptr<Node> Identity::Identity::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_Identity_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<Identity>(new_args.at(0), m_copy);
}

bool Identity::get_copy() const {
    return m_copy;
}

void Identity::set_copy(const bool copy) {
    m_copy = copy;
}
}  // namespace ov
