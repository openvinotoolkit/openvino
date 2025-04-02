// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/identity.hpp"

#include <cstring>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/identity.hpp"

namespace ov {
namespace op {
namespace v16 {

Identity::Identity(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

bool Identity::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_Identity_visit_attributes);
    return true;
}

void Identity::validate_and_infer_types() {
    OV_OP_SCOPE(v16_Identity_validate_and_infer_types);

    // Ensure there is exactly one input
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Identity must have exactly one input.");

    // Get input type and shape
    auto input_type = get_input_element_type(0);
    auto input_shape = get_input_partial_shape(0);

    // Add validation for tensor type (support all types for prim::data)
    NODE_VALIDATION_CHECK(this,
                          input_type.is_dynamic() || input_type.is_static(),
                          "Input type must be static or dynamic for Identity, got: ",
                          input_type);

    // Set the output to match the input
    set_output_type(0, input_type, input_shape);
}

std::shared_ptr<Node> Identity::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_Identity_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<Identity>(new_args.at(0));
}

}  // namespace v16
}  // namespace op
}  // namespace ov
