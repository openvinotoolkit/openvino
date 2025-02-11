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

bool Identity::Identity::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v16_Identity_visit_attributes);
    return true;
}

void Identity::Identity::validate_and_infer_types() {
    OV_OP_SCOPE(v16_Identity_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 1);

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> Identity::Identity::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_Identity_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<Identity>(new_args.at(0));
}
}  // namespace v16
}  // namespace op
}  // namespace ov
