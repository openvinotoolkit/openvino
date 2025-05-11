// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/gather_nd_base.hpp"

#include "gather_nd_shape_inference.hpp"

ov::op::util::GatherNDBase::GatherNDBase(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : Op({data, indices}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}

void ov::op::util::GatherNDBase::validate_inputs_and_infer_shape() {
    // check types of input tensors
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);
}

bool ov::op::util::GatherNDBase::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}
