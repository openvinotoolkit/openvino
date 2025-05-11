// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_logical.hpp"

#include "itt.hpp"
#include "openvino/op/util/elementwise_args.hpp"

ov::op::util::BinaryElementwiseLogical::BinaryElementwiseLogical() = default;

ov::op::util::BinaryElementwiseLogical::BinaryElementwiseLogical(const Output<Node>& arg0,
                                                                 const Output<Node>& arg1,
                                                                 const AutoBroadcastSpec& autob)
    : Op({arg0, arg1}),
      m_autob(autob) {}

void ov::op::util::BinaryElementwiseLogical::validate_and_infer_types() {
    OV_OP_SCOPE(v0_util_BinaryElementwiseLogical_validate_and_infer_types);

    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et == element::boolean,
                          "Operands for logical operators must have boolean element type but have element type ",
                          args_et,
                          ".");

    set_output_type(0, element::boolean, args_pshape);
}

bool ov::op::util::BinaryElementwiseLogical::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_util_BinaryElementwiseLogical_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
