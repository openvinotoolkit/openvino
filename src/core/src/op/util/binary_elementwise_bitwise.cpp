// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_bitwise.hpp"

#include "itt.hpp"
#include "openvino/op/util/elementwise_args.hpp"

ov::op::util::BinaryElementwiseBitwise::BinaryElementwiseBitwise() = default;

ov::op::util::BinaryElementwiseBitwise::BinaryElementwiseBitwise(const Output<Node>& arg0,
                                                                 const Output<Node>& arg1,
                                                                 const AutoBroadcastSpec& autob)
    : Op({arg0, arg1}),
      m_autob(autob) {}

void ov::op::util::BinaryElementwiseBitwise::validate_and_infer_types() {
    OV_OP_SCOPE(v0_util_BinaryElementwiseBitwise_validate_and_infer_types);
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    const auto& args_et = std::get<0>(args_et_pshape);
    const auto& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et.is_integral(),
                          "The element type of the input tensor must be integer or boolean.");

    set_output_type(0, args_et, args_pshape);
}

bool ov::op::util::BinaryElementwiseBitwise::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_util_BinaryElementwiseBitwise_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}
const ov::op::AutoBroadcastSpec& ov::op::util::BinaryElementwiseBitwise::get_autob() const {
    return m_autob;
}
void ov::op::util::BinaryElementwiseBitwise::set_autob(const AutoBroadcastSpec& autob) {
    m_autob = autob;
}
