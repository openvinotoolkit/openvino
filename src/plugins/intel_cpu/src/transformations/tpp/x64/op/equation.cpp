// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "equation.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

EquationTPP::EquationTPP(const OutputVector& arguments, std::vector<OpDescTPP> op_descs) :
                        modifier::TensorProcessingPrimitive(), ov::op::Op(arguments),
                        m_op_descs(std::move(op_descs)) {
    // Initialize input/output ports as memory access ports
    std::set<size_t> ma_iport_idx;
    for (size_t i = 0; i < get_input_size(); i++)
        ma_iport_idx.insert(ma_iport_idx.end(), i);
    ctor_initialize(ma_iport_idx, std::set<size_t>{0});
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> EquationTPP::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    const auto& new_op = std::make_shared<EquationTPP>(new_args, m_op_descs);
    new_op->clone_memory_access_ports(*this);
    return new_op;
}

bool EquationTPP::visit_attributes(AttributeVisitor& visitor) {
    std::ostringstream ss;
    for (size_t i = 0; i + 1 < m_op_descs.size(); i++)
        ss << m_op_descs[i] << ", ";
    ss << m_op_descs.back();
    std::string str = ss.str();
    visitor.on_attribute("op_descs", str);
    return true;
}

void EquationTPP::validate_and_infer_types() {
    element::Type etype = get_input_element_type(0);
    PartialShape shape = get_input_partial_shape(0);
    for (size_t i = 1; i < get_input_size(); i++) {
        OPENVINO_ASSERT(element::Type::merge(etype, etype, get_input_element_type(i)),
                        "Incompatible element types in TPP equation");
        OPENVINO_ASSERT(ov::PartialShape::broadcast_merge_into(shape, get_input_partial_shape(i), ov::op::AutoBroadcastType::NUMPY),
                        "Incompatible element types in TPP equation");
    }
    set_output_type(0, etype, shape);
}

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
