// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::tpp::op {

BrgemmTPP::BrgemmTPP(const Output<Node>& A,
                     const Output<Node>& B,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_c,
                     std::vector<size_t> layout_a,
                     std::vector<size_t> layout_b,
                     std::vector<size_t> layout_c,
                     const float beta)
    : MemoryAccess(std::set<size_t>{0, 1}, std::set<size_t>{0}),
      modifier::TensorProcessingPrimitive(),
      Brgemm(A, B, offset_a, offset_b, offset_c, std::move(layout_a), std::move(layout_b), std::move(layout_c)) {
    set_beta(beta);
}

BrgemmTPP::BrgemmTPP(const Output<Node>& A,
                     const Output<Node>& B,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_c,
                     std::vector<size_t> layout_a,
                     std::vector<size_t> layout_b,
                     std::vector<size_t> layout_c,
                     const float beta)
    : MemoryAccess(PortMap{{0, desc_a}, {1, desc_b}}, PortMap{{0, desc_c}}),
      modifier::TensorProcessingPrimitive(),
      Brgemm(A, B, desc_a, desc_b, desc_c, std::move(layout_a), std::move(layout_b), std::move(layout_c)) {
    set_beta(beta);
}

std::shared_ptr<Node> BrgemmTPP::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmTPP_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmTPP>(
        new_args.at(0),
        new_args.at(1),
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout(),
        m_beta);
}

bool BrgemmTPP::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("beta", m_beta);
    TensorProcessingPrimitive::visit_attributes(visitor);
    return Brgemm::visit_attributes(visitor);
}

}  // namespace ov::intel_cpu::tpp::op
