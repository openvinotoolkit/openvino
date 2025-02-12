// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
using namespace brgemm_utils;

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, type, offset_a, offset_b, offset_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, type, desc_a, desc_b, desc_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

void BrgemmCPU::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                            const std::vector<size_t>& layout_b,
                                                            const std::vector<size_t>& layout_c) {
    INTERNAL_OP_SCOPE(BrgemmCPU_constructor_validate_and_infer_types);
    validate_inputs();

    const std::vector<ov::PartialShape> planar_input_shapes{
        snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
        snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b)};
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), snippets::utils::get_planar_pshape(output_shape, layout_c));
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    validate_inputs();

    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));
}


void BrgemmCPU::validate_inputs() const {
    OPENVINO_ASSERT(
        implication(one_of(m_type, BRGEMM_TYPE::STAND_ALONE, BRGEMM_TYPE::REPACKING_ONLY), get_input_size() == 2),
        "BrgemmCPU expects 2 inputs in cases, when input precisions are f32|f32, u8|i8 or bf16|bf16 (non-AMX system)");
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCPU>(
        new_args.at(0),
        new_args.at(1),
        1,
        m_type,
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

bool BrgemmCPU::visit_attributes(AttributeVisitor& visitor) {
    Brgemm::visit_attributes(visitor);
    visitor.on_attribute("type", m_type);
    return true;
}
}  // namespace ov::intel_cpu
