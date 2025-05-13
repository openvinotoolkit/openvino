// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_cpu.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

GemmCPU::GemmCPU(const Output<Node>& A,
                 const Output<Node>& B,
                 const size_t offset_a,
                 const size_t offset_b,
                 const size_t offset_c,
                 const std::vector<size_t>& layout_a,
                 const std::vector<size_t>& layout_b,
                 const std::vector<size_t>& layout_c)
    : Brgemm() {
    // We call default ctor of Brgemm class to avoid incorrect shape infer in constructor_validate_and_type_infer() call
    set_arguments({A, B});
    set_output_size(1);
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

GemmCPU::GemmCPU(const Output<Node>& A,
                 const Output<Node>& B,
                 const PortDescriptor& desc_a,
                 const PortDescriptor& desc_b,
                 const PortDescriptor& desc_c,
                 const std::vector<size_t>& layout_a,
                 const std::vector<size_t>& layout_b,
                 const std::vector<size_t>& layout_c)
    : Brgemm() {
    set_arguments({A, B});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}};
    m_output_ports = {{0, desc_c}};
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

void GemmCPU::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                          const std::vector<size_t>& layout_b,
                                                          const std::vector<size_t>& layout_c) {
    INTERNAL_OP_SCOPE(BrgemmCPU_constructor_validate_and_infer_types);

    const std::vector<ov::PartialShape> planar_input_shapes{
        snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
        snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b)};
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), snippets::utils::get_planar_pshape(output_shape, layout_c));
}

void GemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);

    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));
}

std::shared_ptr<Node> GemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GemmCPU>(
        new_args.at(0),
        new_args.at(1),
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

bool GemmCPU::visit_attributes(AttributeVisitor& visitor) {
    Brgemm::visit_attributes(visitor);
    return true;
}
}  // namespace ov::intel_cpu::aarch64
