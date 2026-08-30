// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"

#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c) {
    set_arguments({A, B});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}};
    m_output_ports = {{0, desc_c}};
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

void BrgemmCPU::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                            const std::vector<size_t>& layout_b,
                                                            const std::vector<size_t>& layout_c) {
    INTERNAL_OP_SCOPE(BrgemmCPU_constructor_validate_and_infer_types);
    validate_element_types(get_input_element_type(0), get_input_element_type(1));
    const std::vector<ov::PartialShape> planar_input_shapes{
        snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_a),
        snippets::utils::get_planar_pshape(get_input_partial_shape(1), layout_b)};
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), snippets::utils::get_planar_pshape(output_shape, layout_c));
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    validate_element_types(get_input_element_type(0), get_input_element_type(1));
    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));
}

void BrgemmCPU::validate_element_types(const ov::element::Type& type_a, const ov::element::Type& type_b) {
    OPENVINO_ASSERT(type_a == type_b && snippets::utils::any_of(type_a, element::f32, element::bf16, element::f16),
                    "BrgemmCPU supports only matching f32, bf16, or f16 input element types");
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCPU>(
        new_args.at(0),
        new_args.at(1),
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

}  // namespace ov::intel_cpu
