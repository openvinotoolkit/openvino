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
                     BRGEMM_TYPE type,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : Brgemm(),
      m_type(type) {
    // We call default ctor of Brgemm class to avoid incorrect shape infer in constructor_validate_and_type_infer() call
    set_arguments({A, B});
    set_output_size(1);
    ctor_initialize(std::set<size_t>{0, 1}, std::set<size_t>{0});
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     const Output<Node>& scratch,
                     BRGEMM_TYPE type,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_scratch,
                     const size_t offset_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : Brgemm(),
      m_type(type) {
    set_arguments({A, B, scratch});
    set_output_size(1);
    ctor_initialize(std::set<size_t>{0, 1, 2}, std::set<size_t>{0});
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    set_input_port_descriptor({0, offset_scratch}, 2);
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     BRGEMM_TYPE type,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : Brgemm(),
      m_type(type) {
    set_arguments({A, B});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}};
    m_output_ports = {{0, desc_c}};
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     const Output<Node>& scratch,
                     BRGEMM_TYPE type,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_scratch,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : Brgemm(),
      m_type(type) {
    set_arguments({A, B, scratch});
    set_output_size(1);
    m_input_ports = {{0, desc_a}, {1, desc_b}, {2, desc_scratch}};
    m_output_ports = {{0, desc_c}};
    custom_constructor_validate_and_infer_types(layout_a, layout_b, layout_c);
}

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

    // Additional check for 3rd input
    validate_with_scratchpad();
}

void BrgemmCPU::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCPU_validate_and_infer_types);
    validate_inputs();

    const auto planar_input_shapes = get_planar_input_shapes({input(0), input(1)});
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    set_output_type(0, get_output_type(), get_planar_output_shape(output_shape));

    // Additional check for 3rd input
    validate_with_scratchpad();
}

void BrgemmCPU::validate_with_scratchpad() const {
    // Additional check for 3rd input
    if (with_compensations(m_type)) {
        OPENVINO_ASSERT(get_input_element_type(2) == ov::element::f32,
                        "BRGEMM Scratch with compensations must have FP32 element type");
    } else if (with_amx(m_type)) {
        OPENVINO_ASSERT(get_input_partial_shape(2).is_static(), "BRGEMM Scratch must have static shape");
        OPENVINO_ASSERT(get_input_element_type(2) == ov::element::u8, "BRGEMM Scratch must have U8 element type");
    }
}

void BrgemmCPU::validate_inputs() const {
    OPENVINO_ASSERT(
        implication(one_of(m_type, BRGEMM_TYPE::STAND_ALONE, BRGEMM_TYPE::REPACKING_ONLY), get_input_size() == 2),
        "BrgemmCPU expects 2 inputs in cases, when input precisions are f32|f32, u8|i8 or bf16|bf16 (non-AMX system)");
    OPENVINO_ASSERT(
        implication(one_of(m_type, BRGEMM_TYPE::WITH_COMPENSATIONS, BRGEMM_TYPE::WITH_AMX), get_input_size() == 3),
        "BrgemmCPU expects 3 inputs with input precisions i8|i8 and bf16|bf16 on AMX system");
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::shared_ptr<BrgemmCPU> brgemm;
    if (!with_scratchpad(m_type)) {
        return std::make_shared<BrgemmCPU>(
            new_args.at(0),
            new_args.at(1),
            m_type,
            get_input_port_descriptor(0),
            get_input_port_descriptor(1),
            get_output_port_descriptor(0),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
    }
    return std::make_shared<BrgemmCPU>(
        new_args.at(0),
        new_args.at(1),
        new_args.at(2),
        m_type,
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_input_port_descriptor(2),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

size_t BrgemmCPU::get_offset_scratch() const {
    OPENVINO_ASSERT(with_scratchpad(m_type) && get_input_size() == 3,
                    "Offset of scratchpad must be only in Brgemm with scratchpad on 3rd input");
    return get_input_offset(2);
}

bool BrgemmCPU::visit_attributes(AttributeVisitor& visitor) {
    Brgemm::visit_attributes(visitor);
    visitor.on_attribute("type", m_type);
    return true;
}
}  // namespace ov::intel_cpu
