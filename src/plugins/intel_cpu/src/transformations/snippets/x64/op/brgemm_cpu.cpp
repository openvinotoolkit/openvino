// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"

#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
using namespace brgemm_utils;

size_t BrgemmCPU::compute_main_inputs_count(const BRGEMM_TYPE type) {
    switch (type) {
    case BRGEMM_TYPE::STAND_ALONE:
    case BRGEMM_TYPE::REPACKING_ONLY:
        return 2;
    case BRGEMM_TYPE::WITH_AMX:
    case BRGEMM_TYPE::WITH_COMPENSATIONS:
        return 3;
    default:
        OPENVINO_THROW("Unexpected brgemm type!");
    }
}

BrgemmCPU::BrgemmCPU(const ov::OutputVector& inputs,
                     BRGEMM_TYPE type,
                     const std::vector<PortDescriptor>& input_descs,
                     const PortDescriptor& output_desc,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c,
                     PostopsConfig post_ops)
    : Brgemm(),
      m_type(type),
      m_post_ops(std::move(post_ops)),
      m_main_inputs_count(compute_main_inputs_count(type)) {
    set_arguments(inputs);
    set_output_size(1);

    std::set<size_t> input_memory_access_ports;
    for (size_t i = 0; i < inputs.size(); ++i) {
        input_memory_access_ports.insert(i);
    }
    ctor_initialize(input_memory_access_ports, std::set<size_t>{0});

    if (!input_descs.empty()) {
        OPENVINO_ASSERT(input_descs.size() == inputs.size(),
                        "Count of input descriptors must be equal to count of inputs");
        for (size_t i = 0; i < input_descs.size(); ++i) {
            set_input_port_descriptor(input_descs[i], i);
        }
    } else {
        for (size_t i = 0; i < inputs.size(); ++i) {
            set_input_port_descriptor({0, 0}, i);
        }
    }
    set_output_port_descriptor(output_desc, 0);
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
    const auto expected_input_size = m_main_inputs_count + m_post_ops.size();
    OPENVINO_ASSERT(get_input_size() == expected_input_size,
                    "BrgemmCPU expects ",
                    expected_input_size,
                    " inputs whereas it got ",
                    get_input_size(),
                    " inputs");
}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCPU>(
        new_args,
        m_type,
        get_input_port_descriptors(),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout(),
        m_post_ops);
}

size_t BrgemmCPU::get_offset_scratch() const {
    OPENVINO_ASSERT(with_scratchpad(m_type) && m_main_inputs_count == 3,
                    "Offset of scratchpad must be only in Brgemm with scratchpad on 3rd input");
    return get_input_offset(2);
}

bool BrgemmCPU::visit_attributes(AttributeVisitor& visitor) {
    Brgemm::visit_attributes(visitor);
    visitor.on_attribute("type", m_type);
    return true;
}

ov::element::Type BrgemmCPU::get_output_type() const {
    return m_post_ops.empty() ? Brgemm::get_output_type() : input_values().back().get_element_type();
}

ov::OutputVector BrgemmCPU::get_postop_inputs() const {
    const auto& input_values = this->input_values();
    return ov::OutputVector(input_values.begin() + m_main_inputs_count, input_values.end());
}
}  // namespace ov::intel_cpu
