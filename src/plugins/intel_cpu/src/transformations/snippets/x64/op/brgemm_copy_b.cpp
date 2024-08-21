// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x,
                                    const element::Type src_type,
                                    BRGEMM_TYPE type,
                                    const size_t offset_in,
                                    const size_t offset_out0,
                                    const size_t offset_out1,
                                    std::vector<size_t> layout_input)
    : snippets::modifier::MemoryAccess(1, with_compensations(type) ? 2 : 1),
      op::Op({x}),
      m_type(type),
      m_src_type(src_type) {
    set_output_size(with_compensations(m_type) ? 2 : 1);
    set_input_port_descriptor({0, offset_in}, 0);
    set_output_port_descriptor({0, offset_out0}, 0);
    if (with_compensations(m_type)) {
        set_output_port_descriptor({0, offset_out1}, 1);
    }
    custom_constructor_validate_and_infer_types(std::move(layout_input));
}

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x,
                                    const element::Type src_type,
                                    BRGEMM_TYPE type,
                                    const PortDescriptor& desc_in0,
                                    const PortDescriptor& desc_out0,
                                    const PortDescriptor& desc_out1,
                                    std::vector<size_t> layout_input)
    : snippets::modifier::MemoryAccess(1, with_compensations(type) ? 2 : 1),
      op::Op({x}),
      m_type(type),
      m_src_type(src_type) {
    set_output_size(with_compensations(type) ? 2 : 1);
    set_input_port_descriptor(desc_in0, 0);
    set_output_port_descriptor(desc_out0, 0);
    if (with_compensations(m_type)) {
        set_output_port_descriptor(desc_out1, 1);
    }
    custom_constructor_validate_and_infer_types(std::move(layout_input));
}

bool BrgemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmRepack_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("src_type", m_src_type);
    visitor.on_attribute("type", m_type);
    return true;
}

void BrgemmCopyB::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input) {
    INTERNAL_OP_SCOPE(BrgemmRepack_ctor_validate_and_infer_types);
    OPENVINO_ASSERT(m_type == BRGEMM_TYPE::WITH_COMPENSATIONS || m_type == BRGEMM_TYPE::REPACKING_ONLY,
                    "Unsupported BRGEMM_TYPE value");
    // During ctor call, BrgemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto element_type = get_input_element_type(0);
    validate_element_type(element_type);
    // The data always store in planar shape after repacking
    const auto planar_pshape = snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input);
    // data repacking output
    set_output_type(0, element_type, planar_pshape);
    // If compensations are needed, they are provided in 2nd output (which is used in BrgemmCPU)
    if (with_compensations(m_type)) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

void BrgemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmRepack_validate_and_infer_types);
    const auto& element_type = get_input_element_type(0);
    validate_element_type(element_type);
    const auto port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0));
    const auto shape = ov::Shape(port->get_shape());
    const auto& planar_pshape = snippets::utils::get_planar_pshape(shape, port->get_layout());
    set_output_type(0, element_type, planar_pshape);
    if (with_compensations(m_type)) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

void BrgemmCopyB::validate_element_type(const ov::element::Type& element_type) {
    OPENVINO_ASSERT(one_of(element_type, element::f32, element::bf16, element::i8),
                    "BrgemmCopyB doesn't support element type" + element_type.get_type_name());
}

std::shared_ptr<ov::Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(new_args.at(0), m_src_type, m_type,
                                         get_input_port_descriptor(0),
                                         get_output_port_descriptor(0),
                                         with_compensations(m_type) ? get_output_port_descriptor(1) : PortDescriptor{},
                                         snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout());
}

size_t BrgemmCopyB::get_offset_compensations() const {
    OPENVINO_ASSERT(with_compensations(m_type) && get_output_size() == 2,
                    "The offset for compensations must be in BrgemmCopyB only with compensations and 2 outputs!");
    return get_output_offset(1);
}

BrgemmCopyB::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& brg_copyb = ov::as_type_ptr<BrgemmCopyB>(n);
    OPENVINO_ASSERT(brg_copyb, "Got invalid node in BrgemmCopyB::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
    m_num_outs = brg_copyb->get_output_size();
}

ov::snippets::IShapeInferSnippets::Result BrgemmCopyB::ShapeInfer::infer(const std::vector<ov::snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    const auto planar_shape = ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout);
    std::vector<ov::snippets::VectorDims> new_shapes(m_num_outs, planar_shape);
    return {new_shapes, ov::snippets::ShapeInferStatus::success};
}
} // namespace intel_cpu
} // namespace ov