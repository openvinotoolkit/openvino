// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_a.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

BrgemmCopyA::BrgemmCopyA(const Output<Node>& x, BrgemmConfig config,
                         const PortDescriptor& desc_in, const PortDescriptor& desc_out, std::vector<size_t> layout_in)
    : snippets::modifier::MemoryAccess(1, 1), op::Op({x}), m_config(config) {
    set_output_size(1);
    set_input_port_descriptor(desc_in, 0);
    set_output_port_descriptor(desc_out, 0);
    custom_constructor_validate_and_infer_types(std::move(layout_in));
}

BrgemmCopyA::BrgemmCopyA(const Output<Node>& x, BrgemmConfig config,
                         const size_t offset_in, const size_t offset_out, std::vector<size_t> layout_in)
    : BrgemmCopyA(x, std::move(config), PortDescriptor(0, offset_in), PortDescriptor(0, offset_out), layout_in) {}

bool BrgemmCopyA::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmCopyA_visit_attributes);
    visitor.on_attribute("BrgemmConfig", const_cast<BrgemmConfig&>(m_config));
    return MemoryAccess::visit_attributes(visitor);
}

void BrgemmCopyA::custom_constructor_validate_and_infer_types(std::vector<size_t> layout_input) {
    INTERNAL_OP_SCOPE(BrgemmCopyA_ctor_validate_and_infer_types);
    // During ctor call, BrgemmCopyA doesn't know his port descriptors. So we use port descs from source inputs
    set_output_type(0, get_input_element_type(0), snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input));
}

void BrgemmCopyA::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmCopyA_validate_and_infer_types);
    const auto layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    set_output_type(0, get_input_element_type(0), snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout));
}

std::shared_ptr<ov::Node> BrgemmCopyA::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCopyA_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyA>(new_args.at(0), m_config, get_input_port_descriptor(0), get_output_port_descriptor(0),
                                         snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout());
}

BrgemmCopyA::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& op = ov::as_type_ptr<BrgemmCopyA>(n);
    OPENVINO_ASSERT(op, "Got invalid node in BrgemmCopyA::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
}

ov::snippets::IShapeInferSnippets::Result BrgemmCopyA::ShapeInfer::infer(const std::vector<ov::snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    return {{ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout)}, ov::snippets::ShapeInferStatus::success};
}

} // namespace intel_cpu
} // namespace ov