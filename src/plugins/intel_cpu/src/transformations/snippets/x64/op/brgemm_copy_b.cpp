// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_copy_b.hpp"

#include <cassert>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu {
intel_cpu::BrgemmCopyB::BrgemmCopyB() : m_config(brgemm_utils::BrgemmConfig{}) {}

intel_cpu::BrgemmCopyB::BrgemmCopyB(const Output<Node>& x,
                                    const BrgemmConfig& config,
                                    const std::vector<size_t>& layout_input,
                                    const PortDescriptor& desc_in,
                                    const PortDescriptor& desc_out0,
                                    const PortDescriptor& desc_out1)
    : MemoryAccess(MemoryAccess::PortMap{{0, desc_in}},
                   config.with_compensations() ? MemoryAccess::PortMap{{0, desc_out0}, {1, desc_out1}}
                                               : MemoryAccess::PortMap{{0, desc_out0}}),
      op::Op({x}),
      m_config(config) {
    set_output_size(m_config.with_compensations() ? 2 : 1);
    custom_constructor_validate_and_infer_types(layout_input);
}

bool BrgemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(BrgemmRepack_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    auto config = m_config;
    visitor.on_attribute("config", config);
    return true;
}

void BrgemmCopyB::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_input) {
    INTERNAL_OP_SCOPE(BrgemmRepack_ctor_validate_and_infer_types);
    OPENVINO_ASSERT(m_config.with_wei_repacking(), "Unsupported Brgemm config value");  // check
    OPENVINO_ASSERT(m_config.orig_wei_dt() == get_input_element_type(0),
                    "The original weights data type must be equal to the input data type of BrgemmCopyB");
    // During ctor call, BrgemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto planar_pshape = snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input);
    // The data always store in planar shape after repacking
    set_output_type(0, m_config.wei_dt(), planar_pshape);
    // If compensations are needed, they are provided in 2nd output (which is used in BrgemmCPU)
    if (m_config.with_compensations()) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

void BrgemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(BrgemmRepack_validate_and_infer_types);
    OPENVINO_ASSERT(m_config.orig_wei_dt() == get_input_element_type(0),
                    "The original weights data type must be equal to the input data type of BrgemmCopyB");
    const auto port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0));
    const auto shape = ov::Shape(port->get_shape());
    const auto& planar_pshape = snippets::utils::get_planar_pshape(shape, port->get_layout());
    set_output_type(0, m_config.wei_dt(), planar_pshape);
    if (m_config.with_compensations()) {
        set_output_type(1, ov::element::f32, planar_pshape);
    }
}

std::shared_ptr<ov::Node> intel_cpu::BrgemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BrgemmCopyB>(
        new_args.at(0),
        m_config,
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        get_input_port_descriptor(0),
        get_output_port_descriptor(0),
        m_config.with_compensations() ? get_output_port_descriptor(1) : PortDescriptor{});
}

size_t BrgemmCopyB::get_offset_compensations() const {
    assert(m_config.with_compensations() && get_output_size() == 2 &&
           "The offset for compensations must be in BrgemmCopyB only with compensations and 2 outputs!");
    return get_output_offset(1);
}

bool BrgemmCopyB::is_transposed(const std::vector<size_t>& layout) {
    const auto is_transposed = !layout.empty() && layout.back() != layout.size() - 1;
    OPENVINO_ASSERT(IMPLICATION(is_transposed, (layout[layout.size() - 2] == layout.size() - 1)),
                    "supports only N dim placed as last or pre last dimension");
    return is_transposed;
}

BrgemmCopyB::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& brg_copyb = ov::as_type_ptr<BrgemmCopyB>(n);
    OPENVINO_ASSERT(brg_copyb, "Got invalid node in BrgemmCopyB::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
    m_num_outs = brg_copyb->get_output_size();
}

ov::snippets::IShapeInferSnippets::Result BrgemmCopyB::ShapeInfer::infer(
    const std::vector<ov::snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got unexpected number of input shapes");
    const auto planar_shape = ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout);
    std::vector<ov::snippets::VectorDims> new_shapes(m_num_outs, planar_shape);
    return {new_shapes, ov::snippets::ShapeInferStatus::success};
}
}  // namespace ov::intel_cpu
