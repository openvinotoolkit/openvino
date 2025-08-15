// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_copy_b.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

GemmCopyB::GemmCopyB(const Output<Node>& x,
                     const Output<Node>& bias,
                     const PortDescriptor& desc_in0,
                     const PortDescriptor& desc_in1,
                     const PortDescriptor& desc_out0,
                     const std::vector<size_t>& layout_input)
    : snippets::modifier::MemoryAccess(2, 1),
      op::Op({x, bias}) {
    set_output_size(1);
    set_input_port_descriptor(desc_in0, 0);
    set_input_port_descriptor(desc_in1, 1);
    set_output_port_descriptor(desc_out0, 0);
    custom_constructor_validate_and_infer_types(layout_input);
}

bool GemmCopyB::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(GemmCopyB_visit_attributes);
    MemoryAccess::visit_attributes(visitor);
    return true;
}

void GemmCopyB::validate_bias_input() const {
    OPENVINO_ASSERT(get_input_size() == 2, "Expected two inputs");
    const auto bias = input_value(1).get_node_shared_ptr();
    OPENVINO_ASSERT(
        ov::is_type<ov::snippets::op::Buffer>(bias) && bias->get_input_size() == 0 && bias->get_output_size() == 1,
        "GemmCopyB supports only empty buffer on bias input");
    OPENVINO_ASSERT(bias->get_output_partial_shape(0).is_static() && bias->get_output_shape(0) == ov::Shape{1},
                    "GemmCopyB supports only scalar buffer for bias");
    OPENVINO_ASSERT(bias->get_output_element_type(0) == ov::element::u8, "GemmCopyB supports only u8 buffer for bias");
}

void GemmCopyB::custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_input) {
    INTERNAL_OP_SCOPE(GemmRepack_ctor_validate_and_infer_types);
    validate_bias_input();
    // During ctor call, GemmCopyB doesn't know his port descriptors.
    // So we use port descs from source inputs
    const auto& element_type = get_input_element_type(0);
    validate_element_type(element_type);
    // The data always store in planar shape after repacking
    const auto planar_pshape = snippets::utils::get_planar_pshape(get_input_partial_shape(0), layout_input);
    // data repacking output
    set_output_type(0, element_type, planar_pshape);
}

void GemmCopyB::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(GemmRepack_validate_and_infer_types);
    validate_bias_input();
    const auto& element_type = get_input_element_type(0);
    validate_element_type(element_type);
    const auto port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0));
    const auto shape = ov::Shape(port->get_shape());
    const auto& planar_pshape = snippets::utils::get_planar_pshape(shape, port->get_layout());
    set_output_type(0, element_type, planar_pshape);
}

void GemmCopyB::validate_element_type(const ov::element::Type& element_type) {
    OPENVINO_ASSERT(any_of(element_type, element::f32),
                    "GemmCopyB doesn't support element type" + element_type.get_type_name());
}

std::shared_ptr<ov::Node> GemmCopyB::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(GemmRepack_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GemmCopyB>(
        new_args.at(0),
        new_args.at(1),
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout());
}

GemmCopyB::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    const auto& copyb = ov::as_type_ptr<GemmCopyB>(n);
    OPENVINO_ASSERT(copyb, "Got invalid node in GemmCopyB::ShapeInfer");
    m_layout = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->input(0))->get_layout();
}

ov::snippets::IShapeInferSnippets::Result GemmCopyB::ShapeInfer::infer(
    const std::vector<ov::snippets::VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 2, "Got unexpected number of input shapes");
    auto planar_shape = ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_layout);
    std::vector<ov::snippets::VectorDims> new_shapes(1, planar_shape);
    return {new_shapes, ov::snippets::ShapeInferStatus::success};
}
}  // namespace ov::intel_cpu::aarch64
