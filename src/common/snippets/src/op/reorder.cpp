// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/reorder.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace op {

Reorder::Reorder(const Output<Node>& arg, std::vector<size_t> order)
    : ShapeInferOp({arg}) {
    custom_constructor_validate_and_infer_types(std::move(order));
}

void Reorder::custom_constructor_validate_and_infer_types(std::vector<size_t> order) {
    INTERNAL_OP_SCOPE(Reorder_constructor_validate_and_infer_types);

    const auto& input_pshape = get_input_partial_shape(0);
    OPENVINO_ASSERT(input_pshape.rank().is_static() && input_pshape.size() == order.size(),
                   "Incompatible shape and order sizes");

    // During ctor call, Reorder doesn't know his port descriptors.
    // So we use explicit layouts from parameters
    set_output_type(0, get_input_element_type(0), ov::snippets::utils::get_planar_pshape(input_pshape, order));
}

void Reorder::validate_and_infer_types() {
    const auto& input_pshape = get_input_partial_shape(0);
    const auto& order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    OPENVINO_ASSERT(input_pshape.rank().is_static() && input_pshape.size() == order.size(),
                    "Incompatible shape and order sizes");
    const auto output_pshape = utils::get_planar_pshape(get_input_partial_shape(0), order);
    set_output_type(0, get_input_element_type(0), output_pshape);
}

std::shared_ptr<Node> Reorder::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Reorder);
    check_new_args_count(this, new_args);
    const auto& order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    return std::make_shared<Reorder>(new_args.at(0), order);
}

bool Reorder::visit_attributes(AttributeVisitor& visitor) {
    auto order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout();
    visitor.on_attribute("target_order", order);
    return true;
}

Reorder::ShapeInfer::ShapeInfer(const std::shared_ptr<Node>& n) {
    const auto& op = as_type_ptr<ov::snippets::op::Reorder>(n);
    OPENVINO_ASSERT(op, "Invalid node passed to ReorderShapeInfer.");
    m_target_order = lowered::PortDescriptorUtils::get_port_descriptor_ptr(op->input(0))->get_layout();
}

IShapeInferSnippets::Result Reorder::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of shapes is passed in ReorderShapeInfer");
    return {{ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_target_order)}, ShapeInferStatus::success};
}

}// namespace op
}// namespace snippets
}// namespace ov