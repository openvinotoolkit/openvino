// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/utils.hpp"
#include "matmul_shape_inference.hpp"

namespace ngraph {
namespace snippets {
namespace op {

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B, const size_t offset_a, const size_t offset_b, const size_t offset_c)
    : MatMul(), m_offset_a(offset_a), m_offset_b(offset_b), m_offset_c(offset_c) {
    set_arguments({A, B});
    set_output_size(1);
    constructor_validate_and_infer_types();
}

bool Brgemm::visit_attributes(AttributeVisitor& visitor) {
    MatMul::visit_attributes(visitor);
    visitor.on_attribute("offset_a", m_offset_a);
    visitor.on_attribute("offset_b", m_offset_b);
    visitor.on_attribute("offset_c", m_offset_c);
    return true;
}

void Brgemm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Brgemm_validate_and_infer_types);
    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "Brgemm currently supports only static shapes.");

    std::vector<ov::PartialShape> planar_input_shapes;
    for (const auto& in : input_values())
        planar_input_shapes.emplace_back(utils::get_port_planar_shape(in));

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    ov::op::v0::shape_infer(this, planar_input_shapes, output_shapes);
    const auto& output_layout = utils::get_node_output_layout(this);
        output_shapes[0] = utils::get_reordered_planar_shape(output_shapes[0], output_layout);
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> Brgemm::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Brgemm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Brgemm>(new_args.at(0), new_args.at(1), m_offset_a, m_offset_b, m_offset_c);
}

} // namespace op
} // namespace snippets
} // namespace ngraph
