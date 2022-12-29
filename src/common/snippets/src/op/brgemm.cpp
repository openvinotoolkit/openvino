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

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B, bool transposed_a, bool transposed_b,
               const size_t offset_a, const size_t offset_b, const size_t offset_c)
    : MemoryAccess({A, B}), m_transposed_a(transposed_a), m_transposed_b(transposed_b) {
    set_output_size(1);
    set_input_port_descriptor({0, offset_a}, 0);
    set_input_port_descriptor({0, offset_b}, 1);
    set_output_port_descriptor({0, offset_c}, 0);
    constructor_validate_and_infer_types();
}

bool Brgemm::visit_attributes(AttributeVisitor& visitor) {
    MemoryAccess::visit_attributes(visitor);
    visitor.on_attribute("transposed_a", m_transposed_a);
    visitor.on_attribute("transposed_b", m_transposed_b);
    return true;
}

void Brgemm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Brgemm_validate_and_infer_types);
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "Brgemm currently supports only static shapes.");

    std::vector<ov::PartialShape> planar_input_shapes = {
            utils::get_port_planar_shape(input_value(0)),
            utils::get_port_planar_shape(input_value(1))
    };

    auto output_shape = get_output_partial_shape(planar_input_shapes);
    const auto& output_layout = utils::get_node_output_layout(this);
    set_output_type(0,
                    get_output_type(),
                    utils::get_reordered_planar_shape(output_shape, output_layout));
}

std::shared_ptr<Node> Brgemm::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Brgemm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Brgemm>(new_args.at(0), new_args.at(1),
                                    m_transposed_a, m_transposed_b,
                                    get_offset_a(), get_offset_b(), get_offset_c());
}

ov::element::Type Brgemm::get_output_type() const {
    const auto element_type_a = get_input_element_type(0);
    const auto element_type_b = get_input_element_type(1);
    const bool is_f32 = utils::everyone_is(element::f32, element_type_a, element_type_b);
    const bool is_int8 = utils::one_of(element_type_a, element::i8, element::u8) && element_type_b == element::i8;
    const bool is_bf16 = utils::everyone_is(element::bf16, element_type_a, element_type_b);
    if (is_f32 || is_bf16) {
       return element::f32;
    } else if (is_int8) {
        return element::i32;
    } else {
        throw ngraph_error("BrgemmCPU node has incompatible input element types: " +
                            element_type_a.get_type_name() +
                            " and " +
                            element_type_b.get_type_name());
    }
}

ov::PartialShape Brgemm::get_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const {
    NGRAPH_CHECK(input_shapes.size() == 2, "BRGEMM expects 2 input shapes for shape inference");
    auto matmul_in0 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shapes[0]);
    auto matmul_in1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shapes[1]);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(matmul_in0, matmul_in1, m_transposed_a, m_transposed_b);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    ov::op::v0::shape_infer(matmul.get(), input_shapes, output_shapes);
    return output_shapes.front();
}

} // namespace op
} // namespace snippets
} // namespace ngraph
