// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matmul.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "matmul_shape_inference.hpp"
#include "openvino/reference/matmul.hpp"

namespace ov {
namespace op {
namespace matmul {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0,
                             const Tensor& arg1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const Shape& out_shape,
                             const bool transpose_a,
                             const bool transpose_b) {
        reference::matmul(arg0.data<const T>(),
                          arg1.data<const T>(),
                          out.data<T>(),
                          shape0,
                          shape1,
                          out_shape,
                          transpose_a,
                          transpose_b);
        return true;
    }
};
}  // namespace matmul

namespace v0 {

MatMul::MatMul(const Output<Node>& A, const Output<Node>& B, const bool& transpose_a, const bool& transpose_b)
    : Op(OutputVector{A, B}),
      m_transpose_a{transpose_a},
      m_transpose_b{transpose_b} {
    constructor_validate_and_infer_types();
}

bool MatMul::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_MatMul_visit_attributes);
    visitor.on_attribute("transpose_a", m_transpose_a);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

std::shared_ptr<Node> MatMul::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_MatMul_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

bool MatMul::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_MatMul_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto out_shape = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
    outputs[0].set_shape(out_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_MatMul_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      matmul::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      out_shape,
                                      m_transpose_a,
                                      m_transpose_b);
}

bool MatMul::has_evaluate() const {
    OV_OP_SCOPE(v0_MatMul_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

void MatMul::validate_and_infer_types() {
    OV_OP_SCOPE(v0_MatMul_validate_and_infer_types);
    element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");

    const auto& A_shape = get_input_partial_shape(0);
    const auto& B_shape = get_input_partial_shape(1);
    const auto output_shapes = shape_infer(this, std::vector<ov::PartialShape>{A_shape, B_shape});
    set_output_type(0, result_et, output_shapes[0]);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
