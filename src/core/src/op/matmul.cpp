// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/matmul.hpp"

#include <memory>

#include "itt.hpp"
#include "matmul_shape_inference.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/reference/matmul.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::MatMul);

op::MatMul::MatMul(const Output<Node>& A, const Output<Node>& B, const bool& transpose_a, const bool& transpose_b)
    : Op(OutputVector{A, B}),
      m_transpose_a{transpose_a},
      m_transpose_b{transpose_b} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::MatMul::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_MatMul_visit_attributes);
    visitor.on_attribute("transpose_a", m_transpose_a);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

shared_ptr<Node> op::MatMul::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_MatMul_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

namespace matmul {
namespace {
template <element::Type_t ET>
bool evaluate(const op::MatMul* op, const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& output) {
    using T = typename element_type_traits<ET>::value_type;

    ov::Shape arg0_shape = arg0->get_shape();
    ov::Shape arg1_shape = arg1->get_shape();

    std::vector<ov::PartialShape> input_shapes = {arg0_shape, arg1_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    shape_infer(op, input_shapes, output_shapes);

    ov::Shape output_shape = output_shapes[0].to_shape();
    output->set_element_type(arg0->get_element_type());
    output->set_shape(output_shape);

    runtime::reference::matmul<T>(arg0->get_data_ptr<ET>(),
                                  arg1->get_data_ptr<ET>(),
                                  output->get_data_ptr<ET>(),
                                  arg0_shape,
                                  arg1_shape,
                                  output_shape,
                                  op->get_transpose_a(),
                                  op->get_transpose_b());
    return true;
}

bool evaluate_matmul(const op::MatMul* op,
                     const HostTensorPtr& arg0,
                     const HostTensorPtr& arg1,
                     const HostTensorPtr& output) {
    bool rc = true;

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_matmul, i32, op, arg0, arg1, output);
        NGRAPH_TYPE_CASE(evaluate_matmul, i64, op, arg0, arg1, output);
        NGRAPH_TYPE_CASE(evaluate_matmul, u32, op, arg0, arg1, output);
        NGRAPH_TYPE_CASE(evaluate_matmul, u64, op, arg0, arg1, output);
        NGRAPH_TYPE_CASE(evaluate_matmul, f16, op, arg0, arg1, output);
        NGRAPH_TYPE_CASE(evaluate_matmul, f32, op, arg0, arg1, output);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace matmul

bool op::MatMul::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_MatMul_evaluate);
    return matmul::evaluate_matmul(this, inputs[0], inputs[1], outputs[0]);
}

bool op::MatMul::has_evaluate() const {
    OV_OP_SCOPE(v0_MatMul_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

void ngraph::op::v0::MatMul::validate_and_infer_types() {
    OV_OP_SCOPE(v0_MatMul_validate_and_infer_types);
    element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");

    const auto &A_shape = get_input_partial_shape(0), B_shape = get_input_partial_shape(1);
    std::vector<ov::PartialShape> input_shapes = {A_shape, B_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, result_et, output_shapes[0]);
}
