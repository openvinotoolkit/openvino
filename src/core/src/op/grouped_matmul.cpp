// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include "element_visitor.hpp"
#include "grouped_matmul_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/reference/grouped_matmul.hpp"

namespace {
ov::Output<ov::Node> make_empty_i32_const() {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
}

ov::Output<ov::Node> make_empty_typed_const(const ov::element::Type& et) {
    const auto use_et = et.is_dynamic() ? ov::element::f32 : et;
    return ov::op::v0::Constant::create(use_et, ov::Shape{0}, std::vector<float>{});
}
}  // namespace

namespace ov::op {
namespace grouped_matmul {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& mat_a,
                             const Tensor& mat_b,
                             const Tensor* offsets_tensor,
                             const Tensor* bias_tensor,
                             Tensor& out,
                             const Shape& mat_a_shape,
                             const Shape& mat_b_shape,
                             const Shape& out_shape,
                             size_t num_groups) {
        const T* bias_ptr = bias_tensor ? bias_tensor->data<const T>() : nullptr;
        if (offsets_tensor) {
            // With offsets
            const auto offsets_et = offsets_tensor->get_element_type();
            if (offsets_et == element::i32) {
                reference::grouped_matmul(mat_a.data<const T>(),
                                          mat_b.data<const T>(),
                                          offsets_tensor->data<const int32_t>(),
                                          bias_ptr,
                                          out.data<T>(),
                                          mat_a_shape,
                                          mat_b_shape,
                                          out_shape,
                                          num_groups);
            } else {
                reference::grouped_matmul(mat_a.data<const T>(),
                                          mat_b.data<const T>(),
                                          offsets_tensor->data<const int64_t>(),
                                          bias_ptr,
                                          out.data<T>(),
                                          mat_a_shape,
                                          mat_b_shape,
                                          out_shape,
                                          num_groups);
            }
        } else {
            // Without offsets (3D×3D case)
            reference::grouped_matmul<T, int32_t>(mat_a.data<const T>(),
                                                  mat_b.data<const T>(),
                                                  nullptr,
                                                  bias_ptr,
                                                  out.data<T>(),
                                                  mat_a_shape,
                                                  mat_b_shape,
                                                  out_shape,
                                                  num_groups);
        }
        return true;
    }
};

}  // namespace grouped_matmul

namespace v17 {

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b)
    : Op(OutputVector{mat_a, mat_b, make_empty_i32_const(), make_empty_typed_const(mat_a.get_element_type())}) {
    constructor_validate_and_infer_types();
}

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b, const Output<Node>& offsets)
    : Op(OutputVector{mat_a, mat_b, offsets, make_empty_typed_const(mat_a.get_element_type())}) {
    constructor_validate_and_infer_types();
}

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a,
                             const Output<Node>& mat_b,
                             const Output<Node>& offsets,
                             const Output<Node>& bias)
    : Op(OutputVector{mat_a, mat_b, offsets, bias}) {
    constructor_validate_and_infer_types();
}

bool GroupedMatMul::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_GroupedMatMul_visit_attributes);
    return true;
}

void GroupedMatMul::validate_and_infer_types() {
    OV_OP_SCOPE(v17_GroupedMatMul_validate_and_infer_types);

    const auto& mat_a_et = get_input_element_type(0);
    const auto& mat_b_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, mat_a_et, mat_b_et),
                          "Arguments do not have the same element type (mat_a element type: ",
                          mat_a_et,
                          ", mat_b element type: ",
                          mat_b_et,
                          ").");

    // Validate offsets (input 2) when not empty (Shape{0})
    const auto& offsets_pshape = get_input_partial_shape(2);
    const bool offsets_is_empty = offsets_pshape.rank().is_static() &&
                                   offsets_pshape.rank().get_length() == 1 &&
                                   offsets_pshape[0].is_static() &&
                                   offsets_pshape[0].get_length() == 0;
    if (!offsets_is_empty) {
        const auto& offsets_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              offsets_et.is_dynamic() || offsets_et == element::i32 || offsets_et == element::i64,
                              "Offsets element type must be i32 or i64. Got: ",
                              offsets_et);
    }

    // Validate bias (input 3) when not empty (Shape{0})
    const auto& bias_pshape = get_input_partial_shape(3);
    const bool bias_is_empty = bias_pshape.rank().is_static() &&
                                bias_pshape.rank().get_length() == 1 &&
                                bias_pshape[0].is_static() &&
                                bias_pshape[0].get_length() == 0;
    if (!bias_is_empty) {
        const auto& bias_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              bias_et.is_dynamic() || bias_et == result_et,
                              "Bias element type must match mat_a/mat_b element type. Got: ",
                              bias_et);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> GroupedMatMul::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_GroupedMatMul_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GroupedMatMul>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool GroupedMatMul::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_GroupedMatMul_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto out_shape = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
    outputs[0].set_shape(out_shape);

    const auto& mat_a_shape = inputs[0].get_shape();
    const auto& mat_b_shape = inputs[1].get_shape();

    // Determine offsets and number of groups.
    // inputs[2] is offsets — Shape{0} means "empty" (Case 2: 3D×3D).
    const bool offsets_is_empty = (inputs[2].get_shape() == Shape{0});
    size_t num_groups = 0;
    const Tensor* offsets_tensor = nullptr;
    if (!offsets_is_empty) {
        offsets_tensor = &inputs[2];
        num_groups = inputs[2].get_shape()[0];
    } else if (mat_a_shape.size() == 3) {
        num_groups = mat_a_shape[0];
    } else if (mat_b_shape.size() == 3) {
        num_groups = mat_b_shape[0];
    }

    // inputs[3] is bias — Shape{0} means no bias.
    const bool bias_is_empty = (inputs[3].get_shape() == Shape{0});
    const Tensor* bias_tensor = bias_is_empty ? nullptr : &inputs[3];

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v17_GroupedMatMul_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f16, bf16, i32, i64),
                                      grouped_matmul::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      offsets_tensor,
                                      bias_tensor,
                                      outputs[0],
                                      mat_a_shape,
                                      mat_b_shape,
                                      out_shape,
                                      num_groups);
}

bool GroupedMatMul::has_evaluate() const {
    OV_OP_SCOPE(v17_GroupedMatMul_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::bf16:
    case element::f32:
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

}  // namespace v17
}  // namespace ov::op
