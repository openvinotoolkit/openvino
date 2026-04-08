// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include "element_visitor.hpp"
#include "grouped_matmul_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/grouped_matmul.hpp"

namespace ov::op {
namespace grouped_matmul {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& mat_a,
                             const Tensor& mat_b,
                             const Tensor* offsets_tensor,
                             Tensor& out,
                             const Shape& mat_a_shape,
                             const Shape& mat_b_shape,
                             const Shape& out_shape,
                             size_t num_groups) {
        if (offsets_tensor) {
            // With offsets
            const auto offsets_et = offsets_tensor->get_element_type();
            if (offsets_et == element::i32) {
                reference::grouped_matmul(mat_a.data<const T>(),
                                          mat_b.data<const T>(),
                                          offsets_tensor->data<const int32_t>(),
                                          out.data<T>(),
                                          mat_a_shape,
                                          mat_b_shape,
                                          out_shape,
                                          num_groups);
            } else {
                reference::grouped_matmul(mat_a.data<const T>(),
                                          mat_b.data<const T>(),
                                          offsets_tensor->data<const int64_t>(),
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

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b) : Op(OutputVector{mat_a, mat_b}) {
    constructor_validate_and_infer_types();
}

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b, const Output<Node>& offsets)
    : Op(OutputVector{mat_a, mat_b, offsets}) {
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

    if (get_input_size() == 3) {
        const auto& offsets_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              offsets_et.is_dynamic() || offsets_et == element::i32 || offsets_et == element::i64,
                              "Offsets element type must be i32 or i64. Got: ",
                              offsets_et);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> GroupedMatMul::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_GroupedMatMul_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    if (new_args.size() == 2) {
        return std::make_shared<GroupedMatMul>(new_args.at(0), new_args.at(1));
    } else {
        return std::make_shared<GroupedMatMul>(new_args.at(0), new_args.at(1), new_args.at(2));
    }
}

bool GroupedMatMul::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_GroupedMatMul_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto out_shape = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
    outputs[0].set_shape(out_shape);

    const auto& mat_a_shape = inputs[0].get_shape();
    const auto& mat_b_shape = inputs[1].get_shape();

    // Determine number of groups
    size_t num_groups = 0;
    const Tensor* offsets_tensor = nullptr;

    if (inputs.size() == 3) {
        offsets_tensor = &inputs[2];
        num_groups = inputs[2].get_shape()[0];
    } else if (mat_a_shape.size() == 3) {
        num_groups = mat_a_shape[0];
    } else if (mat_b_shape.size() == 3) {
        num_groups = mat_b_shape[0];
    }

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
