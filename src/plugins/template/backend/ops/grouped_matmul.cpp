// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/grouped_matmul.hpp"

#include "element_visitor.hpp"
#include "evaluate_node.hpp"
#include "grouped_matmul_shape_inference.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/grouped_matmul.hpp"

namespace {

struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <ov::element::Type_t ET, class T = ov::fundamental_type_for<ET>>
    static result_type visit(const ov::Tensor& mat_a,
                             const ov::Tensor& mat_b,
                             const ov::Tensor* offsets_tensor,
                             ov::Tensor& out,
                             const ov::Shape& mat_a_shape,
                             const ov::Shape& mat_b_shape,
                             const ov::Shape& out_shape,
                             size_t num_groups) {
        if (offsets_tensor) {
            const auto offsets_et = offsets_tensor->get_element_type();
            if (offsets_et == ov::element::i32) {
                ov::reference::grouped_matmul(mat_a.data<const T>(),
                                              mat_b.data<const T>(),
                                              offsets_tensor->data<const int32_t>(),
                                              out.data<T>(),
                                              mat_a_shape,
                                              mat_b_shape,
                                              out_shape,
                                              num_groups);
            } else {
                ov::reference::grouped_matmul(mat_a.data<const T>(),
                                              mat_b.data<const T>(),
                                              offsets_tensor->data<const int64_t>(),
                                              out.data<T>(),
                                              mat_a_shape,
                                              mat_b_shape,
                                              out_shape,
                                              num_groups);
            }
        } else {
            ov::reference::grouped_matmul<T, int32_t>(mat_a.data<const T>(),
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

}  // namespace

template <>
bool evaluate_node<ov::op::v17::GroupedMatMul>(std::shared_ptr<ov::Node> node,
                                               ov::TensorVector& outputs,
                                               const ov::TensorVector& inputs) {
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto op = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(node);
    const auto out_shapes = ov::op::v17::shape_infer(op.get(), ov::util::get_tensors_partial_shapes(inputs));
    const auto out_shape = out_shapes[0].to_shape();
    outputs[0].set_shape(out_shape);

    const auto& mat_a_shape = inputs[0].get_shape();
    const auto& mat_b_shape = inputs[1].get_shape();

    size_t num_groups = 0;
    const ov::Tensor* offsets_tensor = nullptr;

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
                                      node.get(),
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f16, bf16, i32, i64),
                                      Evaluate,
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
