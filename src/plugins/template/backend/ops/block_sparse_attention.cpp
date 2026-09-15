// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/block_sparse_attention.hpp"

#include "block_sparse_attention_shape_inference.hpp"
#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/block_sparse_attention.hpp"

namespace {

template <ov::element::Type_t ET, ov::element::Type_t ETIndex>
bool evaluate(const std::shared_ptr<ov::op::v17::BlockSparseAttention>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    using TIndex = typename ov::element_type_traits<ETIndex>::value_type;

    const bool has_mask = inputs.size() >= 5;
    const bool has_scale = inputs.size() == 6;
    const char* mask = has_mask ? inputs[4].data<const char>() : nullptr;
    const T* scale = has_scale ? inputs[5].data<const T>() : nullptr;

    // Hack below is needed to support dynamic shapes in the reference implementation, mirroring
    // ScaledDotProductAttention's own template-plugin evaluate: at this point the *actual* fully
    // static input shapes are known, while validate_and_infer_types() may only have seen
    // partially dynamic ones.
    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape = ov::op::v17::shape_infer(op.get(), input_shapes).front().to_shape();
    outputs[0].set_shape(output_shape);

    ov::reference::block_sparse_attention<T, TIndex>(inputs[0].data<const T>(),
                                                      inputs[1].data<const T>(),
                                                      inputs[2].data<const T>(),
                                                      inputs[3].data<const TIndex>(),
                                                      mask,
                                                      scale,
                                                      outputs[0].data<T>(),
                                                      op->get_causal(),
                                                      op->get_block_size(),
                                                      inputs[0].get_shape(),
                                                      inputs[1].get_shape(),
                                                      inputs[2].get_shape(),
                                                      inputs[3].get_shape());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate_by_index_type(const std::shared_ptr<ov::op::v17::BlockSparseAttention>& op,
                            ov::TensorVector& outputs,
                            const ov::TensorVector& inputs) {
    switch (inputs[3].get_element_type()) {
    case ov::element::i32:
        return evaluate<ET, ov::element::i32>(op, outputs, inputs);
    case ov::element::i64:
        return evaluate<ET, ov::element::i64>(op, outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled 'block_indices' data type ",
                       inputs[3].get_element_type(),
                       " in evaluate_node()");
    }
}

}  // namespace

template <>
bool evaluate_node<ov::op::v17::BlockSparseAttention>(std::shared_ptr<ov::Node> node,
                                                      ov::TensorVector& outputs,
                                                      const ov::TensorVector& inputs) {
    const auto op = ov::as_type_ptr<ov::op::v17::BlockSparseAttention>(node);
    switch (node->get_input_element_type(0)) {
    case ov::element::bf16:
        return evaluate_by_index_type<ov::element::bf16>(op, outputs, inputs);
    case ov::element::f16:
        return evaluate_by_index_type<ov::element::f16>(op, outputs, inputs);
    case ov::element::f32:
        return evaluate_by_index_type<ov::element::f32>(op, outputs, inputs);
    case ov::element::f64:
        return evaluate_by_index_type<ov::element::f64>(op, outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_input_element_type(0), " in evaluate_node()");
    }
}
