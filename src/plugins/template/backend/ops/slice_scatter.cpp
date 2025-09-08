// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/slice_scatter.hpp"

#include "evaluate_node.hpp"
#include "slice_scatter_shape_inference.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v15::SliceScatter>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    ov::op::v15::shape_infer(op.get(), input_shapes, ov::make_tensor_accessor(inputs));
    ov::reference::slice_scatter(
        static_cast<const char*>(inputs[0].data()),
        op->get_input_shape(0),
        static_cast<const char*>(inputs[1].data()),
        op->get_input_shape(1),
        static_cast<char*>(outputs[0].data()),
        inputs[0].get_element_type().size(),
        ov::get_tensor_data_as<int64_t>(inputs[2]),
        ov::get_tensor_data_as<int64_t>(inputs[4]),
        inputs.size() > 5 ? ov::util::try_get_normalized_axis_vector(inputs[5], op->get_input_shape(0).size(), *op)
                          : ov::AxisVector{});
    return true;
}

template <>
bool evaluate_node<ov::op::v15::SliceScatter>(std::shared_ptr<ov::Node> node,
                                              ov::TensorVector& outputs,
                                              const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v15::SliceScatter>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
