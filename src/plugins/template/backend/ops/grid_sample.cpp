// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/grid_sample.hpp"

#include "evaluate_node.hpp"

template <ov::element::Type_t DATA_ET>
bool evaluate(const std::shared_ptr<ov::op::v9::GridSample>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using DT = ov::fundamental_type_for<DATA_ET>;
    const auto& attributes = op->get_attributes();

    switch (op->get_input_element_type(1)) {
    case ov::element::f16:
        ov::reference::grid_sample(outputs[0].data<DT>(),
                                   inputs[0].data<const DT>(),
                                   inputs[1].data<const ov::fundamental_type_for<ov::element::f16>>(),
                                   inputs[0].get_shape(),
                                   inputs[1].get_shape(),
                                   attributes.align_corners,
                                   attributes.mode,
                                   attributes.padding_mode);
        break;
    case ov::element::bf16:
        ov::reference::grid_sample(outputs[0].data<DT>(),
                                   inputs[0].data<const DT>(),
                                   inputs[1].data<const ov::fundamental_type_for<ov::element::bf16>>(),
                                   inputs[0].get_shape(),
                                   inputs[1].get_shape(),
                                   attributes.align_corners,
                                   attributes.mode,
                                   attributes.padding_mode);
        break;
    case ov::element::f32:
        ov::reference::grid_sample(outputs[0].data<DT>(),
                                   inputs[0].data<const DT>(),
                                   inputs[1].data<const ov::fundamental_type_for<ov::element::f32>>(),
                                   inputs[0].get_shape(),
                                   inputs[1].get_shape(),
                                   attributes.align_corners,
                                   attributes.mode,
                                   attributes.padding_mode);
        break;
    case ov::element::f64:
        ov::reference::grid_sample(outputs[0].data<DT>(),
                                   inputs[0].data<const DT>(),
                                   inputs[1].data<const ov::fundamental_type_for<ov::element::f64>>(),
                                   inputs[0].get_shape(),
                                   inputs[1].get_shape(),
                                   attributes.align_corners,
                                   attributes.mode,
                                   attributes.padding_mode);
        break;
    default:
        return false;
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v9::GridSample>(std::shared_ptr<ov::Node> node,
                                           ov::TensorVector& outputs,
                                           const ov::TensorVector& inputs) {
    switch (node->get_output_element_type(0)) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v9::GridSample>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
