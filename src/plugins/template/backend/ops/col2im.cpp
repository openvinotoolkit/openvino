// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/col2im.hpp"

#include "col2im_shape_inference.hpp"
#include "element_visitor.hpp"
#include "evaluate_node.hpp"

template <ov::element::Type_t ET_data, ov::element::Type_t ET_idx>
bool evaluate_index_type(const std::shared_ptr<ov::op::v15::Col2Im>& op,
                         ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) {
    using T_data = typename ov::element_type_traits<ET_data>::value_type;
    using T_idx = typename ov::element_type_traits<ET_idx>::value_type;
    const std::vector<ov::PartialShape> input_shapes{op->get_input_shape(0),
                                                     op->get_input_shape(1),
                                                     op->get_input_shape(2)};
    const auto output_shape =
        ov::op::v15::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    outputs.front().set_shape(output_shape);
    ov::reference::col2im(inputs[0].data<const T_data>(),
                          inputs[0].get_shape(),
                          inputs[1].data<const T_idx>(),
                          inputs[2].data<const T_idx>(),
                          outputs[0].data<T_data>(),
                          op->get_strides(),
                          op->get_dilations(),
                          op->get_pads_begin(),
                          op->get_pads_end());
    return true;
}

template <ov::element::Type_t ET_data>
bool evaluate_data_type(const std::shared_ptr<ov::op::v15::Col2Im>& op,
                        ov::TensorVector& outputs,
                        const ov::TensorVector& inputs) {
    const auto& index_type = op->get_input_element_type(1);
    using ov::op::v15::Col2Im;
    using namespace ov::element;
    switch (index_type) {
    case i32:
        return evaluate_index_type<ET_data, i32>(ov::as_type_ptr<Col2Im>(op), outputs, inputs);
    case i64:
        return evaluate_index_type<ET_data, i64>(ov::as_type_ptr<Col2Im>(op), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled index type ", index_type, " in evaluate_node()");
    }
}

template <>
bool evaluate_node<ov::op::v15::Col2Im>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    using ov::op::v15::Col2Im;
    using namespace ov::element;
    switch (element_type) {
    case i8:
        return evaluate_data_type<i8>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case i32:
        return evaluate_data_type<i32>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case i64:
        return evaluate_data_type<i64>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case u8:
        return evaluate_data_type<u8>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case u32:
        return evaluate_data_type<u32>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case u64:
        return evaluate_data_type<u64>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case f16:
        return evaluate_data_type<f16>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    case f32:
        return evaluate_data_type<f32>(ov::as_type_ptr<Col2Im>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node()");
    }
}
