// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/sparse_fill_empty_rows.hpp"

#include "element_visitor.hpp"
#include "evaluate_node.hpp"
#include "sparse_fill_empty_rows_shape_inference.hpp"

template <ov::element::Type_t ET_data, ov::element::Type_t ET_idx>
bool evaluate_index_type(const std::shared_ptr<ov::op::v16::SparseFillEmptyRows>& op,
                         ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) {
    using T_data = typename ov::element_type_traits<ET_data>::value_type;
    using T_idx = typename ov::element_type_traits<ET_idx>::value_type;

    auto input_shapes = std::vector<ov::PartialShape>{
        op->get_input_shape(0),  // values
        op->get_input_shape(1),  // dense_shape
        op->get_input_shape(2),  // indices
        op->get_input_shape(3)   // default_value
    };

    const auto output_shapes = ov::op::v16::shape_infer(op.get(), input_shapes, make_tensor_accessor(inputs));
    outputs[0].set_shape(output_shapes[0].to_shape());  // output_indices
    outputs[1].set_shape(output_shapes[1].to_shape());  // output_values
    outputs[2].set_shape(output_shapes[2].to_shape());  // empty_row_indicator

    auto values = inputs[0].data<const T_data>();
    const T_idx* dense_shape = inputs[1].data<const T_idx>();
    const T_idx* indices = inputs[2].data<const T_idx>();
    const T_data default_value = *inputs[3].data<const T_data>();

    T_idx* output_indices = outputs[0].data<T_idx>();
    T_data* output_values = outputs[1].data<T_data>();
    bool* empty_row_indicator = outputs[2].data<bool>();

    const size_t values_size = inputs[0].get_shape()[0];

    ov::reference::sparse_fill_empty_rows(values,
                                          values_size,
                                          dense_shape,
                                          indices,
                                          default_value,
                                          output_indices,
                                          output_values,
                                          empty_row_indicator);
    return true;
}

template <ov::element::Type_t ET_data>
bool evaluate_data_type(const std::shared_ptr<ov::op::v16::SparseFillEmptyRows>& op,
                        ov::TensorVector& outputs,
                        const ov::TensorVector& inputs) {
    const auto& index_type = op->get_input_element_type(1);
    using ov::op::v16::SparseFillEmptyRows;
    using namespace ov::element;

    switch (index_type) {
    case i32:
        return evaluate_index_type<ET_data, i32>(ov::as_type_ptr<SparseFillEmptyRows>(op), outputs, inputs);
    case i64:
        return evaluate_index_type<ET_data, i64>(ov::as_type_ptr<SparseFillEmptyRows>(op), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled index type ", index_type, " in evaluate_node() for SparseFillEmptyRows");
    }
}

template <>
bool evaluate_node<ov::op::v16::SparseFillEmptyRows>(std::shared_ptr<ov::Node> node,
                                                     ov::TensorVector& outputs,
                                                     const ov::TensorVector& inputs) {
    using ov::op::v16::SparseFillEmptyRows;
    using namespace ov::element;

    switch (const auto& element_type = node->get_output_element_type(1); element_type) {
    case i8:
        return evaluate_data_type<i8>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case i32:
        return evaluate_data_type<i32>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case i64:
        return evaluate_data_type<i64>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case u8:
        return evaluate_data_type<u8>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case u32:
        return evaluate_data_type<u32>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case u64:
        return evaluate_data_type<u64>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case f16:
        return evaluate_data_type<f16>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    case f32:
        return evaluate_data_type<f32>(ov::as_type_ptr<SparseFillEmptyRows>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", element_type, " in evaluate_node() for SparseFillEmptyRows");
    }
}
