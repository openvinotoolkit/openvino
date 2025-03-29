// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/unique.hpp"

#include "evaluate_node.hpp"

template <typename Data_t, typename Index_t, typename Count_t>
void execute_unique(ov::TensorVector& outputs,
                    const ov::TensorVector& inputs,
                    const std::shared_ptr<ov::op::v10::Unique>& op) {
    const auto maybe_extract_axis = [&op]() {
        std::unique_ptr<int64_t> axis;
        if (op->get_input_size() == 2 && ov::op::util::is_constant(op->input_value(1).get_node())) {
            const auto axis_constant = ov::as_type_ptr<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
            const auto axis_vec = axis_constant->cast_vector<int64_t>();
            axis = std::unique_ptr<int64_t>(new int64_t{axis_vec.at(0)});
        }
        return axis;
    };

    const auto unique_elements = ov::reference::find_unique_elements<Data_t, Index_t, Count_t>(inputs[0].data<Data_t>(),
                                                                                               inputs[0].get_shape(),
                                                                                               maybe_extract_axis(),
                                                                                               op->get_sorted());
    const auto tensor_shapes =
        ov::reference::make_tensor_shapes(unique_elements, inputs[0].get_shape(), maybe_extract_axis());

    auto& out_unique_elements = outputs[0];
    auto& out_indices = outputs[1];
    auto& out_rev_indices = outputs[2];
    auto& out_counts = outputs[3];

    out_unique_elements.set_shape(std::get<0>(tensor_shapes));
    out_indices.set_shape(std::get<1>(tensor_shapes));
    out_rev_indices.set_shape(std::get<2>(tensor_shapes));
    out_counts.set_shape(std::get<1>(tensor_shapes));

    ov::reference::unique(out_unique_elements.data<Data_t>(),
                          out_indices.data<Index_t>(),
                          out_rev_indices.data<Index_t>(),
                          out_counts.data<Count_t>(),
                          inputs[0].data<Data_t>(),
                          inputs[0].get_shape(),
                          std::get<0>(tensor_shapes),
                          unique_elements);
}

template <ov::element::Type_t Data_ET>
bool evaluate(const std::shared_ptr<ov::op::v10::Unique>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using Data_t = typename ov::element_type_traits<Data_ET>::value_type;
    if (op->get_index_element_type() == ov::element::i32 && op->get_count_element_type() == ov::element::i32) {
        execute_unique<Data_t, int32_t, int32_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ov::element::i64 && op->get_count_element_type() == ov::element::i64) {
        execute_unique<Data_t, int64_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ov::element::i32 && op->get_count_element_type() == ov::element::i64) {
        execute_unique<Data_t, int32_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ov::element::i64 && op->get_count_element_type() == ov::element::i32) {
        execute_unique<Data_t, int64_t, int32_t>(outputs, inputs, op);
    } else {
        return false;
    }

    return true;
}

template <>
bool evaluate_node<ov::op::v10::Unique>(std::shared_ptr<ov::Node> node,
                                        ov::TensorVector& outputs,
                                        const ov::TensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ov::op::v1::Select>(node) || ov::is_type<ov::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ov::element::boolean:
        return evaluate<ov::element::boolean>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::i4:
        return evaluate<ov::element::i4>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::i8:
        return evaluate<ov::element::i8>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::i16:
        return evaluate<ov::element::i16>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::i32:
        return evaluate<ov::element::i32>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::i64:
        return evaluate<ov::element::i64>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u1:
        return evaluate<ov::element::u1>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u4:
        return evaluate<ov::element::u4>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u8:
        return evaluate<ov::element::u8>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u16:
        return evaluate<ov::element::u16>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u32:
        return evaluate<ov::element::u32>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    case ov::element::u64:
        return evaluate<ov::element::u64>(ov::as_type_ptr<ov::op::v10::Unique>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
