// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/unique.hpp"
#include "openvino/op/util/op_types.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::Gather>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    if (op->get_input_element_type(1) == ov::element::i64) {
        ngraph::runtime::reference::gather<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                       inputs[1]->get_data_ptr<int64_t>(),
                                                       outputs[0]->get_data_ptr<T>(),
                                                       op->get_input_shape(0),
                                                       op->get_input_shape(1),
                                                       op->get_output_shape(0),
                                                       op->get_axis(),
                                                       op->get_batch_dims());
    } else if (op->get_input_element_type(1) == ov::element::i32) {
        ngraph::runtime::reference::gather<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                                       inputs[1]->get_data_ptr<int32_t>(),
                                                       outputs[0]->get_data_ptr<T>(),
                                                       op->get_input_shape(0),
                                                       op->get_input_shape(1),
                                                       op->get_output_shape(0),
                                                       op->get_axis(),
                                                       op->get_batch_dims());
    } else {
        throw ngraph::ngraph_error("Unexpected indices type for Gather operation");
    }
    return true;
}

template <typename Data_t, typename Index_t, typename Count_t>
void execute_unique(const ov::HostTensorVector& outputs,
                    const ov::HostTensorVector& inputs,
                    const std::shared_ptr<ov::op::v10::Unique>& op) {
    const auto maybe_extract_axis = [&op]() {
        std::unique_ptr<int64_t> axis;
        if (op->get_input_size() == 2 && ov::op::util::is_constant(op->input_value(1).get_node())) {
            const auto axis_constant =
                std::dynamic_pointer_cast<ov::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
            const auto axis_vec = axis_constant->cast_vector<int64_t>();
            axis = std::unique_ptr<int64_t>(new int64_t{axis_vec.at(0)});
        }
        return axis;
    };

    const auto unique_elements =
        ngraph::runtime::reference::find_unique_elements<Data_t, Index_t, Count_t>(inputs[0]->get_data_ptr<Data_t>(),
                                                                                   inputs[0]->get_shape(),
                                                                                   maybe_extract_axis(),
                                                                                   op->get_sorted());
    const auto tensor_shapes =
        ngraph::runtime::reference::make_tensor_shapes(unique_elements, inputs[0]->get_shape(), maybe_extract_axis());

    auto& out_unique_elements = outputs[0];
    auto& out_indices = outputs[1];
    auto& out_rev_indices = outputs[2];
    auto& out_counts = outputs[3];

    out_unique_elements->set_shape(std::get<0>(tensor_shapes));
    out_indices->set_shape(std::get<1>(tensor_shapes));
    out_rev_indices->set_shape(std::get<2>(tensor_shapes));
    out_counts->set_shape(std::get<1>(tensor_shapes));

    ngraph::runtime::reference::unique(out_unique_elements->get_data_ptr<Data_t>(),
                                       out_indices->get_data_ptr<Index_t>(),
                                       out_rev_indices->get_data_ptr<Index_t>(),
                                       out_counts->get_data_ptr<Count_t>(),
                                       inputs[0]->get_data_ptr<Data_t>(),
                                       inputs[0]->get_shape(),
                                       std::get<0>(tensor_shapes),
                                       unique_elements);
}

template <ov::element::Type_t Data_ET>
bool evaluate(const std::shared_ptr<ov::op::v10::Unique>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
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
