// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <typename Data_t, typename Index_t, typename Count_t>
void execute_unique(const ngraph::HostTensorVector& outputs,
                    const ngraph::HostTensorVector& inputs,
                    const std::shared_ptr<ngraph::op::v10::Unique>& op) {
    const auto maybe_extract_axis = [&op]() {
        std::unique_ptr<int64_t> axis;
        if (op->get_input_size() == 2 && ov::op::util::is_constant(op->input_value(1).get_node())) {
            const auto axis_constant =
                std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
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

template <ngraph::element::Type_t Data_ET>
bool evaluate(const std::shared_ptr<ngraph::op::v10::Unique>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using Data_t = typename ngraph::element_type_traits<Data_ET>::value_type;
    if (op->get_index_element_type() == ngraph::element::i32 && op->get_count_element_type() == ngraph::element::i32) {
        execute_unique<Data_t, int32_t, int32_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ngraph::element::i64 &&
               op->get_count_element_type() == ngraph::element::i64) {
        execute_unique<Data_t, int64_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ngraph::element::i32 &&
               op->get_count_element_type() == ngraph::element::i64) {
        execute_unique<Data_t, int32_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == ngraph::element::i64 &&
               op->get_count_element_type() == ngraph::element::i32) {
        execute_unique<Data_t, int64_t, int32_t>(outputs, inputs, op);
    } else {
        return false;
    }

    return true;
}