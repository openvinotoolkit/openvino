// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "openvino/op/gather_nd.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::GatherND>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    if (op->get_input_element_type(1) == ov::element::i64) {
        ngraph::runtime::reference::gather_nd<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                  inputs[1]->get_data_ptr<int64_t>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  static_cast<int>(op->get_batch_dims()));
    } else if (op->get_input_element_type(1) == ov::element::i32) {
        ngraph::runtime::reference::gather_nd<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                                  inputs[1]->get_data_ptr<int32_t>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  static_cast<int>(op->get_batch_dims()));
    } else {
        throw ngraph::ngraph_error("Unexpected indices type for GatherND operation");
    }
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::GatherND>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    if (op->get_input_element_type(1) == ov::element::i64) {
        ngraph::runtime::reference::gather_nd<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                  inputs[1]->get_data_ptr<int64_t>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  static_cast<int>(op->get_batch_dims()));
    } else if (op->get_input_element_type(1) == ov::element::i32) {
        ngraph::runtime::reference::gather_nd<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                                  inputs[1]->get_data_ptr<int32_t>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  inputs[1]->get_shape(),
                                                  outputs[0]->get_shape(),
                                                  static_cast<int>(op->get_batch_dims()));
    } else {
        throw ngraph::ngraph_error("Unexpected indices type for GatherND operation");
    }
    return true;
}
