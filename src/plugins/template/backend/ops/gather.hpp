// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/gather.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v8::Gather>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    if (op->get_input_element_type(1) == ngraph::element::i64) {
        ngraph::runtime::reference::gather<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                                       inputs[1]->get_data_ptr<int64_t>(),
                                                       outputs[0]->get_data_ptr<T>(),
                                                       op->get_input_shape(0),
                                                       op->get_input_shape(1),
                                                       op->get_output_shape(0),
                                                       op->get_axis(),
                                                       op->get_batch_dims());
    } else if (op->get_input_element_type(1) == ngraph::element::i32) {
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