// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/gather_elements.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v6::GatherElements>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::Shape params_shape = inputs[0]->get_shape();
    ov::Shape indices_shape = inputs[1]->get_shape();

    outputs[0]->set_shape(indices_shape);

    if (inputs[1]->get_element_type() == ov::element::i64) {
        ngraph::runtime::reference::gather_elements<T, int64_t>(inputs[0]->get_data_ptr<ET>(),
                                                                inputs[1]->get_data_ptr<int64_t>(),
                                                                outputs[0]->get_data_ptr<ET>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_shape(),
                                                                outputs[0]->get_shape(),
                                                                op->get_axis());
    } else if (inputs[1]->get_element_type() == ov::element::i32) {
        ngraph::runtime::reference::gather_elements<T, int32_t>(inputs[0]->get_data_ptr<ET>(),
                                                                inputs[1]->get_data_ptr<int32_t>(),
                                                                outputs[0]->get_data_ptr<ET>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_shape(),
                                                                outputs[0]->get_shape(),
                                                                op->get_axis());
    } else {
        throw ngraph::ngraph_error("Unexpected indices type");
    }

    return true;
}
