// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/gather_elements.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v6::GatherElements>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::Shape params_shape = inputs[0]->get_shape();
    ngraph::Shape indices_shape = inputs[1]->get_shape();

    outputs[0]->set_shape(indices_shape);

    if (inputs[1]->get_element_type() == ngraph::element::i64) {
        ngraph::runtime::reference::gather_elements<T, int64_t>(inputs[0]->get_data_ptr<ET>(),
                                                                inputs[1]->get_data_ptr<int64_t>(),
                                                                outputs[0]->get_data_ptr<ET>(),
                                                                inputs[0]->get_shape(),
                                                                inputs[1]->get_shape(),
                                                                outputs[0]->get_shape(),
                                                                op->get_axis());
    } else if (inputs[1]->get_element_type() == ngraph::element::i32) {
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