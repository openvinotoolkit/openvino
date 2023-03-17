// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/lstm_cell.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::LSTMCell>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::lstm_cell_v1<T>(inputs[0]->get_data_ptr<ET>(),
                                                inputs[0]->get_shape(),
                                                inputs[1]->get_data_ptr<ET>(),
                                                inputs[1]->get_shape(),
                                                inputs[2]->get_data_ptr<ET>(),
                                                inputs[2]->get_shape(),
                                                inputs[3]->get_data_ptr<ET>(),
                                                inputs[3]->get_shape(),
                                                inputs[4]->get_data_ptr<ET>(),
                                                inputs[4]->get_shape(),
                                                inputs[5]->get_data_ptr<ET>(),
                                                inputs[5]->get_shape(),
                                                inputs[6]->get_data_ptr<ET>(),
                                                inputs[6]->get_shape(),
                                                outputs[0]->get_data_ptr<ET>(),
                                                outputs[1]->get_data_ptr<ET>(),
                                                op->get_activations()[0],
                                                op->get_activations()[1],
                                                op->get_activations()[2],
                                                op->get_clip(),
                                                op->get_weights_format(),
                                                op->get_input_forget());
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v4::LSTMCell>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::lstm_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                             inputs[0]->get_shape(),
                                             inputs[1]->get_data_ptr<ET>(),
                                             inputs[1]->get_shape(),
                                             inputs[2]->get_data_ptr<ET>(),
                                             inputs[2]->get_shape(),
                                             inputs[3]->get_data_ptr<ET>(),
                                             inputs[3]->get_shape(),
                                             inputs[4]->get_data_ptr<ET>(),
                                             inputs[4]->get_shape(),
                                             inputs[5]->get_data_ptr<ET>(),
                                             inputs[5]->get_shape(),
                                             outputs[0]->get_data_ptr<ET>(),
                                             outputs[1]->get_data_ptr<ET>(),
                                             op->get_activations()[0],
                                             op->get_activations()[1],
                                             op->get_activations()[2],
                                             op->get_clip());
    return true;
}