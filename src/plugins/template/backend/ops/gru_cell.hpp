// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/gru_cell.hpp"
#include "ov_ops/augru_cell.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::GRUCell>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<ET>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<ET>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<ET>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<ET>(),
                                            inputs[4]->get_shape(),
                                            outputs[0]->get_data_ptr<ET>(),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_clip(),
                                            op->get_linear_before_reset());
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::internal::AUGRUCell>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<ET>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<ET>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<ET>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<ET>(),
                                            inputs[4]->get_shape(),
                                            outputs[0]->get_data_ptr<ET>(),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_clip(),
                                            op->get_linear_before_reset(),
                                            inputs[5]->get_data_ptr<ET>());
    return true;
}