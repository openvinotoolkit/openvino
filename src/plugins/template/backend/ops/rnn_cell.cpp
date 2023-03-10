// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::RNNCell>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::rnn_cell<T>(inputs[0]->get_data_ptr<ET>(),
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
                                            op->get_activations().front(),
                                            op->get_clip());
    return true;
}