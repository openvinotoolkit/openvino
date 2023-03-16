// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::GatherTree>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    ngraph::runtime::reference::gather_tree(inputs[0]->get_data_ptr<const char>(),
                                            inputs[1]->get_data_ptr<const char>(),
                                            inputs[2]->get_data_ptr<const char>(),
                                            inputs[3]->get_data_ptr<const char>(),
                                            outputs[0]->get_data_ptr<char>(),
                                            op->get_input_shape(0),
                                            op->get_input_shape(1),
                                            op->get_input_shape(2),
                                            op->get_input_shape(3),
                                            inputs[1]->get_element_type());
    return true;
}