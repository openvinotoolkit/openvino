// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::Mod>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::mod<T>(inputs[0]->get_data_ptr<T>(),
                                       inputs[1]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       inputs[0]->get_shape(),
                                       inputs[1]->get_shape(),
                                       op->get_autob());
    return true;
}