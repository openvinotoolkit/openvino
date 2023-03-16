// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::NormalizeL2>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                                outputs[0]->get_data_ptr<T>(),
                                                op->get_input_shape(0),
                                                op->get_reduction_axes(),
                                                op->get_eps(),
                                                op->get_eps_mode());
    return true;
}