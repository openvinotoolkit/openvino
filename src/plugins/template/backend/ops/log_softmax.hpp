// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v5::LogSoftmax>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    int64_t i_axis = op->get_axis();
    if (i_axis < 0) {
        i_axis += inputs[0]->get_partial_shape().rank().get_length();
    }
    ngraph::runtime::reference::log_softmax<T>(inputs[0]->get_data_ptr<const T>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               op->get_output_shape(0),
                                               ngraph::AxisSet{(size_t)i_axis});
    return true;
}