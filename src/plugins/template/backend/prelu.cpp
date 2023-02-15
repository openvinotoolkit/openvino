// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/prelu.hpp"
#include "openvino/op/prelu.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::PRelu>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::prelu<T>(inputs[0]->get_data_ptr<T>(),
                                 inputs[1]->get_data_ptr<T>(),
                                 outputs[0]->get_data_ptr<T>(),
                                 inputs[0]->get_shape(),
                                 inputs[1]->get_shape());
    return true;
}
