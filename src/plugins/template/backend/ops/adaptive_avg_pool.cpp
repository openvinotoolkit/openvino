// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::AdaptiveAvgPool>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::adaptive_avg_pool(inputs[0]->get_data_ptr<T>(),
                                                  outputs[0]->get_data_ptr<T>(),
                                                  inputs[0]->get_shape(),
                                                  op->get_output_shape(0));
    return true;
}