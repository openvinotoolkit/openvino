// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "openvino/op/batch_norm.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::BatchNormInference>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::batch_norm_inference<T>(static_cast<float>(op->get_eps_value()),
                                                inputs[2]->get_data_ptr<T>(),
                                                inputs[0]->get_data_ptr<T>(),
                                                inputs[1]->get_data_ptr<T>(),
                                                inputs[3]->get_data_ptr<T>(),
                                                inputs[4]->get_data_ptr<T>(),
                                                outputs[0]->get_data_ptr<T>(),
                                                inputs[2]->get_shape());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v5::BatchNormInference>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::batch_norm_inference<T>(static_cast<float>(static_cast<float>(op->get_eps_value())),
                                                inputs[0]->get_data_ptr<const T>(),
                                                inputs[1]->get_data_ptr<const T>(),
                                                inputs[2]->get_data_ptr<const T>(),
                                                inputs[3]->get_data_ptr<const T>(),
                                                inputs[4]->get_data_ptr<const T>(),
                                                outputs[0]->get_data_ptr<T>(),
                                                op->get_input_shape(0));
    return true;
}
