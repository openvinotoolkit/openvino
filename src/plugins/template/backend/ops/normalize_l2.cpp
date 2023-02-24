// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/normalize_l2.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/normalize_l2.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::NormalizeL2>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                                outputs[0]->get_data_ptr<T>(),
                                                op->get_input_shape(0),
                                                op->get_reduction_axes(),
                                                op->get_eps(),
                                                op->get_eps_mode());
    return true;
}
