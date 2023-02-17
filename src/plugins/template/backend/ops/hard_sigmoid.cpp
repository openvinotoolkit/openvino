// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/hard_sigmoid.hpp"
#include "openvino/op/hard_sigmoid.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::HardSigmoid>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::hard_sigmoid<T>(inputs[0]->get_data_ptr<T>(),
                                        inputs[1]->get_data_ptr<const T>()[0],
                                        inputs[2]->get_data_ptr<const T>()[0],
                                        outputs[0]->get_data_ptr<T>(),
                                        shape_size(outputs[0]->get_shape()));
    return true;
}
