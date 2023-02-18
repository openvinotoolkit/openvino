// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/avg_pool.hpp"

#include "evaluates_map.hpp"
#include "openvino/op/avg_pool.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::AvgPool>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::avg_pool<T>(inputs[0]->get_data_ptr<T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            inputs[0]->get_shape(),
                                            op->get_output_shape(0),
                                            op->get_kernel(),
                                            op->get_strides(),
                                            op->get_pads_begin(),
                                            op->get_pads_end(),
                                            !op->get_exclude_pad());
    return true;
}
