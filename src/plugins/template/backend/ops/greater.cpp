// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "openvino/op/greater.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Greater>& op, const ov::HostTensorVector& outputs, const ov::HostTensorVector& inputs) {
    const auto in0_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto in1_data_ptr = inputs[1]->get_data_ptr<ET>();
    const auto out_data_ptr = outputs[0]->get_data_ptr<ov::element::Type_t::boolean>();
    const auto in0_shape = inputs[0]->get_shape();
    const auto in1_shape = inputs[1]->get_shape();
    const auto broadcast_spec = op->get_autob();
    ngraph::runtime::reference::greater<typename ov::element_type_traits<ET>::value_type,
                                typename ov::element_type_traits<ov::element::Type_t::boolean>::value_type>(in0_data_ptr,
                                                                                                    in1_data_ptr,
                                                                                                    out_data_ptr,
                                                                                                    in0_shape,
                                                                                                    in1_shape,
                                                                                                    broadcast_spec);
    return true;
}
