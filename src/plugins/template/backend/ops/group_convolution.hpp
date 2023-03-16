// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::GroupConvolution>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto filter_data = inputs[1]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& filter_shape = inputs[1]->get_shape();
    ngraph::runtime::reference::group_convolution<typename ngraph::element_type_traits<ET>::value_type>(
        in_data_ptr,
        filter_data,
        out_data_ptr,
        in_shape,
        filter_shape,
        out_shape,
        op->get_strides(),
        op->get_dilations(),
        op->get_pads_begin(),
        op->get_pads_end());
    return true;
}