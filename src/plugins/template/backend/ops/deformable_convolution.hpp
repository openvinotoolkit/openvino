// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v8::DeformableConvolution>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto offset_data_ptr = inputs[1]->get_data_ptr<ET>();
    const auto filter_data_ptr = inputs[2]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& offset_shape = inputs[1]->get_shape();
    const auto& filter_shape = inputs[2]->get_shape();
    if (inputs.size() == 3) {
        ngraph::runtime::reference::deformable_convolution<typename ngraph::element_type_traits<ET>::value_type>(
            in_data_ptr,
            offset_data_ptr,
            filter_data_ptr,
            out_data_ptr,
            in_shape,
            offset_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_group(),
            op->get_deformable_group(),
            op->get_bilinear_interpolation_pad());
    } else {
        const auto mask_data_ptr = inputs[3]->get_data_ptr<ET>();
        const auto& mask_shape = inputs[3]->get_shape();
        ngraph::runtime::reference::deformable_convolution<typename ngraph::element_type_traits<ET>::value_type>(
            in_data_ptr,
            offset_data_ptr,
            filter_data_ptr,
            mask_data_ptr,
            out_data_ptr,
            in_shape,
            offset_shape,
            filter_shape,
            mask_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_group(),
            op->get_deformable_group(),
            op->get_bilinear_interpolation_pad());
    }
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::DeformableConvolution>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto offset_data_ptr = inputs[1]->get_data_ptr<ET>();
    const auto filter_data_ptr = inputs[2]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& offset_shape = inputs[1]->get_shape();
    const auto& filter_shape = inputs[2]->get_shape();
    ngraph::runtime::reference::deformable_convolution<typename ngraph::element_type_traits<ET>::value_type>(
        in_data_ptr,
        offset_data_ptr,
        filter_data_ptr,
        out_data_ptr,
        in_shape,
        offset_shape,
        filter_shape,
        out_shape,
        op->get_strides(),
        op->get_dilations(),
        op->get_pads_begin(),
        op->get_pads_end(),
        op->get_group(),
        op->get_deformable_group());
    return true;
}