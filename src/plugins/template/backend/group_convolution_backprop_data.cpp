// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/group_convolution_backprop_data.hpp"
#include "openvino/op/group_conv.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::Convolution>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto filter_data = inputs[1]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& filter_shape = inputs[1]->get_shape();
    ngraph::runtime::reference::convolution<typename ov::element_type_traits<ET>::value_type>(in_data_ptr,
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

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::ConvolutionBackpropData>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto filter_data = inputs[1]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& filter_shape = inputs[1]->get_shape();
    ov::Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
    std::fill(in_dilation.begin(), in_dilation.end(), 1);
    ngraph::runtime::reference::convolution_backprop_in<typename ov::element_type_traits<ET>::value_type>(in_data_ptr,
                                                                                              filter_data,
                                                                                              out_data_ptr,
                                                                                              in_shape,
                                                                                              filter_shape,
                                                                                              out_shape,
                                                                                              in_dilation,
                                                                                              op->get_dilations(),
                                                                                              op->get_pads_begin(),
                                                                                              op->get_pads_end(),
                                                                                              op->get_strides(),
                                                                                              op->get_output_padding());
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::GroupConvolution>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto filter_data = inputs[1]->get_data_ptr<ET>();
    auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto& out_shape = outputs[0]->get_shape();
    const auto& in_shape = inputs[0]->get_shape();
    const auto& filter_shape = inputs[1]->get_shape();
    ngraph::runtime::reference::group_convolution<typename ov::element_type_traits<ET>::value_type>(in_data_ptr,
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

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData>& op,
              const ov::HostTensorVector& outputs,
              const ov::HostTensorVector& inputs) {
    const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
    const auto filter_data_ptr = inputs[1]->get_data_ptr<ET>();
    const auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
    const auto in_shape = inputs[0]->get_shape();
    const auto filter_shape = inputs[1]->get_shape();
    const auto out_shape = outputs[0]->get_shape();
    ngraph::runtime::reference::group_convolution_backprop_data<typename ov::element_type_traits<ET>::value_type>(
        in_data_ptr,
        filter_data_ptr,
        out_data_ptr,
        in_shape,
        filter_shape,
        out_shape,
        op->get_strides(),
        op->get_dilations(),
        op->get_pads_begin(),
        op->get_pads_end(),
        op->get_output_padding());
    return true;
}

