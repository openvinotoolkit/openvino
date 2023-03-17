// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/binary_convolution.hpp"

namespace bin_conv_v1 {
template <ngraph::element::Type_t t_in, ngraph::element::Type_t t_f>
inline void evaluate(const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op,
                     const ngraph::HostTensorVector& outputs,
                     const ngraph::HostTensorVector& inputs) {
    using T_IN = typename ngraph::element_type_traits<t_in>::value_type;
    using T_F = typename ngraph::element_type_traits<t_f>::value_type;

    const auto in_data_ptr = inputs[0]->get_data_ptr<T_IN>();
    const auto filter_data_ptr = inputs[1]->get_data_ptr<T_F>();
    auto out_data_ptr = outputs[0]->get_data_ptr<T_IN>();
    const auto in_shape = inputs[0]->get_shape();
    const auto filter_shape = inputs[1]->get_shape();
    const auto out_shape = outputs[0]->get_shape();

    ngraph::runtime::reference::binary_convolution<T_IN, T_F>(in_data_ptr,
                                                              filter_data_ptr,
                                                              out_data_ptr,
                                                              in_shape,
                                                              filter_shape,
                                                              out_shape,
                                                              op->get_strides(),
                                                              op->get_dilations(),
                                                              op->get_pads_begin(),
                                                              op->get_pads_end(),
                                                              op->get_pad_value());
}
}  // namespace bin_conv_v1

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v1::BinaryConvolution>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case ngraph::element::Type_t::u1:
        bin_conv_v1::evaluate<ET, ngraph::element::Type_t::u8>(op, outputs, inputs);
        break;
    default:
        throw std::runtime_error("BinaryConvolution supports only u1 element type for filters input");
        break;
    }
    return true;
}