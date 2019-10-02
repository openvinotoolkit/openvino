// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/op.hpp"
#include <memory>

namespace ngraph {
namespace op {

class QuantizedConvolutionBiasFused : public Op {
public:
    QuantizedConvolutionBiasFused(const std::shared_ptr<op::QuantizedConvolutionBias>& qconv,
            const std::shared_ptr<Node>& w_scale);

    QuantizedConvolutionBiasFused(const std::shared_ptr<Node>& data_batch,
            const std::shared_ptr<Node>& filters,
            const std::shared_ptr<Node>& bias,
            const Strides& window_movement_strides,
            const Strides& window_dilation_strides,
            const CoordinateDiff& padding_below,
            const CoordinateDiff& padding_above,
            const Strides& data_dilation_strides,
            const std::shared_ptr<Node> scale,
            const std::shared_ptr<Node> w_scale,
            const bool with_relu = false);

    const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
    const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
    const CoordinateDiff& get_padding_below() const { return m_padding_below; }
    const CoordinateDiff& get_padding_above() const { return m_padding_above; }
    const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
    std::shared_ptr<Node> get_w_scale() { return get_argument(4); }
    std::shared_ptr<Node> get_bias() { return get_argument(2); }
    std::shared_ptr<Node> get_filters() { return get_argument(1); }
    std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
    bool with_relu() const { return m_with_relu; }
    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

protected:
    Strides m_window_movement_strides;
    Strides m_window_dilation_strides;
    CoordinateDiff m_padding_below;
    CoordinateDiff m_padding_above;
    Strides m_data_dilation_strides;
    bool m_with_relu;
};

}  // namespace op
}  // namespace ngraph
