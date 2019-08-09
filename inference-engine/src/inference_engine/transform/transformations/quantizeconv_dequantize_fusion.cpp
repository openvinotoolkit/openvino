// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <ngraph_ops/quantize_conv_bias_fused.hpp>

#include "quantizeconv_dequantize_fusion.hpp"

#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/pattern/matcher.hpp"


void ngraph::pass::QuantizeConvDequantizeFusion::quantize_conv_fusion() {
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto filters = std::make_shared<pattern::op::Label>(element::f32, shape);
    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto scale = std::make_shared<pattern::op::Label>(element::f32, Shape{});

    auto conv = std::make_shared<ngraph::op::QuantizedConvolutionBias>(data_batch,
                                                                       filters,
                                                                       bias,
                                                                       Strides{1, 1},
                                                                       Strides{1, 1},
                                                                       CoordinateDiff{0, 0},
                                                                       CoordinateDiff{0, 0},
                                                                       Strides{1, 1},
                                                                       scale);

    auto dscale = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto doffset = std::make_shared<pattern::op::Label>(element::i8, Shape{});

    auto dequantize = std::make_shared<ngraph::op::Dequantize>(conv, dscale, doffset, element::f32, AxisSet{});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto dequantize = std::dynamic_pointer_cast<op::Dequantize>(m.get_match_root());
        auto conv = std::dynamic_pointer_cast<ngraph::op::QuantizedConvolutionBias>(dequantize->get_argument(0));

        auto w_scale = std::dynamic_pointer_cast<Node>(dequantize->get_argument(1));

        if (!conv || !dequantize) {
            return false;
        }

        auto conv_fused = std::make_shared<ngraph::op::QuantizedConvolutionBiasFused>(conv, w_scale);
        conv_fused->set_friendly_name(conv->get_friendly_name());
        ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<Node>(conv_fused));

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(dequantize, "CPUFusion.QuantizeConvDequantize");
    this->add_matcher(m, callback);
}
