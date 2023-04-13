// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_group_conv.hpp"

#include <numeric>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::ConvertGroupConvolution::ConvertGroupConvolution() {
    auto gconv = ngraph::pattern::wrap_type<opset8::GroupConvolution>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        enum Inputs {Data, Weights};
        auto gconv = std::dynamic_pointer_cast<opset8::GroupConvolution>(m.get_match_root());
        if (!gconv) {
            return false;
        }

        auto data_shape = gconv->get_input_shape(Inputs::Data);
        // Weights layout GOIYX
        size_t groups = gconv->get_input_shape(Inputs::Weights)[0];
        if (groups == data_shape.at(1) && groups == gconv->get_output_shape(0)[1]) { // depthwise case
            return false;
        }

        ngraph::NodeVector replace_nodes;
        auto split_weights = std::make_shared<ov::opset1::Split>(gconv->input_value(Inputs::Weights),
                                                                 ov::opset8::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {0}),
                                                                 groups);
        replace_nodes.push_back(split_weights);

        auto axis  = ov::opset8::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {1});
        auto split = std::make_shared<ov::opset1::Split>(gconv->input_value(Inputs::Data), axis, groups);
        replace_nodes.push_back(split);

        ngraph::NodeVector concat_inputs;
        for (size_t g = 0; g < groups; g++) {
            auto out = split->output(g);
            auto filter = std::make_shared<ov::opset1::Squeeze>(split_weights->output(g),
                                                                ov::opset8::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {0}));
            auto conv = std::make_shared<ov::opset8::Convolution>(out,
                                                                  filter,
                                                                  gconv->get_strides(),
                                                                  gconv->get_pads_begin(),
                                                                  gconv->get_pads_end(),
                                                                  gconv->get_dilations(),
                                                                  gconv->get_auto_pad());
            concat_inputs.push_back(conv);
            replace_nodes.push_back(conv);
        }
        auto concat = std::make_shared<ov::opset8::Concat>(concat_inputs, 1);
        replace_nodes.push_back(concat);

        concat->set_friendly_name(gconv->get_friendly_name());
        ngraph::copy_runtime_info(gconv, replace_nodes);
        ngraph::replace_node(gconv, concat);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(gconv, "ConvertGroupConvolution");
    register_matcher(m, callback);
}
