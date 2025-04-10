// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_group_conv.hpp"

#include <numeric>
#include <openvino/opsets/opset8.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"

ov::intel_cpu::ConvertGroupConvolution::ConvertGroupConvolution() {
    auto gconv = ov::pass::pattern::wrap_type<opset8::GroupConvolution>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        enum Inputs { Data, Weights };
        auto gconv = ov::as_type_ptr<opset8::GroupConvolution>(m.get_match_root());
        if (!gconv) {
            return false;
        }
        const unsigned int channel_axis = 1;
        const auto& input0 = gconv->input_value(0);
        const auto& output_shape = gconv->get_output_partial_shape(0);
        const auto& data_shape = input0.get_partial_shape();
        // Weights layout GOIYX
        int64_t groups = gconv->get_input_shape(Inputs::Weights)[0];

        if (data_shape[channel_axis].is_dynamic() || output_shape[channel_axis].is_dynamic()) {
            return false;
        }

        if (groups == data_shape[channel_axis].get_length() &&
            groups == output_shape[channel_axis].get_length()) {  // depthwise case
            return false;
        }

        ov::NodeVector replace_nodes;
        auto split_weights = std::make_shared<ov::opset1::Split>(
            gconv->input_value(Inputs::Weights),
            ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{}, {0}),
            groups);
        replace_nodes.push_back(split_weights);

        auto axis = ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{}, {channel_axis});
        auto split = std::make_shared<ov::opset1::Split>(gconv->input_value(Inputs::Data), axis, groups);
        replace_nodes.push_back(split);

        ov::NodeVector concat_inputs;
        for (int64_t g = 0; g < groups; g++) {
            auto out = split->output(g);
            auto filter = std::make_shared<ov::opset1::Squeeze>(
                split_weights->output(g),
                ov::opset8::Constant::create<int64_t>(ov::element::i64, ov::Shape{}, {0}));
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
        ov::copy_runtime_info(gconv, replace_nodes);
        ov::replace_node(gconv, concat);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(gconv, "ConvertGroupConvolution");
    register_matcher(m, callback);
}
