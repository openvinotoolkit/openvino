// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/split_convolution_with_large_buffer_size.hpp"

#include <numeric>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "backend/gna_limitations.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SplitConvolution, "SplitConvolution", 0);

static std::vector<int64_t> GetConvSplitSizes(std::shared_ptr<ngraph::Node> conv) {
    uint32_t width = conv->get_input_shape(0).back();
    uint32_t in_channels = conv->get_input_shape(0).at(1);
    uint32_t usedWidth = 0;
    std::vector<int64_t> split_sizes;
    uint32_t width_max_size = GNALimitations::bufferMaxSize / in_channels;
    width_max_size = width_max_size - width_max_size % 64;
    while (usedWidth < width) {
        uint32_t width_part = std::min(width - usedWidth, width_max_size);
        split_sizes.push_back(width_part);
        usedWidth += width_part;
    }
    IE_ASSERT(usedWidth == width);
    return split_sizes;
}

SplitConvolution::SplitConvolution() {
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({ngraph::pattern::any_input(),
        ngraph::pattern::any_input()});
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto conv_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, add});
    auto out_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({conv_output,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});
    auto root = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv_output, out_fq});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto& pattern_map = m.get_pattern_value_map();
        auto conv_node = pattern_map.at(conv).get_node_shared_ptr();
        auto input_size = std::accumulate(std::begin(conv_node->get_input_shape(0)),
            std::end(conv_node->get_input_shape(0)), 1, std::multiplies<size_t>());
        if (input_size <= GNALimitations::bufferMaxSize) {
            return false;
        }

        auto split_sizes = GetConvSplitSizes(conv_node);
        IE_ASSERT(split_sizes.size() > 1);

        auto conv_output = conv_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
        std::shared_ptr<ngraph::Node> bias_node = nullptr;
        if (std::dynamic_pointer_cast<ngraph::opset7::Add>(conv_output)) {
            auto add_second_input = conv_output->input_value(1).get_node_shared_ptr();
            if (std::dynamic_pointer_cast<ngraph::opset7::Constant>(conv_output)) {
                bias_node = conv_output->input_value(1).get_node_shared_ptr();
                conv_output = conv_output->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            }
        }
        std::shared_ptr<ngraph::Node> fq_node = std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(conv_output);

        /* TODO check if it's NHWC convolution wrapped with transposes or all input dimensions except of width == 1,
           otherwise this split axis isn't supported */
        const int64_t width_axis = conv_node->get_input_shape(0).size() - 1;
        auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(conv_node->input_value(0),
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{width_axis}),
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes.size()}), split_sizes));
        split_node->set_friendly_name(conv_node->get_friendly_name() + "/split");
        ngraph::OutputVector convOutputs;
        for (int i = 0; i < split_sizes.size(); ++i) {
            std::shared_ptr<ngraph::Node> output = conv_node->clone_with_new_inputs({split_node->output(i), conv_node->input_value(1)});
            output->set_friendly_name(conv_node->get_friendly_name() + "_" + std::to_string(i));
            if (bias_node) {
                output = std::make_shared<ngraph::opset7::Add>(output, bias_node);
            }

            if (fq_node) {
                output = fq_node->clone_with_new_inputs({output, fq_node->input_value(1), fq_node->input_value(2),
                    fq_node->input_value(3), fq_node->input_value(4)});
            }
            convOutputs.push_back(output);
        }

        auto concat = std::make_shared<ngraph::opset7::Concat>(convOutputs, width_axis);
        concat->set_friendly_name(conv_node->get_friendly_name() + "/concat");
        ngraph::replace_node(conv_node, concat);

        if (fq_node) {
            ngraph::replace_output_update_name(fq_node->output(0), fq_node->input_value(0));
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "SplitConvolution");
    this->register_matcher(m, callback);
}