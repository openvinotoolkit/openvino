// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/split_convolution_with_large_buffer_size.hpp"

#include <numeric>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "backend/gna_limitations.hpp"

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(SplitConvolution, "SplitConvolution", 0);
NGRAPH_RTTI_DEFINITION(SplitConvolutionWithBias, "SplitConvolutionWithBias", 0);
NGRAPH_RTTI_DEFINITION(SplitConvolutionWithFq, "SplitConvolutionWithFq", 0);

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

static bool Convert(std::shared_ptr<ngraph::Node> conv,
                    std::shared_ptr<ngraph::Node> add,
                    std::shared_ptr<ngraph::Node> bias,
                    std::shared_ptr<ngraph::Node> fq) {
    auto input_size = std::accumulate(std::begin(conv->get_input_shape(0)),
        std::end(conv->get_input_shape(0)), 1, std::multiplies<size_t>());
    if (input_size <= GNALimitations::bufferMaxSize) {
        return false;
    }

    auto split_sizes = GetConvSplitSizes(conv);
    IE_ASSERT(split_sizes.size() > 1);

    /* TODO check if it's NHWC convolution wrapped with transposes or all input dimensions except of width == 1,
        otherwise this split axis isn't supported */
    const int64_t width_axis = conv->get_input_shape(0).size() - 1;
    auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(conv->input_value(0),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{width_axis}),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes.size()}), split_sizes));
    ngraph::copy_runtime_info(conv, split_node);
    split_node->set_friendly_name(conv->get_friendly_name() + "/split");
    ngraph::OutputVector convOutputs;
    std::shared_ptr<ngraph::Node> root_node = fq ? fq : (add ? add : conv);
    for (int i = 0; i < split_sizes.size(); ++i) {
        std::shared_ptr<ngraph::Node> output = conv->clone_with_new_inputs({split_node->output(i), conv->input_value(1)});
        ngraph::copy_runtime_info(split_node, output);
        output->set_friendly_name(conv->get_friendly_name() + "_" + std::to_string(i));
        if (bias) {
            output = std::make_shared<ngraph::opset7::Add>(output, bias);
            ngraph::copy_runtime_info(conv, output);
        }

        if (fq) {
            output = fq->clone_with_new_inputs({output, fq->input_value(1), fq->input_value(2),
                fq->input_value(3), fq->input_value(4)});
            ngraph::copy_runtime_info(fq, output);
        }
        convOutputs.push_back(output);
    }

    auto concat = std::make_shared<ngraph::opset7::Concat>(convOutputs, width_axis);
    ngraph::copy_runtime_info(conv, concat);
    concat->set_friendly_name(conv->get_friendly_name());
    ngraph::replace_node(root_node, concat);
    return true;
}

SplitConvolution::SplitConvolution() {
    MATCHER_SCOPE(SplitConvolution);
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({ngraph::pattern::any_input(),
        ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(conv).get_node_shared_ptr(), nullptr, nullptr, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

SplitConvolutionWithBias::SplitConvolutionWithBias() {
    MATCHER_SCOPE(SplitConvolutionWithBias);
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({ngraph::pattern::any_input(),
        ngraph::pattern::any_input()});
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, bias});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(conv).get_node_shared_ptr(), pattern_map.at(add).get_node_shared_ptr(),
            pattern_map.at(bias).get_node_shared_ptr(), nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, matcher_name);
    this->register_matcher(m, callback);
}

SplitConvolutionWithFq::SplitConvolutionWithFq() {
    MATCHER_SCOPE(SplitConvolutionWithFq);
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({ngraph::pattern::any_input(),
        ngraph::pattern::any_input()});
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto add = ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, bias});
    auto conv_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{conv, add});
    auto out_fq = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({conv_output,
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto add_it = pattern_map.find(add);
        auto add_node = (add_it == std::end(pattern_map) ? nullptr : add_it->second.get_node_shared_ptr());
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        return Convert(pattern_map.at(conv).get_node_shared_ptr(), add_node, bias_node, pattern_map.at(out_fq).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(out_fq, matcher_name);
    this->register_matcher(m, callback);
}