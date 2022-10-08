// Copyright (C) 2018-2022 Intel Corporation
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
#include "layers/gna_split_layer.hpp"
#include "layers/gna_convolution_layer.hpp"

using namespace ov::intel_gna::pass;

// Don't split when convolution is 2D and is not mappable to 1D
static bool shouldSplitCnn(const ngraph::Output<ngraph::Node>& node) {
    auto convolution = dynamic_cast<ngraph::opset7::Convolution*>(node.get_node());
    IE_ASSERT(convolution != nullptr);
    auto& input = convolution->get_input_shape(0);
    auto& filters = convolution->get_input_shape(1);
    uint32_t width = input.back();
    uint32_t in_channels = input.at(1);
    if (input.size() >= 4 && filters.size() >= 4) {
        uint32_t height = input.at(2);
        auto kH = filters.at(2);
        auto kW = filters.at(3);
        auto sH = convolution->get_strides().at(0);
        auto sW = convolution->get_strides().at(1);
        if (GNAPluginNS::GNAConvolutionLayer::isConv2D(height, width, in_channels, kH, kW) &&
            !GNAPluginNS::GNAConvolutionLayer::isMappableFrom2DTo1D(height, width, in_channels, kH, kW, sH, sW)) {
            return false;
        }
    }
    return true;
}

static std::shared_ptr<ngraph::Node> getConvForMatcher() {
    return ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({ ngraph::pattern::any_input(),
    ngraph::pattern::any_input() }, [](const ngraph::Output<ngraph::Node>& convolution) {
            return shouldSplitCnn(convolution);
        });
}

static bool Convert(std::shared_ptr<ngraph::Node> conv,
                    std::shared_ptr<ngraph::Node> add,
                    std::shared_ptr<ngraph::Node> bias,
                    std::shared_ptr<ngraph::Node> fq) {
    auto input_size = std::accumulate(std::begin(conv->get_input_shape(0)),
        std::end(conv->get_input_shape(0)), size_t(1), std::multiplies<size_t>());
    if (input_size <= GNAPluginNS::GNALimitations::bufferMaxSize) {
        return false;
    }
    auto& input = conv->get_input_shape(0);
    uint32_t width = input.back();
    uint32_t in_channels = input.at(1);
    auto split_sizes = GNAPluginNS::GetAlignedSplitSizes(width, GNAPluginNS::GNALimitations::bufferMaxSize / in_channels);
    IE_ASSERT(split_sizes.size() > 1);
    std::vector<int64_t> split_sizes_casted(split_sizes.size());
    std::transform(std::begin(split_sizes), std::end(split_sizes), std::begin(split_sizes_casted), [](uint32_t size) {
        return static_cast<int64_t>(size);
    });

    /* TODO check if it's NHWC convolution wrapped with transposes or all input dimensions except of width == 1,
        otherwise this split axis isn't supported */
    const int64_t width_axis = input.size() - 1;
    auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(conv->input_value(0),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{width_axis}),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes_casted.size()}), split_sizes_casted));
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
    auto conv = getConvForMatcher();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(conv).get_node_shared_ptr(), nullptr, nullptr, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

SplitConvolutionWithBias::SplitConvolutionWithBias() {
    MATCHER_SCOPE(SplitConvolutionWithBias);
    auto conv = getConvForMatcher();
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
    auto conv = getConvForMatcher();
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
