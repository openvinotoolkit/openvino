// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/convert_dwsc_to_scaleshifts.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/rt_info.hpp>
#include <ie_common.h>
#include "utils/transformation_helper.hpp"


using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(ConvertDWSCToScaleShifts, "ConvertDWSCToScaleShifts", 0);
NGRAPH_RTTI_DEFINITION(ConvertDWSCBiasToScaleShifts, "ConvertDWSCBiasToScaleShifts", 0);
NGRAPH_RTTI_DEFINITION(ConvertDWSCWithFqToScaleShifts, "ConvertDWSCWithFqToScaleShifts", 0);

static bool VerifyDWSC(std::shared_ptr<ngraph::opset7::GroupConvolution> dwsc) {
    // Verify it's a 1D convolution
    // Verify that filter group count == input channel count
    // Verify that per group filter output channel count == 1
    if (dwsc->get_input_shape(1)[3] != 1 || dwsc->get_input_shape(0)[2] != 1 || dwsc->get_output_shape(0)[2] != 1 ||
        dwsc->get_input_shape(1)[0] != dwsc->get_input_shape(0)[1] ||
        dwsc->get_input_shape(1)[1] != 1)
        return false;

    return true;
}

static std::shared_ptr<ngraph::Node> DecomposeDWSC(std::shared_ptr<ngraph::opset7::GroupConvolution> dwsc,
    std::shared_ptr<ngraph::opset7::Constant> bias_const, std::shared_ptr<ngraph::opset7::FakeQuantize> fq_bias,
    std::shared_ptr<ngraph::opset7::Reshape> flat_input_plane, std::shared_ptr<ngraph::Node> flat_filters_plane) {
    std::shared_ptr<ngraph::opset7::Constant> const_zero_padding;
    std::shared_ptr<ngraph::Node> reshaped_bias;
    ngraph::OutputVector output_chunks;
    auto input_channel_count = dwsc->get_input_shape(0)[1];
    auto input_width = dwsc->get_input_shape(0)[3];
    auto output_width = dwsc->get_output_shape(0)[3];
    auto filter_width = dwsc->get_input_shape(1)[4];
    auto pads_begin = dwsc->get_pads_begin()[1];
    auto stride_width = dwsc->get_strides()[1];
    auto dilation_width = dwsc->get_dilations()[1];

    // Constant with zero padding
    if (pads_begin) {
        const_zero_padding = std::make_shared<ngraph::opset7::Constant>(dwsc->get_element_type(), ngraph::Shape{1, input_channel_count}, 0);
        copy_runtime_info(dwsc, const_zero_padding);
    }

    // Reshape bias const
    if (bias_const) {
        auto bias_size = shape_size(bias_const->get_shape());
        reshaped_bias = ngraph::op::util::make_try_fold<ngraph::opset7::Reshape>(bias_const,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, bias_size}), false);
    }

    // Move filter over input performing multiplication and addition (scaleshift), take padding, stride, dilation and bias into account
    for (int32_t input_position = -pads_begin, o = 0; o < output_width; input_position += stride_width, o++) {
        std::shared_ptr<ngraph::Node> previous_layer_output, last_layer_output;
        int32_t filter_end = input_position + filter_width * dilation_width;
        bool first = true;

        filter_end = filter_end < input_width ? filter_end : input_width;

        for (int32_t filter_pos = input_position, filter_idx = 0; filter_pos < filter_end; filter_pos += dilation_width, filter_idx++) {
            if (filter_pos >= 0) {
                auto conv_input_slice = FlatCrop(flat_input_plane, filter_pos * input_channel_count, input_channel_count);
                auto conv_filter_slice = FlatCrop(flat_filters_plane, filter_idx * input_channel_count, input_channel_count);

                if (first) {
                    first = false;
                    previous_layer_output = std::make_shared<ngraph::opset7::Multiply>(conv_input_slice, conv_filter_slice);
                    copy_runtime_info(dwsc, previous_layer_output);
                    if (bias_const) {
                        previous_layer_output = std::make_shared<ngraph::opset7::Add>(previous_layer_output, reshaped_bias);
                        copy_runtime_info(dwsc, previous_layer_output);
                        previous_layer_output = InsertFQLayer(fq_bias, previous_layer_output);
                    }
                    last_layer_output = previous_layer_output;
                } else {
                    last_layer_output = std::make_shared<ngraph::opset7::Multiply>(conv_input_slice, conv_filter_slice);
                    copy_runtime_info(dwsc, last_layer_output);
                    last_layer_output = std::make_shared<ngraph::opset7::Add>(last_layer_output, previous_layer_output);
                    copy_runtime_info(dwsc, last_layer_output);
                    previous_layer_output = last_layer_output;
                }
            }
        }

        if (!last_layer_output) {
            IE_ASSERT(const_zero_padding);
            last_layer_output = const_zero_padding;
        }

        output_chunks.push_back(last_layer_output);
    }

    // Concat and transpose is only needed when output width > 1
    if (output_chunks.size() > 1) {
        auto concat_output_plane = std::make_shared<ngraph::opset7::Concat>(output_chunks, 0);
        auto transposed_concat_output_plane = std::make_shared<ngraph::opset7::Transpose>(concat_output_plane,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, 0}));
        copy_runtime_info(dwsc, {concat_output_plane, transposed_concat_output_plane});
        return transposed_concat_output_plane;
    }

    return output_chunks[0].get_node_shared_ptr();
}

static bool Convert(std::shared_ptr<ngraph::Node> dwsc_node,
    std::shared_ptr<ngraph::Node> reshape_filters_const_node,
    std::shared_ptr<ngraph::Node> bias_node,
    std::shared_ptr<ngraph::Node> bias_const_node,
    std::shared_ptr<ngraph::Node> fq_bias_node) {
    auto dwsc = std::dynamic_pointer_cast<ngraph::opset7::GroupConvolution>(dwsc_node);
    auto reshape_filters_const = std::dynamic_pointer_cast<ngraph::opset7::Reshape>(reshape_filters_const_node);
    auto bias = std::dynamic_pointer_cast<ngraph::opset7::Add>(bias_node);
    auto bias_const = std::dynamic_pointer_cast<ngraph::opset7::Constant>(bias_const_node);
    auto fq_bias = std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_bias_node);

    if (!VerifyDWSC(dwsc))
        return false;

    auto input_channel_count = dwsc->get_input_shape(0)[1];
    auto input_width = dwsc->get_input_shape(0)[3];
    auto output_channel_count = dwsc->get_output_shape(0)[1];
    auto output_width = dwsc->get_output_shape(0)[3];
    auto original_last_node = (fq_bias ? fq_bias_node : (bias_const ? bias_node : dwsc_node));

    // Prepare flat input data
    auto reshaped_input_plane = std::make_shared<ngraph::opset7::Reshape>(dwsc->input_value(0),
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            ngraph::Shape{input_channel_count, input_width}), false);

    auto transposed_input_plane = std::make_shared<ngraph::opset7::Transpose>(reshaped_input_plane,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, 0}));

    auto flat_input_plane = std::make_shared<ngraph::opset7::Reshape>(transposed_input_plane,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2},
            ngraph::Shape{1, shape_size(dwsc->input_value(0).get_shape())}), false);

    // Prepare flat filter data
    auto filters_const = std::dynamic_pointer_cast<ngraph::Node>(dwsc->get_input_node_shared_ptr(1));
    auto filters_size = shape_size(filters_const->get_shape());

    auto transposed_filters_const = ngraph::op::util::make_try_fold<ngraph::opset7::Transpose>(filters_const,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{5}, ngraph::Shape{4, 1, 2, 3, 0}));

    auto flat_filters_plane = ngraph::op::util::make_try_fold<ngraph::opset7::Reshape>(transposed_filters_const,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, filters_size}), false);

    copy_runtime_info(dwsc, {reshaped_input_plane, transposed_input_plane, flat_input_plane, transposed_filters_const, flat_filters_plane});

    // Convert DWSC to a set of diagonal layers
    auto output_plane = DecomposeDWSC(dwsc, bias_const, fq_bias, flat_input_plane, flat_filters_plane);

    // Restore the original output shape
    auto result = std::make_shared<ngraph::opset7::Reshape>(output_plane,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4},
            ngraph::Shape{1, output_channel_count, 1, output_width}), false);
    copy_runtime_info(dwsc, result);

    // We need to put here the original Group Convolution layer name, so the new layer output can be used as a network result
    std::string result_name = original_last_node->get_friendly_name();
    replace_node(original_last_node, result);
    result->set_friendly_name(result_name);

    return true;
}

ConvertDWSCToScaleShifts::ConvertDWSCToScaleShifts() {
    MATCHER_SCOPE(ConvertDWSCToScaleShifts);

    auto dwsc = ngraph::pattern::wrap_type<ngraph::opset7::GroupConvolution>(
        {ngraph::pattern::any_input(), ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(5))},
        ngraph::pattern::rank_equals(4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        return Convert(pattern_map.at(dwsc).get_node_shared_ptr(), nullptr, nullptr, nullptr, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(dwsc, matcher_name);
    this->register_matcher(m, callback);
}

ConvertDWSCBiasToScaleShifts::ConvertDWSCBiasToScaleShifts() {
    MATCHER_SCOPE(ConvertDWSCBiasToScaleShifts);

    auto dwsc = ngraph::pattern::wrap_type<ngraph::opset7::GroupConvolution>(
        {ngraph::pattern::any_input(), ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(5))},
        consumers_and_rank(1, 4));
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({dwsc, const_input});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        std::shared_ptr<ngraph::Node> bias_const = nullptr;

        if (bias_node && (bias_const = VerifyBiasGetConst(pattern_map.at(dwsc).get_node_shared_ptr(), bias_node)) == nullptr)
            return false;

        return Convert(pattern_map.at(dwsc).get_node_shared_ptr(), nullptr, bias_node, bias_const, nullptr);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bias, matcher_name);
    this->register_matcher(m, callback);
}

ConvertDWSCWithFqToScaleShifts::ConvertDWSCWithFqToScaleShifts() {
    MATCHER_SCOPE(ConvertDWSCWithFqToScaleShifts);

    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto filters_const_fq = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4));
    auto fq_filters_const = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({filters_const_fq, const_input, const_input, const_input, const_input},
        consumers_and_rank(1, 4));
    auto reshape_filters_const = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>({fq_filters_const, const_input});
    auto dwsc = ngraph::pattern::wrap_type<ngraph::opset7::GroupConvolution>(
        {ngraph::pattern::any_input(), reshape_filters_const}, consumers_and_rank(1, 4));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({dwsc, const_input});
    auto fq_bias = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>({bias, const_input, const_input, const_input, const_input},
        consumers_and_rank(1, 4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        std::shared_ptr<ngraph::Node> bias_const = nullptr;

        if (bias_node && (bias_const = VerifyBiasGetConst(pattern_map.at(dwsc).get_node_shared_ptr(), bias_node)) == nullptr)
            return false;

        return Convert(pattern_map.at(dwsc).get_node_shared_ptr(), pattern_map.at(reshape_filters_const).get_node_shared_ptr(),
            bias_node, bias_const, pattern_map.at(fq_bias).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_bias, matcher_name);
    this->register_matcher(m, callback);
}
