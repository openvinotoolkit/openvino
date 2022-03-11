// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quant_pass.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/utils/utils.hpp>

#include "ngraph_ops/channel_fake_quant_internal.hpp"
#include "ngraph_ops/fake_dequant_internal.hpp"
#include "ngraph_ops/fake_quant_dequant_internal.hpp"
#include "ngraph_ops/fake_quant_internal.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddle_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

static std::shared_ptr<FakeQuantize> get_quant_common(const int range,
                                                      const float scale,
                                                      const ov::Output<ov::Node>& input) {
    // Because Convolution/GroupConvolution/Multiply use u8 and it's range [0, 255]
    //   which is same as [-128, 127] + 128. Paddle always uses i8 [-127, 127]. Here
    //   1, limit the quantized value to [-scale, scale]. If not do so the case will fail: scale=-123, data=-125
    //   2, compute the value at point -128
    // If ConvolutionBackpropData/MatMul use u8 will be same as previous
    //   if use i8 because levels set to 256, openvino will use [-128, 127] automaticly
    //   (openvino supported both [-127, 127] and [-128, 127] based on levels)
    const auto clip = std::make_shared<Clamp>(input, -scale, scale);
    const auto scale_low = -scale * 128 / range;
    const auto scale_high = scale * 127 / range;
    const auto input_low = Constant::create(element::f32, {1}, {scale_low});
    const auto input_high = Constant::create(element::f32, {1}, {scale_high});
    const auto output_low = Constant::create(element::f32, {1}, {scale_low});
    const auto output_high = Constant::create(element::f32, {1}, {scale_high});
    return std::make_shared<FakeQuantize>(clip, input_low, input_high, output_low, output_high, 256);
}

static std::shared_ptr<FakeQuantize> get_fakequant_for_data(
    const std::shared_ptr<const ngraph::op::internal::FakeQuantInternal> fake_internal) {
    const auto range = (1 << (fake_internal->m_bit_length - 1)) - 1;
    const auto input = fake_internal->input_value(1).get_node_shared_ptr();
    const auto& const_input = std::dynamic_pointer_cast<Constant>(input);
    // mark the scale is input scale and the dequant can ignore it
    const_input->get_rt_info()["in_scale"] = {};
    std::vector<float> scales = const_input->cast_vector<float>();
    return get_quant_common(range, scales[0], fake_internal->input_value(0));
}

ov::frontend::pass::FuseQuant::FuseQuant() {
    auto fake_quant = ngraph::pattern::wrap_type<ngraph::op::internal::FakeQuantInternal>(
        {ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    matcher_pass_callback callback = [fake_quant](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto& op = pattern_to_output.at(fake_quant).get_node_shared_ptr();
        const auto& fake_quant_op = std::dynamic_pointer_cast<const ngraph::op::internal::FakeQuantInternal>(op);
        // deal activation
        auto fq_data = get_fakequant_for_data(fake_quant_op);
        replace_node(op, fq_data);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_quant, "fuse quant");
    register_matcher(m, callback);
}

static std::shared_ptr<Node> get_dequant_common(const std::vector<float>& weights,
                                                const Shape& weights_shape,
                                                const element::Type& data_type,
                                                const std::vector<float>& scales,
                                                const Shape& scales_shape) {
    // dequantize: const(int8)->convert->multply
    const auto const_input = Constant::create(element::i8, weights_shape, weights);
    const auto weight_float = std::make_shared<Convert>(const_input, data_type);

    auto mul = std::make_shared<Multiply>(weight_float, Constant::create(element::f32, scales_shape, scales));
    return mul;
}

static std::shared_ptr<Node> get_fakequant_for_weight(
    const std::shared_ptr<const ngraph::op::internal::FakeDequantInternal> fake_internal) {
    const auto range = (1 << (fake_internal->m_bit_length - 1)) - 1;
    const auto org_conv = fake_internal->input_value(0).get_node_shared_ptr();
    const auto max_range = fake_internal->m_max_range;
    float scale = static_cast<float>(range * range) / max_range;

    const float norm_scale = scale / range;
    const auto& const_weight_org = std::dynamic_pointer_cast<Constant>(org_conv->input_value(1).get_node_shared_ptr());
    auto fake_op = get_dequant_common(const_weight_org->cast_vector<float>(),
                                      const_weight_org->get_shape(),
                                      org_conv->get_input_element_type(0),
                                      {norm_scale},
                                      {1});
    auto new_conv = org_conv->clone_with_new_inputs({org_conv->input_value(0), fake_op});
    return new_conv;
}

ov::frontend::pass::FuseDequant::FuseDequant() {
    auto fake_dequant = ngraph::pattern::wrap_type<ngraph::op::internal::FakeDequantInternal>();

    matcher_pass_callback callback = [fake_dequant](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto& op = pattern_to_output.at(fake_dequant).get_node_shared_ptr();
        const auto& fake_dequant_op = std::dynamic_pointer_cast<const ngraph::op::internal::FakeDequantInternal>(op);
        // deal weight
        auto new_conv = get_fakequant_for_weight(fake_dequant_op);
        replace_node(op, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_dequant, "fuse dequant");
    register_matcher(m, callback);
}

static std::shared_ptr<Node> get_fakequant_for_weight(
    const std::shared_ptr<const ngraph::op::internal::ChannelFakeQuantInternal> fake_internal) {
    const auto range = (1 << (fake_internal->m_bit_length - 1)) - 1;
    const auto org_conv = fake_internal->input_value(0).get_node_shared_ptr();
    bool is_groupconv = std::dynamic_pointer_cast<GroupConvolution>(org_conv) != nullptr;

    // get scales
    std::vector<float> scales;
    for (size_t i = 1; i < fake_internal->get_input_size(); i++) {
        const auto input = fake_internal->input_value(i).get_node_shared_ptr();
        const auto& const_input = std::dynamic_pointer_cast<Constant>(input);
        auto rt_info = const_input->get_rt_info();
        if (rt_info.count("in_scale"))
            continue;
        scales = const_input->cast_vector<float>();
        break;
    }
    std::vector<float> norm_scales = scales;
    std::transform(norm_scales.begin(), norm_scales.end(), norm_scales.begin(), [&](const float v) {
        return v / range;
    });

    // compute weight, scale shape
    auto scales_shape_len = org_conv->input_value(1).get_partial_shape().rank().get_length();
    if (is_groupconv) {
        scales_shape_len -= 1;
    }
    OPENVINO_ASSERT(scales_shape_len > 1, "scale shape should great than 1.");
    std::vector<size_t> scales_shape(scales_shape_len, 1);
    std::vector<size_t> const_shape(scales_shape_len, 1);
    const auto& const_weight_org = std::dynamic_pointer_cast<Constant>(org_conv->input_value(1).get_node_shared_ptr());
    const auto const_shape_org = const_weight_org->get_shape();
    OPENVINO_ASSERT(const_shape_org.size() > 2, "weight shape should great than 2.");
    if (is_groupconv) {
        scales_shape[0] = scales.size();
        memcpy(&const_shape[1], &const_shape_org[2], (scales_shape_len - 1) * sizeof(const_shape[1]));
        const_shape[0] = const_shape_org[0] * const_shape_org[1];
    } else {
        scales_shape[fake_internal->m_quant_axis] = scales.size();
        const_shape = const_shape_org;
    }

    // get dequant node
    auto mul = get_dequant_common(const_weight_org->cast_vector<float>(),
                                  const_shape,
                                  org_conv->get_input_element_type(0),
                                  norm_scales,
                                  scales_shape);
    if (std::dynamic_pointer_cast<GroupConvolution>(org_conv)) {
        // group convolution need a reshape in the weight path
        const auto out_shape = org_conv->input_value(1).get_shape();
        auto shape_const = Constant::create(element::i64, {out_shape.size()}, out_shape);
        auto reshape = std::make_shared<Reshape>(mul, shape_const, false);
        return org_conv->clone_with_new_inputs({org_conv->input_value(0), reshape});
    } else {
        return org_conv->clone_with_new_inputs({org_conv->input_value(0), mul});
    }
}

ov::frontend::pass::FuseChannelDequant::FuseChannelDequant() {
    auto fake_dequant = ngraph::pattern::wrap_type<ngraph::op::internal::ChannelFakeQuantInternal>();

    matcher_pass_callback callback = [fake_dequant](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto& op = pattern_to_output.at(fake_dequant).get_node_shared_ptr();
        const auto& fake_dequant_op =
            std::dynamic_pointer_cast<const ngraph::op::internal::ChannelFakeQuantInternal>(op);
        // deal weight
        auto new_conv = get_fakequant_for_weight(fake_dequant_op);
        replace_node(op, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_dequant, "fuse channel dequant");
    register_matcher(m, callback);
}

static std::shared_ptr<ngraph::Node> get_fakequant_for_data(
    const std::shared_ptr<const ngraph::op::internal::FakeQuantDequantInternal> fake_internal) {
    const auto range = (1 << (fake_internal->m_bit_length - 1)) - 1;
    std::shared_ptr<ngraph::Node> input_low, input_high, output_low, output_high;
    if (fake_internal->m_op_type == "fake_quantize_dequantize_abs_max") {
        // fake quant for weight
        const auto input = fake_internal->input_value(0).get_node_shared_ptr();
        const auto& const_input = std::dynamic_pointer_cast<Constant>(input);
        std::vector<float> weights = const_input->cast_vector<float>();
        auto abs_compare_func = [](float a, float b) {
            return (std::abs(a) < std::abs(b));
        };
        float scale = std::abs(*std::max_element(weights.begin(), weights.end(), abs_compare_func));
        std::transform(weights.begin(), weights.end(), weights.begin(), [&](const float v) {
            return std::round(v / scale * range);
        });

        return get_dequant_common(weights, const_input->get_shape(), input->get_element_type(), {scale / range}, {1});
    } else if (fake_internal->m_op_type == "fake_quantize_dequantize_moving_average_abs_max") {
        // fake quant for data
        const auto input = fake_internal->input_value(1).get_node_shared_ptr();
        const auto& const_input = std::dynamic_pointer_cast<Constant>(input);
        std::vector<float> scales = const_input->cast_vector<float>();
        return get_quant_common(range, scales[0], fake_internal->input_value(0));
    } else {
        // fake quant for weight
        // fake_channel_wise_quantize_dequantize_abs_max
        const auto input = fake_internal->input_value(1).get_node_shared_ptr();
        const auto& const_input = std::dynamic_pointer_cast<Constant>(input);
        auto scales = const_input->cast_vector<float>();
        std::vector<float> neg_scales = scales;
        std::transform(neg_scales.begin(), neg_scales.end(), neg_scales.begin(), [](const float v) {
            return -v;
        });
        std::vector<size_t> scales_shape(fake_internal->input_value(0).get_partial_shape().rank().get_length(), 1);
        scales_shape[fake_internal->m_quant_axis] = scales.size();
        input_low = Constant::create(element::f32, scales_shape, neg_scales);
        input_high = Constant::create(element::f32, scales_shape, scales);
        output_low = Constant::create(element::f32, scales_shape, neg_scales);
        output_high = Constant::create(element::f32, scales_shape, scales);
    }

    return std::make_shared<FakeQuantize>(fake_internal->input_value(0),
                                          input_low,
                                          input_high,
                                          output_low,
                                          output_high,
                                          range * 2 + 1);
}

ov::frontend::pass::FuseQuantDequant::FuseQuantDequant() {
    auto fake_quant = ngraph::pattern::wrap_type<ngraph::op::internal::FakeQuantDequantInternal>();

    matcher_pass_callback callback = [fake_quant](pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto& op = pattern_to_output.at(fake_quant).get_node_shared_ptr();
        const auto& fake_op = std::dynamic_pointer_cast<const ngraph::op::internal::FakeQuantDequantInternal>(op);
        // deal activation
        auto new_quant = get_fakequant_for_data(fake_op);
        replace_node(op, new_quant);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fake_quant, "fuse quant dequant");
    register_matcher(m, callback);
}
