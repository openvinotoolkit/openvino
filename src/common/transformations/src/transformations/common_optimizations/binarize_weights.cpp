// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/binarize_weights.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;

static float quantize(float f, float input_low, float input_high, float output_low, float output_high) {
    if (f <= input_low)
        return output_low;
    if (f > input_high)
        return output_high;
    return std::round((f - input_low) / (input_high - input_low)) * (output_high - output_low) + output_low;
}

static std::vector<float> quantize_weights(const Shape& weights_shape,
                                           std::vector<float>& weights,
                                           Shape input_low_high_shape,
                                           const std::vector<float>& input_low,
                                           const std::vector<float>& input_high,
                                           Shape output_low_high_shape,
                                           const std::vector<float>& output_low,
                                           const std::vector<float>& output_high) {
    OPENVINO_ASSERT(shape_size(input_low_high_shape) == 1 || shape_size(input_low_high_shape) == weights_shape[0]);
    OPENVINO_ASSERT(shape_size(output_low_high_shape) == 1 || shape_size(output_low_high_shape) == weights_shape[0]);
    size_t out_feat_off = 1;
    for (size_t i = 1; i < weights_shape.size(); i++)
        out_feat_off *= weights_shape[i];

    std::vector<float> out;
    out.reserve(shape_size(weights_shape));

    auto get_idx = [out_feat_off](size_t i, const Shape& shape) -> size_t {
        return (i / out_feat_off) % shape[0];
    };

    for (size_t i = 0; i < shape_size(weights_shape); i++) {
        size_t in_idx = get_idx(i, input_low_high_shape);
        size_t out_idx = get_idx(i, output_low_high_shape);
        out.push_back(
            quantize(weights[i], input_low[in_idx], input_high[in_idx], output_low[out_idx], output_high[out_idx]));
    }
    return out;
}

pass::BinarizeWeights::BinarizeWeights() {
    MATCHER_SCOPE(BinarizeWeights);
    auto activations_fq_pattern =
        pattern::wrap_type<ov::op::v0::FakeQuantize>({pattern::any_input(),
                                                      pattern::wrap_type<ov::op::v0::Constant>(),
                                                      pattern::wrap_type<ov::op::v0::Constant>(),
                                                      pattern::wrap_type<ov::op::v0::Constant>(),
                                                      pattern::wrap_type<ov::op::v0::Constant>()},
                                                     pattern::consumers_count(1));
    auto weights_fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>({pattern::wrap_type<ov::op::v0::Constant>(),
                                                                            pattern::wrap_type<ov::op::v0::Constant>(),
                                                                            pattern::wrap_type<ov::op::v0::Constant>(),
                                                                            pattern::wrap_type<ov::op::v0::Constant>(),
                                                                            pattern::wrap_type<ov::op::v0::Constant>()},
                                                                           pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<ov::op::v1::Convolution>({activations_fq_pattern, weights_fq_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(m.get_match_root());
        if (!conv)
            return false;
        auto activations_fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(conv->input_value(0).get_node_shared_ptr());
        if (!activations_fq || activations_fq->get_levels() != 2)
            return false;
        auto weights_fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        if (!weights_fq || weights_fq->get_levels() != 2)
            return false;

        auto weights_const = ov::as_type_ptr<ov::op::v0::Constant>(weights_fq->input_value(0).get_node_shared_ptr());
        if (!weights_const)
            return false;

        auto check_output_low_high = [](const std::vector<float>& output_low,
                                        const std::vector<float>& output_high) -> std::tuple<bool, bool> {
            bool output_low_is_zero = true;
            bool output_low_high_are_opposite = true;
            for (size_t i = 0; i < output_low.size(); i++) {
                output_low_is_zero = output_low_is_zero && output_low[i] == 0.0f;
                output_low_high_are_opposite = output_low_high_are_opposite && output_low[i] == -output_high[i];
            }
            return std::tuple<bool, bool>{output_low_is_zero, output_low_high_are_opposite};
        };

        auto activations_output_low_const =
            ov::as_type_ptr<ov::op::v0::Constant>(activations_fq->input_value(3).get_node_shared_ptr());
        auto activations_output_high_const =
            ov::as_type_ptr<ov::op::v0::Constant>(activations_fq->input_value(4).get_node_shared_ptr());
        if (!activations_output_low_const || !activations_output_high_const)
            return false;

        // Check output low and high on activations FQ first
        bool act_out_low_is_zero = false;
        bool act_out_low_high_are_opposite = false;
        auto activations_output_low = activations_output_low_const->cast_vector<float>();
        auto activations_output_high = activations_output_high_const->cast_vector<float>();
        std::tie(act_out_low_is_zero, act_out_low_high_are_opposite) =
            check_output_low_high(activations_output_low, activations_output_high);
        if (!(act_out_low_high_are_opposite || act_out_low_is_zero))
            return false;

        auto weights_input_low_const =
            ov::as_type_ptr<ov::op::v0::Constant>(weights_fq->input_value(1).get_node_shared_ptr());
        auto weights_input_high_const =
            ov::as_type_ptr<ov::op::v0::Constant>(weights_fq->input_value(2).get_node_shared_ptr());
        if (!weights_input_low_const || !weights_input_high_const)
            return false;
        auto weights_output_low_const =
            ov::as_type_ptr<ov::op::v0::Constant>(weights_fq->input_value(3).get_node_shared_ptr());
        auto weights_output_high_const =
            ov::as_type_ptr<ov::op::v0::Constant>(weights_fq->input_value(4).get_node_shared_ptr());
        if (!weights_output_low_const || !weights_output_high_const)
            return false;

        // Check output low and high on weights FQ
        bool weights_out_low_high_are_opposite = false;
        auto weights_output_low = weights_output_low_const->cast_vector<float>();
        auto weights_output_high = weights_output_high_const->cast_vector<float>();
        std::tie(std::ignore, weights_out_low_high_are_opposite) =
            check_output_low_high(weights_output_low, weights_output_high);
        if (!weights_out_low_high_are_opposite)
            return false;

        // Normalize output low and high to either (0, 1) or (-1, 1)
        auto normalize_output_low_high = [](std::vector<float>& output_low, std::vector<float>& output_high) {
            for (size_t i = 0; i < output_low.size(); i++) {
                output_low[i] /= output_high[i];
                output_high[i] = 1.0f;
            }
        };

        normalize_output_low_high(activations_output_low, activations_output_high);
        normalize_output_low_high(weights_output_low, weights_output_high);

        // Choose additional normalization factor that has to be put after Convolution
        const std::shared_ptr<Node>& activations_norm_factor = activations_output_high_const;
        const std::shared_ptr<Node>& weights_norm_factor = weights_output_high_const;

        // Create new FQ on activations with new output low/high
        auto output_low_normalized = ov::op::v0::Constant::create(element::f32,
                                                                  activations_output_low_const->get_shape(),
                                                                  activations_output_low);
        output_low_normalized->set_friendly_name(activations_output_low_const->get_friendly_name());
        auto output_high_normalized = ov::op::v0::Constant::create(element::f32,
                                                                   activations_output_high_const->get_shape(),
                                                                   activations_output_high);
        output_high_normalized->set_friendly_name(activations_output_high_const->get_friendly_name());
        auto new_activations_fq = activations_fq->clone_with_new_inputs({activations_fq->input_value(0),
                                                                         activations_fq->input_value(1),
                                                                         activations_fq->input_value(2),
                                                                         output_low_normalized,
                                                                         output_high_normalized});
        new_activations_fq->set_friendly_name(activations_fq->get_friendly_name());

        // Quantize weights - here we get rid of FQ on weights and create a constant with quantized weights
        auto weights = weights_const->cast_vector<float>();
        auto weights_input_low = weights_input_low_const->cast_vector<float>();
        auto weights_input_high = weights_input_high_const->cast_vector<float>();
        auto quantized_weights = quantize_weights(weights_const->get_shape(),
                                                  weights,
                                                  weights_input_low_const->get_shape(),
                                                  weights_input_low,
                                                  weights_input_high,
                                                  weights_output_low_const->get_shape(),
                                                  weights_output_low,
                                                  weights_output_high);
        auto quantized_weights_const =
            ov::op::v0::Constant::create(element::f32, weights_const->get_shape(), quantized_weights);
        quantized_weights_const->set_friendly_name(weights_const->get_friendly_name());
        auto new_conv = conv->clone_with_new_inputs({new_activations_fq, quantized_weights_const});

        std::vector<int64_t> norm_factor_shape = {-1};
        for (size_t i = 2; i < weights_const->get_shape().size(); i++)
            norm_factor_shape.push_back(1);
        auto norm_factor_shape_const =
            ov::op::v0::Constant::create(element::i64, Shape{norm_factor_shape.size()}, norm_factor_shape);

        auto activations_norm_factor_reshaped =
            std::make_shared<ov::op::v1::Reshape>(activations_norm_factor, norm_factor_shape_const, false);
        auto mul = std::make_shared<ov::op::v1::Multiply>(new_conv, activations_norm_factor_reshaped);
        auto weights_norm_factor_reshaped =
            std::make_shared<ov::op::v1::Reshape>(weights_norm_factor, norm_factor_shape_const, false);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(mul, weights_norm_factor_reshaped);

        copy_runtime_info(
            {activations_fq, weights_fq, conv},
            {new_activations_fq, new_conv, activations_norm_factor_reshaped, mul, weights_norm_factor_reshaped, mul2});
        mul2->set_friendly_name(conv->get_friendly_name());
        replace_node(conv, mul2);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}
