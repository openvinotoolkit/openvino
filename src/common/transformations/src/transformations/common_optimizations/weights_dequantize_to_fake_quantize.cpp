// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::WeightsDequantizeToFakeQuantize::WeightsDequantizeToFakeQuantize() {
    MATCHER_SCOPE(WeightsDequantizeToFakeQuantize);

    const auto weights = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(element::i8));
    const auto convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights});
    const auto sub_c_integer = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches(element::i8));
    const auto convert_sub_c_integer = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({sub_c_integer});
    const auto sub_integer = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({convert, convert_sub_c_integer});
    const auto sub_c = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto sub = ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({convert, sub_c});
    const auto sub_or_sub_integer = std::make_shared<pattern::op::Or>(OutputVector{sub_integer, sub});
    const auto sub_or_convert = std::make_shared<pattern::op::Or>(OutputVector{convert, sub_or_sub_integer});

    const auto mul_c = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({sub_or_convert, mul_c});

    ov::matcher_pass_callback callback;
    callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        const auto& weights_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights));
        const auto& convert_node = pattern_map.at(convert);
        const auto& multiply_node = pattern_map.at(mul);
        const auto& scale_node = pattern_map.at(mul_c);
        if (!weights_node || !convert_node || !multiply_node || !scale_node) {
            return false;
        }

        const auto* data = weights_node->get_data_ptr<int8_t>();
        const int8_t weights_minimum = *std::min_element(data, data + shape_size(weights_node->get_shape()));
        int64_t levels = (weights_minimum == static_cast<int8_t>(-128)) ? 256 : 255;
        int64_t in_low = -(levels / 2), in_high = levels + in_low - 1;

        const auto& input_low = ov::op::v0::Constant::create(convert_node->get_element_type(), {}, {in_low});
        const auto& input_high = ov::op::v0::Constant::create(convert_node->get_element_type(), {}, {in_high});
        std::shared_ptr<ov::op::v0::Constant> zero_point;
        if (pattern_map.count(sub_c)) {
            const auto& sub_c_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(sub_c));
            zero_point = sub_c_node;
        } else if (pattern_map.count(sub_c_integer)) {
            const auto& sub_c_integer_node = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(sub_c_integer));
            zero_point = ov::op::v0::Constant::create(convert_node->get_element_type(),
                                                      sub_c_integer_node->get_output_shape(0),
                                                      sub_c_integer_node->get_vector<int8_t>());
        } else {
            zero_point = ov::op::v0::Constant::create(convert_node->get_element_type(), {}, {0});
        }

        const auto& output_low_const = op::util::make_try_fold<ov::op::v1::Subtract>(input_low, zero_point);
        const auto& output_low = op::util::make_try_fold<ov::op::v1::Multiply>(output_low_const, scale_node);
        const auto& output_high_const = op::util::make_try_fold<ov::op::v1::Subtract>(input_high, zero_point);
        const auto& output_high = op::util::make_try_fold<ov::op::v1::Multiply>(output_high_const, scale_node);

        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(convert_node,
                                                             input_low,
                                                             input_high,
                                                             output_low,
                                                             output_high,
                                                             levels);

        NodeVector nodes_to_copy_RT_info_from{multiply_node, scale_node, zero_point};
        if (pattern_map.count(sub))
            nodes_to_copy_RT_info_from.push_back(sub);

        ov::copy_runtime_info(nodes_to_copy_RT_info_from, fq);
        multiply_node->output(0).replace(fq->output(0));
        fq->set_friendly_name(multiply_node->get_friendly_name());

        if (ov::constant_folding_is_disabled(convert_node))
            ov::enable_constant_folding(convert_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}
