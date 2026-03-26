// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "transformations/pattern_blocks/qdq_block.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

ConvertQuantizeDequantize::ConvertQuantizeDequantize(const ov::element::TypeVector& supported_low_precisions,
                                                     const ov::element::TypeVector& supported_original_precisions) {
    MATCHER_SCOPE(ConvertQuantizeDequantize);

    using namespace ov::pass::pattern;
    auto qdq_block = std::make_shared<op::QDQBlock>(
        type_matches_any(supported_original_precisions),
        type_matches_any(supported_low_precisions) && consumers_count(1),
        type_matches_any(supported_original_precisions) && consumers_count(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto data = *qdq_block->get_anchor("data_pattern", pattern_map);
        auto input_low = *qdq_block->get_anchor("input_low_pattern", pattern_map);
        auto input_high = *qdq_block->get_anchor("input_high_pattern", pattern_map);
        auto output_low =
            ov::as_type_ptr<v0::Constant>(qdq_block->get_anchor("output_low_pattern", pattern_map)->get_node_shared_ptr());
        if (!output_low)
            return false;
        auto output_high =
            ov::as_type_ptr<v0::Constant>(qdq_block->get_anchor("output_high_pattern", pattern_map)->get_node_shared_ptr());
        if (!output_high)
            return false;
        auto fq = ov::as_type_ptr<v0::FakeQuantize>(qdq_block->get_anchor("fq_pattern", pattern_map)->get_node_shared_ptr());
        if (!fq)
            return false;
        auto scale = *qdq_block->get_anchor("scale_pattern", pattern_map);
        auto convert1 = *qdq_block->get_anchor("q_convert_pattern", pattern_map);
        auto convert2 = *qdq_block->get_anchor("dq_convert_pattern", pattern_map);
        auto mul = qdq_block->get_anchor("mul_pattern", pattern_map)->get_node_shared_ptr();

        static const std::unordered_set<size_t> supported_levels{256, 65536};
        const auto levels = fq->get_levels();
        if (!supported_levels.count(levels))
            return false;

        float out_low_val;
        if (!op_util::get_single_value(output_low, out_low_val))
            return false;
        float out_high_val;
        if (!op_util::get_single_value(output_high, out_high_val))
            return false;

#define PRECISION_LIMITS_FOR(type)                                                                           \
    {                                                                                                        \
        ov::element::type,                                                                                   \
            std::make_pair(                                                                                  \
                static_cast<float>(std::numeric_limits<ov::fundamental_type_for<ov::element::type>>::min()), \
                static_cast<float>(std::numeric_limits<ov::fundamental_type_for<ov::element::type>>::max())) \
    }

        static const std::unordered_map<ov::element::Type_t, std::pair<float, float>> supported_intervals{
            PRECISION_LIMITS_FOR(i8),
            PRECISION_LIMITS_FOR(u8),
            PRECISION_LIMITS_FOR(i16),
            PRECISION_LIMITS_FOR(u16)};
#undef PRECISION_LIMITS_FOR

        const auto& type = convert1.get_element_type();
        // check if (out_low_val, out_high_val) pair is mapped on the expected precision ranges
        auto interval_it = supported_intervals.find(type);
        if (interval_it == supported_intervals.end() ||
            interval_it->second != std::make_pair(out_low_val, out_high_val)) {
            return false;
        }

        const auto zp_anchor = qdq_block->get_anchor("zero_point_pattern", pattern_map);
        const bool has_zero_point = zp_anchor.has_value();
        std::shared_ptr<Node> new_out_low = output_low, new_out_high = output_high;
        if (has_zero_point) {
            const auto& zero_point = *zp_anchor;
            new_out_low = std::make_shared<v1::Subtract>(new_out_low, zero_point);
            new_out_high = std::make_shared<v1::Subtract>(new_out_high, zero_point);
        }
        new_out_low = std::make_shared<v1::Multiply>(new_out_low, scale);
        new_out_high = std::make_shared<v1::Multiply>(new_out_high, scale);

        // check if new_out_low/high shapes are broadcastable to FQ's input
        auto data_shape = data.get_partial_shape();
        if (data_shape.rank().is_dynamic())
            return false;
        auto out_low_shape = new_out_low->get_output_partial_shape(0);
        if (out_low_shape.rank().is_dynamic() || out_low_shape.rank().get_length() > data_shape.rank().get_length())
            return false;
        auto out_high_shape = new_out_high->get_output_partial_shape(0);
        if (out_high_shape.rank().is_dynamic() || out_high_shape.rank().get_length() > data_shape.rank().get_length())
            return false;

        std::shared_ptr<Node> const_out_low = ov::util::get_constant_from_source(new_out_low);
        if (const_out_low)
            new_out_low = const_out_low;
        std::shared_ptr<Node> const_out_high = ov::util::get_constant_from_source(new_out_high);
        if (const_out_high)
            new_out_high = const_out_high;

        auto new_fq =
            std::make_shared<v0::FakeQuantize>(data, input_low, input_high, new_out_low, new_out_high, levels);
        new_fq->set_friendly_name(mul->get_friendly_name());

        copy_runtime_info({fq, convert1.get_node_shared_ptr(), convert2.get_node_shared_ptr()}, new_fq);
        replace_node(mul, new_fq);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(qdq_block, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
