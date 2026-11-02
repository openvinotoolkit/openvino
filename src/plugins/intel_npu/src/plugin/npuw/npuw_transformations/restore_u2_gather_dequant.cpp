// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "restore_u2_gather_dequant.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

constexpr float MAX_U2_VALUE = 3.f;
constexpr size_t CODES_PER_BYTE = 4;

}  // namespace

class RestoreU2GatherDequantMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::RestoreU2GatherDequantMatcher");

    RestoreU2GatherDequantMatcher() {
        auto packed = opp::wrap_type<ov::op::v0::Constant>();
        auto indices = opp::any_input();
        auto axis = opp::wrap_type<ov::op::v0::Constant>();
        auto gather = opp::wrap_type<ov::op::v8::Gather>({packed, indices, axis});
        auto bytes = opp::wrap_type<ov::op::v0::Convert>({gather});
        auto expanded = opp::wrap_type<ov::op::v1::Reshape>({bytes, opp::any_input()});
        auto shift_amounts = opp::wrap_type<ov::op::v0::Constant>();
        auto shifted = opp::wrap_type<ov::op::v15::BitwiseRightShift>({expanded, shift_amounts});
        auto mask = opp::wrap_type<ov::op::v0::Constant>();
        auto codes = opp::wrap_type<ov::op::v13::BitwiseAnd>({shifted, mask});
        auto grouped = opp::wrap_type<ov::op::v1::Reshape>({codes, opp::any_input()});
        auto values = opp::wrap_type<ov::op::v0::Convert>({grouped});
        auto zero_point = opp::wrap_type<ov::op::v0::Constant>();
        // The GGUF frontend emits the zero point as a Convert of a Constant
        auto zero_point_f16 = opp::optional<ov::op::v0::Convert>({zero_point->output(0)});
        auto centered = opp::wrap_type<ov::op::v1::Subtract>({values, zero_point_f16});
        auto scales = opp::wrap_type<ov::op::v0::Constant>();
        auto scales_gather = opp::wrap_type<ov::op::v8::Gather>({scales, indices, axis});
        auto dequantized = opp::wrap_type<ov::op::v1::Multiply>({centered, scales_gather});
        auto restored = opp::wrap_type<ov::op::v1::Reshape>({dequantized, opp::any_input()});
        auto result = opp::wrap_type<ov::op::v0::Convert>({restored});

        ov::matcher_pass_callback callback = [=](opp::Matcher& matcher) {
            const auto& pattern_map = matcher.get_pattern_value_map();
            const auto packed_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(packed).get_node_shared_ptr());
            const auto scales_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(scales).get_node_shared_ptr());
            const auto axis_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(axis).get_node_shared_ptr());
            const auto shift_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(shift_amounts).get_node_shared_ptr());
            const auto mask_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(mask).get_node_shared_ptr());
            const auto vocab_gather = ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather).get_node_shared_ptr());
            const auto scales_gather_node = pattern_map.at(scales_gather).get_node_shared_ptr();

            // Both Gathers must select the same rows, and the codes must be unpacked from the LSB first,
            // which is the element order of u2 in a Constant.
            if (packed_const->get_element_type() != ov::element::u8 || packed_const->get_shape().size() != 3 ||
                scales_const->get_element_type() != ov::element::f16 || vocab_gather->get_batch_dims() != 0 ||
                axis_const->cast_vector<int64_t>() != std::vector<int64_t>{0} ||
                shift_const->cast_vector<int64_t>() != std::vector<int64_t>{0, 2, 4, 6} ||
                mask_const->cast_vector<int64_t>() != std::vector<int64_t>{3} ||
                scales_gather_node->input_value(1) != pattern_map.at(indices)) {
                return false;
            }

            const auto& packed_shape = packed_const->get_shape();
            const size_t vocab = packed_shape[0];
            const size_t groups = packed_shape[1];
            const size_t group_size = packed_shape[2] * CODES_PER_BYTE;
            const size_t hidden = groups * group_size;
            const auto result_shape = pattern_map.at(result).get_partial_shape();
            if (scales_const->get_shape() != ov::Shape{vocab, groups, 1} || result_shape.rank().is_dynamic() ||
                result_shape[result_shape.rank().get_length() - 1] != hidden) {
                return false;
            }

            // The shifts must broadcast along a size-1 axis that follows the packed bytes, so the four codes of
            // a byte stay adjacent after the grouping Reshape, as in the u2 element order.
            const auto expanded_shape = pattern_map.at(expanded).get_partial_shape();
            const auto grouped_shape = pattern_map.at(grouped).get_partial_shape();
            const auto& shift_shape = shift_const->get_shape();
            if (ov::shape_size(shift_shape) != CODES_PER_BYTE || shift_shape.back() != CODES_PER_BYTE ||
                expanded_shape.rank().is_dynamic() || grouped_shape.rank().is_dynamic() || expanded_shape.size() < 2 ||
                shift_shape.size() > expanded_shape.size() || grouped_shape.size() + 1 != expanded_shape.size() ||
                expanded_shape[expanded_shape.size() - 1] != 1 ||
                expanded_shape[expanded_shape.size() - 2] != packed_shape[2] ||
                grouped_shape[grouped_shape.size() - 1] != group_size) {
                return false;
            }

            const auto zero_point_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zero_point).get_node_shared_ptr());
            if (!zero_point_const->get_shape().empty()) {
                return false;
            }
            const float zero_point_value = zero_point_const->cast_vector<float>().front();
            if (zero_point_value != std::floor(zero_point_value) || zero_point_value < 0.f ||
                zero_point_value > MAX_U2_VALUE) {
                return false;
            }

            auto codes_u2 = std::make_shared<ov::op::v0::Constant>(ov::element::u2,
                                                                   ov::Shape{vocab, groups, group_size},
                                                                   packed_const->get_data_ptr(),
                                                                   packed_const);
            auto zero_point_u2 = ov::op::v0::Constant::create(
                ov::element::u2,
                ov::Shape{vocab, groups, 1},
                std::vector<uint8_t>(vocab * groups, static_cast<uint8_t>(zero_point_value)));
            auto codes_f16 = std::make_shared<ov::op::v0::Convert>(codes_u2, ov::element::f16);
            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point_u2, ov::element::f16);
            auto centered_f16 = std::make_shared<ov::op::v1::Subtract>(codes_f16, zero_point_f16);
            auto dequantized_f16 = std::make_shared<ov::op::v1::Multiply>(centered_f16, scales_const);
            auto table_f16 = std::make_shared<ov::op::v1::Reshape>(
                dequantized_f16,
                ov::op::v0::Constant::create(
                    ov::element::i64,
                    ov::Shape{2},
                    std::vector<int64_t>{static_cast<int64_t>(vocab), static_cast<int64_t>(hidden)}),
                false);
            auto table_f32 = std::make_shared<ov::op::v0::Convert>(table_f16, ov::element::f32);
            auto rows = std::make_shared<ov::op::v8::Gather>(table_f32, pattern_map.at(indices), axis_const);

            rows->set_friendly_name(pattern_map.at(result).get_node_shared_ptr()->get_friendly_name());
            ov::copy_runtime_info(pattern_map.at(result).get_node_shared_ptr(), rows);
            return ov::replace_output_update_name(pattern_map.at(result), rows->output(0));
        };

        register_matcher(std::make_shared<opp::Matcher>(result, "RestoreU2GatherDequantMatcher"), callback);
    }
};

ov::npuw::RestoreU2GatherDequant::RestoreU2GatherDequant() {
    add_matcher<RestoreU2GatherDequantMatcher>();
}
