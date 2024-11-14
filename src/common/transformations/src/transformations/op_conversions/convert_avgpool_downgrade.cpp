// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertAvgPool14ToAvgPool1::ConvertAvgPool14ToAvgPool1() {
    MATCHER_SCOPE(ConvertAvgPool14ToAvgPool1);

    const auto avg_pool_v14_pattern = pattern::wrap_type<ov::op::v14::AvgPool>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto avg_pool_v14 = ov::as_type_ptr<ov::op::v14::AvgPool>(m.get_match_root());
        if (!avg_pool_v14 || transformation_callback(avg_pool_v14)) {
            return false;
        }
        const auto rounding_type_v14 = avg_pool_v14->get_rounding_type();
        const auto rounding_type_v1 =
            rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH ? ov::op::RoundingType::CEIL : rounding_type_v14;

        const auto exclude_pad = avg_pool_v14->get_exclude_pad();
        const auto input = avg_pool_v14->input_value(0);
        NodeRegistry node_registry;
        ov::Shape pads_begin;
        ov::Shape pads_end;
        ov::Output<ov::Node> new_input;

        using ov::op::v0::Constant;
        using ov::op::v0::Concat;
        using ov::op::v1::Pad;
        using ov::op::v1::Subtract;
        using ov::op::v1::ConvertLike;
        using ov::op::v3::Broadcast;
        using ov::op::v3::ShapeOf;
        using ov::op::v4::Range;

        if (!exclude_pad && rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH) {
            const auto zero = node_registry.make<Constant>(element::f32, Shape{}, 0);
            const auto zero_node = node_registry.make<ConvertLike>(zero, input);
            const auto zero_i64 = node_registry.make<Constant>(element::i64, Shape{}, 0);
            const auto shape = node_registry.make<ShapeOf>(input, element::i64);
            const auto rank = node_registry.make<ShapeOf>(shape, element::i64);
            const auto pads_begin_v14 = avg_pool_v14->get_pads_begin();
            const auto pads_begin_node =
                node_registry.make<Constant>(element::i64, Shape{pads_begin_v14.size()}, pads_begin_v14);
            const auto pads_end_v14 = avg_pool_v14->get_pads_end();
            const auto pads_end_node =
                node_registry.make<Constant>(element::i64, Shape{pads_end_v14.size()}, pads_end_v14);
            const auto pads_len = node_registry.make<Constant>(element::i64, Shape{}, pads_begin_v14.size());
            const auto pads_diff = node_registry.make<Subtract>(rank, pads_len);
            const auto pads_remaining = node_registry.make<Broadcast>(zero_i64, pads_diff);
            const auto pads_begin_v1 = node_registry.make<ov::op::v0::Concat>(
                OutputVector{std::move(pads_remaining), std::move(pads_begin_node)},
                0);
            const auto pads_end_v1 = node_registry.make<ov::op::v0::Concat>(
                OutputVector{std::move(pads_remaining), std::move(pads_begin_node)},
                0);
            const auto pad_node =
                node_registry.make<Pad>(input, pads_begin_v1, pads_end_v1, zero_node, ov::op::PadMode::CONSTANT);
            pads_begin = Shape(pads_begin_v14.size(), 0);
            pads_end = Shape(pads_begin_v14.size(), 0);
            new_input = pad_node;
        } else {
            pads_begin = avg_pool_v14->get_pads_begin();
            pads_end = avg_pool_v14->get_pads_end();
            new_input = input;
        }
        const auto avg_pool_v1 = node_registry.make<ov::op::v1::AvgPool>(new_input,
                                                                         avg_pool_v14->get_strides(),
                                                                         pads_begin,
                                                                         pads_end,
                                                                         avg_pool_v14->get_kernel(),
                                                                         exclude_pad,
                                                                         rounding_type_v1,
                                                                         avg_pool_v14->get_auto_pad());
        avg_pool_v1->set_friendly_name(avg_pool_v14->get_friendly_name());
        copy_runtime_info(avg_pool_v14, node_registry.get());
        replace_node(avg_pool_v14, avg_pool_v1);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
