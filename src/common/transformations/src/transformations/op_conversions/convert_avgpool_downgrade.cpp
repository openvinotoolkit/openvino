// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v14 = ov::op::v14;

namespace ov::pass {

ConvertAvgPool14ToAvgPool1::ConvertAvgPool14ToAvgPool1() {
    MATCHER_SCOPE(ConvertAvgPool14ToAvgPool1);

    const auto avg_pool_v14_pattern = pattern::wrap_type<v14::AvgPool>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto avg_pool_v14 = ov::as_type_ptr<v14::AvgPool>(m.get_match_root());
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

        using v0::Constant;
        using v0::Concat;
        using v1::Pad;
        using v1::Subtract;
        using v1::ConvertLike;
        using v3::Broadcast;
        using v3::ShapeOf;
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
            const auto pads_begin_v1 =
                node_registry.make<v0::Concat>(OutputVector{std::move(pads_remaining), std::move(pads_begin_node)}, 0);
            const auto pads_end_v1 =
                node_registry.make<v0::Concat>(OutputVector{std::move(pads_remaining), std::move(pads_begin_node)}, 0);
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
        const auto avg_pool_v1 = node_registry.make<v1::AvgPool>(new_input,
                                                                 avg_pool_v14->get_strides(),
                                                                 pads_begin,
                                                                 pads_end,
                                                                 avg_pool_v14->get_kernel(),
                                                                 exclude_pad,
                                                                 rounding_type_v1,
                                                                 avg_pool_v14->get_auto_pad());
        std::shared_ptr<ov::Node> result = avg_pool_v1;
        if (rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH) {
            // Unlike CEIL_TORCH, CEIL keeps a last window starting in the right padding; slice it off.
            const auto& kernel = avg_pool_v14->get_kernel();
            const auto& strides = avg_pool_v14->get_strides();
            const auto& pads_begin_v14 = avg_pool_v14->get_pads_begin();
            const auto& pads_end_v14 = avg_pool_v14->get_pads_end();
            const auto auto_pad = avg_pool_v14->get_auto_pad();
            const bool has_explicit_pads = auto_pad == ov::op::PadType::EXPLICIT || auto_pad == ov::op::PadType::NOTSET;
            const auto& shape_in = input.get_partial_shape();
            const auto& shape_v14 = avg_pool_v14->get_output_partial_shape(0);
            const auto& shape_v1 = avg_pool_v1->get_output_partial_shape(0);
            const auto spatial = static_cast<int64_t>(kernel.size());
            std::vector<int64_t> axes, offsets, steps, stops;
            bool static_stops = true;
            for (int64_t i = 0; i < spatial; ++i) {
                const auto axis = i - spatial;
                const bool is_static = shape_in.rank().is_static() && shape_v14.rank().is_static() &&
                                       shape_in[axis].is_static() && shape_v14[axis].is_static() &&
                                       shape_v1[axis].is_static();
                if (is_static && shape_v14[axis].get_length() < shape_v1[axis].get_length()) {
                    stops.push_back(shape_v14[axis].get_length());
                    // Makes the runtime bound below evaluate to the static extent.
                    offsets.push_back(stops.back() - shape_in[axis].get_length());
                    steps.push_back(1);
                } else if (!is_static && has_explicit_pads && pads_end_v14[i] + strides[i] > kernel[i] &&
                           pads_end_v14[i] <= kernel[i] && (exclude_pad || pads_begin_v14[i] == pads_end_v14[i])) {
                    // With pads_end <= kernel at most one CEIL window starts in the padding, and the workaround
                    // above matches CEIL_TORCH only for symmetric padding.
                    static_stops = false;
                    offsets.push_back(static_cast<int64_t>(pads_begin_v14[i] + strides[i] - 1));
                    steps.push_back(static_cast<int64_t>(strides[i]));
                } else {
                    continue;
                }
                axes.push_back(axis);
            }
            if (!axes.empty()) {
                const auto axes_node = node_registry.make<Constant>(element::i64, Shape{axes.size()}, axes);
                ov::Output<ov::Node> stop;
                if (static_stops) {
                    stop = node_registry.make<Constant>(element::i64, Shape{stops.size()}, stops);
                } else {
                    // CEIL_TORCH keeps ceil((L + pads_begin) / stride) windows at most.
                    const auto shape = node_registry.make<ShapeOf>(input, element::i64);
                    const auto zero = node_registry.make<Constant>(element::i64, Shape{}, 0);
                    const auto lengths = node_registry.make<ov::op::v8::Gather>(shape, axes_node, zero);
                    const auto offsets_node =
                        node_registry.make<Constant>(element::i64, Shape{offsets.size()}, offsets);
                    const auto shifted = node_registry.make<v1::Add>(lengths, offsets_node);
                    const auto steps_node = node_registry.make<Constant>(element::i64, Shape{steps.size()}, steps);
                    stop = node_registry.make<v1::Divide>(shifted, steps_node);
                }
                const auto start = node_registry.make<Constant>(element::i64, Shape{axes.size()}, 0);
                const auto step = node_registry.make<Constant>(element::i64, Shape{axes.size()}, 1);
                result = node_registry.make<ov::op::v8::Slice>(avg_pool_v1, start, stop, step, axes_node);
            }
        }
        result->set_friendly_name(avg_pool_v14->get_friendly_name());
        copy_runtime_info(avg_pool_v14, node_registry.get());
        replace_node(avg_pool_v14, result);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
