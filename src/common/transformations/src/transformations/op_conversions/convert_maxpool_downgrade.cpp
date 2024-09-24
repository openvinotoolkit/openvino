// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertMaxPool8ToMaxPool1::ConvertMaxPool8ToMaxPool1() {
    MATCHER_SCOPE(ConvertMaxPool8ToMaxPool1);

    auto maxpool_v8_pattern = pattern::wrap_type<ov::op::v8::MaxPool>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto maxpool_v8_node = ov::as_type_ptr<ov::op::v8::MaxPool>(m.get_match_root());

        if (!maxpool_v8_node || maxpool_v8_node->get_output_target_inputs(1).size() != 0)
            return false;

        for (auto dilation : maxpool_v8_node->get_dilations())
            if (dilation != 1)
                return false;

        auto maxpool_v1_node = std::make_shared<ov::op::v1::MaxPool>(maxpool_v8_node->input_value(0),
                                                                     maxpool_v8_node->get_strides(),
                                                                     maxpool_v8_node->get_pads_begin(),
                                                                     maxpool_v8_node->get_pads_end(),
                                                                     maxpool_v8_node->get_kernel(),
                                                                     maxpool_v8_node->get_rounding_type(),
                                                                     maxpool_v8_node->get_auto_pad());

        OPENVINO_SUPPRESS_DEPRECATED_START
        auto out_name = ov::op::util::create_ie_output_name(maxpool_v8_node->output(0));
        OPENVINO_SUPPRESS_DEPRECATED_END

        maxpool_v1_node->set_friendly_name(maxpool_v8_node->get_friendly_name());
        maxpool_v8_node->output(0).replace(maxpool_v1_node->output(0));
        ov::copy_runtime_info(maxpool_v8_node, maxpool_v1_node);
        maxpool_v8_node->clear_control_dependencies();

        OPENVINO_SUPPRESS_DEPRECATED_START
        ov::descriptor::set_ov_tensor_legacy_name(maxpool_v1_node->output(0).get_tensor(), out_name);
        OPENVINO_SUPPRESS_DEPRECATED_END

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(maxpool_v8_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertMaxPool14ToMaxPool8::ConvertMaxPool14ToMaxPool8() {
    MATCHER_SCOPE(ConvertMaxPool14ToMaxPool8);
    const auto max_pool_v14_pattern = pattern::wrap_type<ov::op::v14::MaxPool>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        using ov::op::v0::Constant;
        using ov::op::v0::Concat;
        using ov::op::v1::Subtract;
        using ov::op::v1::Multiply;
        using ov::op::v1::Greater;
        using ov::op::v1::Select;
        using ov::op::v1::ConvertLike;
        using ov::op::v1::Add;
        using ov::op::v3::ShapeOf;
        using ov::op::v4::Range;
        using ov::op::v8::Gather;
        using ov::op::v12::Pad;

        const auto max_pool_v14 = ov::as_type_ptr<ov::op::v14::MaxPool>(m.get_match_root());
        if (!max_pool_v14 || transformation_callback(max_pool_v14)) {
            return false;
        }
        const auto rounding_type_v14 = max_pool_v14->get_rounding_type();
        std::shared_ptr<ov::op::v8::MaxPool> max_pool_v8;
        NodeRegistry node_registry;
        if (rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH) {
            auto input = max_pool_v14->input_value(0);
            const auto strides = max_pool_v14->get_strides();
            const auto padding_begin = max_pool_v14->get_pads_begin();
            const auto padding_begin_node =
                node_registry.make<Constant>(element::i32, Shape{padding_begin.size()}, padding_begin);
            const auto padding_end = max_pool_v14->get_pads_end();
            const auto padding_end_node =
                node_registry.make<Constant>(element::i32, Shape{padding_end.size()}, padding_end);
            const auto zero = node_registry.make<Constant>(element::i32, Shape{}, 0);
            const auto one = node_registry.make<Constant>(element::i32, Shape{}, 1);
            const auto two = node_registry.make<Constant>(element::i32, Shape{}, 2);

            const auto pads_size = max_pool_v14->get_pads_begin().size();
            const auto pads_len = node_registry.make<Constant>(element::i32, Shape{}, pads_size);
            const auto pads_remaining =
                node_registry.make<Constant>(element::i32, Shape{2}, std::vector<int64_t>{0, 0});

            // gather input spatial dims and prepare for compare as values (in_dim + pad)
            const auto end = node_registry.make<Constant>(element::i32, Shape{}, pads_size + 2);
            const auto dim_idxs = node_registry.make<Range>(two, end, one, element::i32);
            const auto shape = node_registry.make<ShapeOf>(input, element::i32);
            const auto gth_in_dims = node_registry.make<Gather>(shape, dim_idxs, zero);
            const auto in_left_padded = node_registry.make<Add>(gth_in_dims, padding_begin_node);

            // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
            const auto mp = node_registry.make<ov::op::v8::MaxPool>(input,
                                                                    max_pool_v14->get_strides(),
                                                                    max_pool_v14->get_dilations(),
                                                                    max_pool_v14->get_pads_begin(),
                                                                    max_pool_v14->get_pads_end(),
                                                                    max_pool_v14->get_kernel(),
                                                                    ov::op::RoundingType::CEIL);
            const auto shape_of_mp = node_registry.make<ShapeOf>(mp, element::i32);
            const auto gth_out_dims = node_registry.make<Gather>(shape_of_mp, dim_idxs, zero);
            const auto out_sub_one = node_registry.make<Subtract>(gth_out_dims, one);
            const auto stride_node = node_registry.make<Constant>(element::i32, Shape{strides.size()}, strides);
            const auto out_mul_stride = node_registry.make<Multiply>(out_sub_one, stride_node);

            // if (in_dim + pad) > ((out_dim - 1) * stride) sliding window in bound use end padding.
            const auto in_gt_out = node_registry.make<Greater>(in_left_padded, out_mul_stride);
            const auto selected_pads = node_registry.make<Select>(in_gt_out, padding_end_node, zero);

            // apply padding on input clear pads attribute
            const auto pb = node_registry.make<Concat>(OutputVector{pads_remaining->output(0), padding_begin_node}, 0);
            const auto pe = node_registry.make<Concat>(OutputVector{pads_remaining, selected_pads}, 0);
            auto minus_inf =
                node_registry.make<Constant>(element::f32, Shape{}, -std::numeric_limits<float>::infinity());
            std::shared_ptr<ov::Node> convert_like_node = node_registry.make<ConvertLike>(minus_inf, input);
            const auto pad_node = node_registry.make<Pad>(input, pb, pe, convert_like_node, op::PadMode::CONSTANT);
            auto pads_begin = max_pool_v14->get_pads_begin();
            auto pads_end = max_pool_v14->get_pads_end();
            std::fill_n(pads_begin.begin(), pads_begin.size(), 0);
            std::fill_n(pads_end.begin(), pads_end.size(), 0);

            max_pool_v8 = node_registry.make<ov::op::v8::MaxPool>(pad_node,
                                                                  max_pool_v14->get_strides(),
                                                                  max_pool_v14->get_dilations(),
                                                                  pads_begin,
                                                                  pads_end,
                                                                  max_pool_v14->get_kernel(),
                                                                  ov::op::RoundingType::CEIL,
                                                                  ov::op::PadType::EXPLICIT,
                                                                  max_pool_v14->get_index_element_type(),
                                                                  max_pool_v14->get_axis());
            copy_runtime_info(max_pool_v14, node_registry.get());
        } else {
            max_pool_v8 = std::make_shared<ov::op::v8::MaxPool>(max_pool_v14->input_value(0),
                                                                max_pool_v14->get_strides(),
                                                                max_pool_v14->get_dilations(),
                                                                max_pool_v14->get_pads_begin(),
                                                                max_pool_v14->get_pads_end(),
                                                                max_pool_v14->get_kernel(),
                                                                rounding_type_v14,
                                                                max_pool_v14->get_auto_pad(),
                                                                max_pool_v14->get_index_element_type(),
                                                                max_pool_v14->get_axis());
            copy_runtime_info(max_pool_v14, max_pool_v8);
        }
        max_pool_v8->set_friendly_name(max_pool_v14->get_friendly_name());
        replace_node(max_pool_v14, max_pool_v8);
        max_pool_v14->clear_control_dependencies();
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(max_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
