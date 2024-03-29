// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertMaxPool8ToMaxPool1::ConvertMaxPool8ToMaxPool1() {
    MATCHER_SCOPE(ConvertMaxPool8ToMaxPool1);

    auto maxpool_v8_pattern = pattern::wrap_type<ov::op::v8::MaxPool>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto maxpool_v8_node = std::dynamic_pointer_cast<ov::op::v8::MaxPool>(m.get_match_root());

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

    const matcher_pass_callback callback = [](pattern::Matcher& m) {
        const auto max_pool_v14 = std::dynamic_pointer_cast<ov::op::v14::MaxPool>(m.get_match_root());

        if (!max_pool_v14 || max_pool_v14->get_output_target_inputs(1).size() != 0) {
            return false;
        }

        const auto rounding_type_v14 = max_pool_v14->get_rounding_type();
        if (rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH) {
        const auto zero = ov::op::v0::Constant::create(element::i32, Shape{}, {0});
        const auto one = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
        const auto two = ov::op::v0::Constant::create(element::i32, Shape{}, {2});

        const auto padding = max_pool_v14->get_pads_end();
        const auto pads_len = ov::op::v0::Constant::create(element::i32, Shape{}, {pads.size()});
        const auto pads_remaining = ov::op::v0::Constant::create(element::i32, Shape{2}, {0, 0});

        // gather input spatial dims and prepare for compare as values (in_dim + pad)
        const auto input_shape_rank = max_pool_v14->input_value(0).get_shape().size();
        const auto end = ov::op::v0::Constant::create(element::i32, Shape{}, {pads.size() + 2});
        const auto dim_idxs = std::make_shared<ov::op::v4::Range>(two, end, one, element::i32);
        const auto gth_in_dims =
            std::make_shared<ov::op::v8::Gather>(std::get<0>(input_shape_rank), dim_idxs, zero);
        const auto in_left_padded = std::make_shared<ov::op::v1::Add>(gth_in_dims, padding);

        // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
        const auto mp = 
            std::make_shared<ov::op::v8::MaxPool>(max_pool_v14->input_value(0), strides, dilations, pads, pads, max_pool_v14->get_kernel(), ov::op::RoundingType::CEIL);
        const auto shape_of_mp = std::make_shared<ov::op::v3::ShapeOf>(mp, element::i32);
        const auto gth_out_dims = std::make_shared<ov::op::v8::Gather>(shape_of_mp, dim_idxs, zero);
        const auto out_sub_one = std::make_shared<ov::op::v1::Subtract>(gth_out_dims, one);
        const auto stride_node = max_pool_v14->get_strides();
        const auto out_mul_stride = std::make_shared<ov::op::v1::Multiply>(out_sub_one, stride_node);

        // if (in_dim + pad) > ((out_dim - 1) * stride) sliding window in bound use end padding.
        const auto in_gt_out = std::make_shared<ov::op::v1::Greater>(in_left_padded, out_mul_stride);
        const auto selected_pads = std::make_shared<ov::op::v1::Select>(in_gt_out, padding, zero);

        // apply padding on input clear pads attribute
        const auto pb = std::make_shared<ov::op::v0::Concat>(OutputVector{pads_remaining, padding}, 0);
        const auto pe = std::make_shared<ov::op::v0::Concat>(OutputVector{pads_remaining, selected_pads}, 0);
        auto minus_inf =
            ov::op::v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()});
        minus_inf = std::make_shared<ov::op::v1::ConvertLike>(minus_inf, max_pool_v14-);
        input = std::make_shared<ov::op::v12::Pad>(input, pb, pe, minus_inf, op::PadMode::CONSTANT);
        std::fill_n(pads.begin(), pads.size(), 0);
        } else {
            const auto rounding_type_v8 =
            rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH ? ov::op::RoundingType::CEIL : rounding_type_v14;
            const auto max_pool_v8 = std::make_shared<ov::op::v8::MaxPool>(max_pool_v14->input_value(0),
                                                                           max_pool_v14->get_strides(),
                                                                           max_pool_v14->get_dilations(),
                                                                           max_pool_v14->get_pads_begin(),
                                                                           max_pool_v14->get_pads_end(),
                                                                           max_pool_v14->get_kernel(),
                                                                           rounding_type_v8,
                                                                           max_pool_v14->get_auto_pad(),
                                                                           max_pool_v14->get_index_element_type(),
                                                                           max_pool_v14->get_axis());
            max_pool_v8->set_friendly_name(max_pool_v14->get_friendly_name());
            max_pool_v14->output(0).replace(max_pool_v8->output(0));
            copy_runtime_info(max_pool_v14, max_pool_v8);
            max_pool_v14->clear_control_dependencies();
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(max_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
