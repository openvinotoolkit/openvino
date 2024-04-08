// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertAvgPool14ToAvgPool1::ConvertAvgPool14ToAvgPool1() {
    MATCHER_SCOPE(ConvertAvgPool14ToAvgPool1);

    const auto avg_pool_v14_pattern = pattern::wrap_type<ov::op::v14::AvgPool>();

    const matcher_pass_callback callback = [](pattern::Matcher& m) {
        const auto avg_pool_v14 = std::dynamic_pointer_cast<ov::op::v14::AvgPool>(m.get_match_root());
        if (!avg_pool_v14) {
            return false;
        }

        std::shared_ptr<ov::op::v1::AvgPool> avg_pool_v1;
        if (avg_pool_v14->get_rounding_type() == ov::op::RoundingType::CEIL_TORCH) {
            if (avg_pool_v14->is_dynamic()) {
                return false;
            }
            auto input = avg_pool_v14->input_value(0);
            const auto strides = avg_pool_v14->get_strides();
            const auto padding_begin = avg_pool_v14->get_pads_begin();
            const auto padding_begin_node =
                ov::op::v0::Constant::create(element::i64, Shape{padding_begin.size()}, padding_begin);
            const auto padding_end = avg_pool_v14->get_pads_end();
            const auto padding_end_node =
                ov::op::v0::Constant::create(element::i64, Shape{padding_end.size()}, padding_end);
            const auto zero = ov::op::v0::Constant::create(element::i64, Shape{}, {0});
            const auto one = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
            const auto two = ov::op::v0::Constant::create(element::i64, Shape{}, {2});

            const auto pads_size = avg_pool_v14->get_pads_begin().size();
            const auto pads_len = ov::op::v0::Constant::create(element::i64, Shape{}, {pads_size});
            const auto pads_remaining = ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 0});

            // gather input spatial dims and prepare for compare as values (in_dim + pad)
            const auto end = ov::op::v0::Constant::create(element::i64, Shape{}, {pads_size + 2});
            const auto dim_idxs = std::make_shared<ov::op::v4::Range>(two, end, one, element::i64);
            const auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, element::i64);
            const auto gth_in_dims = std::make_shared<ov::op::v8::Gather>(shape, dim_idxs, zero);
            const auto in_left_padded = std::make_shared<ov::op::v1::Add>(gth_in_dims, padding_begin_node);

            // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
            const auto ap = std::make_shared<ov::op::v1::AvgPool>(input,
                                                                  avg_pool_v14->get_strides(),
                                                                  avg_pool_v14->get_pads_begin(),
                                                                  avg_pool_v14->get_pads_end(),
                                                                  avg_pool_v14->get_kernel(),
                                                                  avg_pool_v14->get_exclude_pad(),
                                                                  ov::op::RoundingType::CEIL);
            const auto shape_of_ap = std::make_shared<ov::op::v3::ShapeOf>(ap, element::i64);
            const auto gth_out_dims = std::make_shared<ov::op::v8::Gather>(shape_of_ap, dim_idxs, zero);
            const auto out_sub_one = std::make_shared<ov::op::v1::Subtract>(gth_out_dims, one);
            const auto stride_node = ov::op::v0::Constant::create(element::i64, Shape{strides.size()}, strides);
            const auto out_mul_stride = std::make_shared<ov::op::v1::Multiply>(out_sub_one, stride_node);

            // if (in_dim + pad) < ((out_dim - 1) * stride) sliding window in bound use end padding.
            const auto in_gt_out = std::make_shared<ov::op::v1::Greater>(out_mul_stride, in_left_padded);
            const auto selected_pads = std::make_shared<ov::op::v1::Select>(in_gt_out, padding_end_node, zero);

            // apply padding on input clear pads attribute
            const auto pb =
                std::make_shared<ov::op::v0::Concat>(OutputVector{pads_remaining->output(0), padding_end_node}, 0);
            const auto pe = std::make_shared<ov::op::v0::Concat>(OutputVector{pads_remaining, selected_pads}, 0);
            auto minus_inf =
                ov::op::v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()})
                    ->output(0);
            std::shared_ptr<ov::Node> convert_like_node = std::make_shared<ov::op::v1::ConvertLike>(minus_inf, input);
            const auto pad_node =
                std::make_shared<ov::op::v12::Pad>(input, pb, pe, convert_like_node, op::PadMode::CONSTANT);
            auto pads_begin = avg_pool_v14->get_pads_begin();
            auto pads_end = avg_pool_v14->get_pads_end();
            std::fill_n(pads_begin.begin(), pads_begin.size(), 0);
            std::fill_n(pads_end.begin(), pads_end.size(), 0);

            avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(pad_node,
                                                                avg_pool_v14->get_strides(),
                                                                pads_begin,
                                                                pads_end,
                                                                avg_pool_v14->get_kernel(),
                                                                avg_pool_v14->get_exclude_pad(),
                                                                ov::op::RoundingType::CEIL,
                                                                ov::op::PadType::EXPLICIT);
            copy_runtime_info(avg_pool_v14,
                              ov::NodeVector{dim_idxs,
                                             shape,
                                             gth_in_dims,
                                             in_left_padded,
                                             ap,
                                             shape_of_ap,
                                             gth_out_dims,
                                             out_sub_one,
                                             out_mul_stride,
                                             in_gt_out,
                                             selected_pads,
                                             pb,
                                             pe,
                                             convert_like_node,
                                             pad_node,
                                             avg_pool_v1});
        } else {
            avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(avg_pool_v14->input_value(0),
                                                                avg_pool_v14->get_strides(),
                                                                avg_pool_v14->get_pads_begin(),
                                                                avg_pool_v14->get_pads_end(),
                                                                avg_pool_v14->get_kernel(),
                                                                avg_pool_v14->get_exclude_pad(),
                                                                avg_pool_v14->get_rounding_type(),
                                                                avg_pool_v14->get_auto_pad());
            copy_runtime_info(avg_pool_v14, avg_pool_v1);
        }
        avg_pool_v1->set_friendly_name(avg_pool_v14->get_friendly_name());
        copy_runtime_info(avg_pool_v14, avg_pool_v1);
        replace_node(avg_pool_v14, avg_pool_v1);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
