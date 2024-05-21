// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max_poolnd(const NodeContext& context) {
    num_inputs_check(context, 3, 6);
    auto kernel = context.const_input<Shape>(1);
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2);
    }
    const bool use_kernel = context.input_is_none(2) || (strides.size() == 0);
    if (use_kernel) {
        // In case strides are not provided default is kernel
        strides = kernel;
    }
    Shape pads;
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    }
    Strides dilations;
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL : RoundingType::FLOOR;
    }

    auto input = context.get_input(0);
    if (rounding_type == RoundingType::CEIL) {
        // The corner case of Max Pooling with ceil_mode on
        // PyTorch allows sliding window go off bound, which leads to this accommodation.
        // More detail on https://github.com/pytorch/pytorch/issues/57178
        const auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        const auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
        const auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));

        const auto& padding =
            context.input_is_none(3)
                ? context.mark_node(std::make_shared<v0::Constant>(element::i32, Shape{pads.size()}, 0))->output(0)
                : get_input_as_i32(context, 3);
        const auto pads_len = context.mark_node(v0::Constant::create(element::i32, Shape{}, {pads.size()}));
        const auto pads_remaining = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {0, 0}));

        // gather input spatial dims and prepare for compare as values (in_dim + pad)
        const auto input_shape_rank = get_shape_rank(context, input);
        const auto end = context.mark_node(v0::Constant::create(element::i32, Shape{}, {pads.size() + 2}));
        const auto dim_idxs = context.mark_node(std::make_shared<v4::Range>(two, end, one, element::i32));
        const auto gth_in_dims =
            context.mark_node(std::make_shared<v8::Gather>(std::get<0>(input_shape_rank), dim_idxs, zero));
        const auto in_left_padded = context.mark_node(std::make_shared<v1::Add>(gth_in_dims, padding));

        // gather output spatial dims and prepare it for compare as values (out_dim - 1) * stride
        const auto mp = context.mark_node(
            std::make_shared<v8::MaxPool>(input, strides, dilations, pads, pads, kernel, rounding_type));
        const auto shape_of_mp = context.mark_node(std::make_shared<v3::ShapeOf>(mp, element::i32));
        const auto gth_out_dims = context.mark_node(std::make_shared<v8::Gather>(shape_of_mp, dim_idxs, zero));
        const auto out_sub_one = context.mark_node(std::make_shared<v1::Subtract>(gth_out_dims, one));
        const auto& stride_node = use_kernel ? context.get_input(1) : context.get_input(2);
        const auto stride_node_i32 = context.mark_node(std::make_shared<v0::Convert>(stride_node, element::i32));
        const auto out_mul_stride = context.mark_node(std::make_shared<v1::Multiply>(out_sub_one, stride_node_i32));

        // if (in_dim + pad) > ((out_dim - 1) * stride) sliding window in bound use end padding.
        const auto in_gt_out = context.mark_node(std::make_shared<v1::Greater>(in_left_padded, out_mul_stride));
        const auto selected_pads = context.mark_node(std::make_shared<v1::Select>(in_gt_out, padding, zero));

        // apply padding on input clear pads attribute
        const auto pb = context.mark_node(std::make_shared<v0::Concat>(OutputVector{pads_remaining, padding}, 0));
        const auto& pe =
            context.mark_node(std::make_shared<v0::Concat>(OutputVector{pads_remaining, selected_pads}, 0));
        auto minus_inf =
            context.mark_node(v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()}));
        minus_inf = context.mark_node(std::make_shared<v1::ConvertLike>(minus_inf, input));
        input = context.mark_node(std::make_shared<v12::Pad>(input, pb, pe, minus_inf, op::PadMode::CONSTANT));
        std::fill_n(pads.begin(), pads.size(), 0);
    }

    auto res = context.mark_node(std::make_shared<v8::MaxPool>(input,
                                                               strides,
                                                               dilations,
                                                               pads,
                                                               pads,
                                                               kernel,
                                                               rounding_type,
                                                               PadType::EXPLICIT,
                                                               element::i64,
                                                               2));
    if (context.get_output_size() == 2) {
        auto out1 = res->output(0);
        auto out2 = res->output(1);
        return {std::move(out1), std::move(out2)};
    } else {
        return {res};
    }
};

OutputVector translate_max_poolnd_fx(const NodeContext& context) {
    auto output = translate_max_poolnd(context);
    return {context.mark_node(make_list_construct(output))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
