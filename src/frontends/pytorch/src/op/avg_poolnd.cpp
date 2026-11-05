// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;
OutputVector translate_avg_pool_base(const NodeContext& context, int dims) {
    num_inputs_check(context, 2, 7);
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto const_0 = v0::Constant::create(element::i64, Shape{1}, {0});
    auto const_1 = v0::Constant::create(element::i64, Shape{1}, {1});
    bool is_static = input.get_partial_shape().rank().is_static();
    bool no_batch_dim = is_static && input.get_partial_shape().rank().get_length() == dims + 1;

    if (is_static) {
        if (no_batch_dim) {
            input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        }
    } else {
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_0));
        auto unsqueeze_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input));
        auto rank = context.mark_node(std::make_shared<v0::ShapeOf>(unsqueeze_shape));
        auto end_index = context.mark_node(std::make_shared<v1::Add>(rank, const_1));
        auto start_index = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims - 2}));
        auto reshape_pattern =
            context.mark_node(std::make_shared<v8::Slice>(unsqueeze_shape, start_index, end_index, const_1, const_0));
        input = context.mark_node(std::make_shared<v1::Reshape>(input, reshape_pattern, true));
    }

    auto kernel = context.const_input<Shape>(1);
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2);
    }
    if (context.input_is_none(2) || strides.size() == 0) {
        // In case strides are not provided default is kernel
        strides = kernel;
    }
    Shape pads;
    bool count_include_pad = true;
    if (context.input_is_none(3)) {
        count_include_pad = false;
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric padding
    }
    ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR;
    if (!(context.input_is_none(4))) {
        rounding_type = context.const_input<bool>(4) ? ov::op::RoundingType::CEIL_TORCH : ov::op::RoundingType::FLOOR;
    }
    if (!(context.input_is_none(5))) {
        count_include_pad = context.const_input<bool>(5);
    }
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(6),
                                "Translation for aten::avg_pool2d do not support divisor_override input.");

    // PyTorch 1D/2D ceil-mode pooling clips a kernel larger than an unpadded axis to the input, while AvgPool
    // requires it to fit. Excluded end padding of kernel - 1 makes it fit for any non-empty axis, and with
    // stride >= kernel it adds no window starting inside the input, so results are unchanged.
    bool exclude_pad = !count_include_pad;
    auto pads_begin = pads;
    auto pads_end = pads;
    bool extended = false;
    if (dims < 3 && rounding_type == RoundingType::CEIL_TORCH && strides.size() == kernel.size() &&
        pads.size() == kernel.size()) {
        const auto& shape = input.get_partial_shape();
        const bool has_spatial_dims = shape.rank().is_static() && shape.size() == kernel.size() + 2;
        for (size_t i = 0; i < kernel.size(); ++i) {
            const bool may_be_short =
                !has_spatial_dims || shape[i + 2].get_min_length() < static_cast<int64_t>(kernel[i]);
            if (kernel[i] > 1 && pads[i] == 0 && strides[i] >= kernel[i] && may_be_short) {
                pads_end[i] = kernel[i] - 1;
                extended = true;
            }
        }
    }
    Output<Node> window_count;
    std::shared_ptr<v0::Constant> padded_axes;
    if (extended && !exclude_pad) {
        // Included padding becomes explicit zeros, since the extra end padding must stay out of the divisor.
        std::vector<int64_t> explicit_pads(kernel.size() + 2, 0), axes, offsets, steps;
        for (size_t i = 0; i < kernel.size(); ++i) {
            if (pads[i] > 0) {
                explicit_pads[i + 2] = static_cast<int64_t>(pads[i]);
                pads_begin[i] = pads_end[i] = 0;
                axes.push_back(static_cast<int64_t>(i + 2));
                offsets.push_back(static_cast<int64_t>(pads[i] + strides[i] - 1));
                steps.push_back(static_cast<int64_t>(strides[i]));
            }
        }
        if (!axes.empty()) {
            padded_axes = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
            auto shape_of = context.mark_node(std::make_shared<v3::ShapeOf>(input));
            auto zero = v0::Constant::create(element::i64, Shape{}, {0});
            auto lengths = context.mark_node(std::make_shared<v8::Gather>(shape_of, padded_axes, zero));
            // PyTorch drops a last window starting in the right padding, which AvgPool now sees as data.
            auto offsets_const = v0::Constant::create(element::i64, Shape{offsets.size()}, offsets);
            auto steps_const = v0::Constant::create(element::i64, Shape{steps.size()}, steps);
            auto padded_lengths = context.mark_node(std::make_shared<v1::Add>(lengths, offsets_const));
            window_count = context.mark_node(std::make_shared<v1::Divide>(padded_lengths, steps_const));
            auto pads_const = v0::Constant::create(element::i64, Shape{explicit_pads.size()}, explicit_pads);
            input = context.mark_node(std::make_shared<v12::Pad>(input, pads_const, pads_const, PadMode::CONSTANT));
        }
    }
    Output<Node> res = context.mark_node(std::make_shared<v14::AvgPool>(input,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        kernel,
                                                                        exclude_pad || extended,
                                                                        rounding_type));
    if (padded_axes) {
        const auto& axes_shape = padded_axes->get_shape();
        auto zeros = v0::Constant::create(element::i64, axes_shape, {0});
        auto ones = v0::Constant::create(element::i64, axes_shape, {1});
        res = context.mark_node(std::make_shared<v8::Slice>(res, zeros, window_count, ones, padded_axes));
    }

    if (is_static) {
        if (no_batch_dim) {
            res = context.mark_node(std::make_shared<v0::Squeeze>(res, const_0));
        }
    } else {
        auto pooled_output_shape = context.mark_node(std::make_shared<v3::ShapeOf>(res));

        auto start_index_input = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims}));
        auto slice_input_shape =
            context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, start_index_input, const_1, const_0));

        auto start_index_pooled = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-dims}));
        auto end_index_pooled = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {2 + dims}));
        auto slice_pooled_output_shape = context.mark_node(
            std::make_shared<v8::Slice>(pooled_output_shape, start_index_pooled, end_index_pooled, const_1, const_0));

        auto concat_shape = context.mark_node(
            std::make_shared<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0));
        res = context.mark_node(std::make_shared<v1::Reshape>(res, concat_shape, true));
    }

    return {res};
};

OutputVector translate_avg_pool1d(const NodeContext& context) {
    return translate_avg_pool_base(context, 1);
};

OutputVector translate_avg_pool2d(const NodeContext& context) {
    return translate_avg_pool_base(context, 2);
};

OutputVector translate_avg_pool3d(const NodeContext& context) {
    return translate_avg_pool_base(context, 3);
};

}  // namespace ov::frontend::pytorch::op
