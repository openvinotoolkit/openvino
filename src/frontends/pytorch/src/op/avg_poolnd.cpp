// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/reduce_mean.hpp"
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

    // PyTorch 1D/2D ceil-mode pooling clips a kernel larger than the padded axis to it, while AvgPool requires it
    // to fit. Excluded extra end padding makes it fit exactly for the extents PyTorch accepts.
    bool exclude_pad = !count_include_pad;
    auto pads_begin = pads;
    auto pads_end = pads;
    bool extended = false;
    std::vector<bool> sliced(kernel.size(), false);
    const auto& shape = input.get_partial_shape();
    const bool has_spatial_dims = shape.rank().is_static() && shape.size() == kernel.size() + 2;
    const bool is_supported = dims < 3 && rounding_type == RoundingType::CEIL_TORCH &&
                              strides.size() == kernel.size() && pads.size() == kernel.size();
    // A kernel covering every static unpadded axis yields a single window, so reduce instead of scanning the kernel.
    bool single_window = is_supported && has_spatial_dims;
    bool oversized = false;
    for (size_t i = 0; i < kernel.size() && single_window; ++i) {
        const auto& dim = shape[i + 2];
        const auto k = static_cast<int64_t>(kernel[i]);
        single_window = pads[i] == 0 && dim.is_static() && dim.get_length() > 0 && dim.get_length() <= k &&
                        dim.get_length() + static_cast<int64_t>(strides[i]) > k;
        oversized = oversized || (single_window && dim.get_length() < k);
    }
    single_window = single_window && oversized;
    if (is_supported && !single_window) {
        for (size_t i = 0; i < kernel.size(); ++i) {
            const auto min_padded_length =
                has_spatial_dims ? shape[i + 2].get_min_length() + 2 * static_cast<int64_t>(pads[i]) : 0;
            // Padding of at least (kernel - 1) / 2 per side already fits the kernel to any non-empty axis.
            if (strides[i] > 1 && kernel[i] > 2 * pads[i] + 1 && min_padded_length < static_cast<int64_t>(kernel[i])) {
                // Included padding is made explicit below; AvgPool requires excluded padding below the kernel.
                const auto kept_pad = exclude_pad ? pads[i] : 0;
                pads_end[i] += std::min(strides[i], kernel[i] - kept_pad) - 1;
                extended = true;
                // The extra padding may add a window past the last one PyTorch uses.
                sliced[i] = strides[i] + kept_pad < kernel[i];
            }
        }
    }
    std::vector<int64_t> explicit_pads(kernel.size() + 2, 0);
    bool pad_explicitly = false;
    for (size_t i = 0; i < kernel.size() && extended && !exclude_pad; ++i) {
        if (pads[i] > 0) {
            // Included padding becomes explicit zeros, since the extra end padding must stay out of the divisor.
            explicit_pads[i + 2] = static_cast<int64_t>(pads[i]);
            pads_begin[i] -= pads[i];
            pads_end[i] -= pads[i];
            sliced[i] = true;
            pad_explicitly = true;
        }
    }
    std::vector<int64_t> slice_axes, window_offsets, start_offsets, slice_steps;
    for (size_t i = 0; i < kernel.size(); ++i) {
        if (sliced[i]) {
            const auto k = static_cast<int64_t>(kernel[i]);
            const auto s = static_cast<int64_t>(strides[i]);
            const auto p = static_cast<int64_t>(pads[i]);
            // PyTorch has ceil((L + 2p - k) / s) + 1 windows, but none starting in the right padding.
            slice_axes.push_back(static_cast<int64_t>(i + 2));
            window_offsets.push_back(2 * p + 2 * s - k - 1);
            start_offsets.push_back(p + s - 1);
            slice_steps.push_back(s);
        }
    }
    std::shared_ptr<v0::Constant> slice_axes_const;
    Output<Node> window_count;
    if (!slice_axes.empty()) {
        slice_axes_const = v0::Constant::create(element::i64, Shape{slice_axes.size()}, slice_axes);
        auto shape_of = context.mark_node(std::make_shared<v3::ShapeOf>(input));
        auto zero = v0::Constant::create(element::i64, Shape{}, {0});
        auto lengths = context.mark_node(std::make_shared<v8::Gather>(shape_of, slice_axes_const, zero));
        auto steps_const = v0::Constant::create(element::i64, Shape{slice_steps.size()}, slice_steps);
        const auto count = [&](const std::vector<int64_t>& offsets) {
            auto offsets_const = v0::Constant::create(element::i64, Shape{offsets.size()}, offsets);
            auto shifted_lengths = context.mark_node(std::make_shared<v1::Add>(lengths, offsets_const));
            return context.mark_node(std::make_shared<v1::Divide>(shifted_lengths, steps_const));
        };
        window_count = context.mark_node(std::make_shared<v1::Minimum>(count(window_offsets), count(start_offsets)));
    }
    if (pad_explicitly) {
        auto pads_const = v0::Constant::create(element::i64, Shape{explicit_pads.size()}, explicit_pads);
        input = context.mark_node(std::make_shared<v12::Pad>(input, pads_const, pads_const, PadMode::CONSTANT));
    }
    Output<Node> res;
    if (single_window) {
        std::vector<int64_t> spatial_axes(kernel.size());
        std::iota(spatial_axes.begin(), spatial_axes.end(), 2);
        auto axes = v0::Constant::create(element::i64, Shape{spatial_axes.size()}, spatial_axes);
        res = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, true));
    } else {
        res = context.mark_node(std::make_shared<v14::AvgPool>(input,
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               kernel,
                                                               exclude_pad || extended,
                                                               rounding_type));
    }
    if (slice_axes_const) {
        const auto& axes_shape = slice_axes_const->get_shape();
        auto zeros = v0::Constant::create(element::i64, axes_shape, {0});
        auto ones = v0::Constant::create(element::i64, axes_shape, {1});
        res = context.mark_node(std::make_shared<v8::Slice>(res, zeros, window_count, ones, slice_axes_const));
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
