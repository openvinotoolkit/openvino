// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
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

namespace {
// PyTorch only accepts an avg_pool kernel larger than the (unpadded) spatial input when the pooling
// degenerates to a single output element per oversized axis - a global average pool (with a smaller
// stride and an oversized kernel PyTorch itself raises "Output size is too small"). ov::op::AvgPool
// instead requires the kernel to fit the padded data shape. The traced input shape is available even
// when the runtime graph keeps the spatial dimensions dynamic, so it is used to recognise an oversized
// kernel. Returns, per trailing spatial axis, whether the kernel exceeds the (static) traced extent.
std::vector<bool> oversized_spatial_axes(const PartialShape& traced_shape, const Shape& kernel, int dims) {
    std::vector<bool> oversized(static_cast<size_t>(dims), false);
    if (!traced_shape.rank().is_static() || traced_shape.rank().get_length() < dims) {
        return oversized;
    }
    const auto rank = traced_shape.rank().get_length();
    for (int i = 0; i < dims; ++i) {
        const auto& spatial_dim = traced_shape[rank - dims + i];
        oversized[static_cast<size_t>(i)] =
            spatial_dim.is_static() && static_cast<int64_t>(kernel[i]) > spatial_dim.get_length();
    }
    return oversized;
}
}  // namespace

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

    // An oversized kernel on an unpadded axis is a global average over that axis. Reduce it before
    // pooling so the graph remains valid for dynamic spatial dimensions. For padded axes, keep kernels
    // that fit the padded extent unchanged; only clamp kernels that exceed the entire padded extent.
    const auto traced_shape = context.get_decoder()->get_input_complete_shape(0);
    const auto oversized = oversized_spatial_axes(traced_shape, kernel, dims);
    std::vector<int64_t> reduce_axes;
    const auto traced_rank = traced_shape.rank().is_static() ? traced_shape.rank().get_length() : 0;
    for (int i = 0; i < dims; ++i) {
        const auto axis = static_cast<size_t>(i);
        if (oversized[axis] && pads[axis] == 0) {
            reduce_axes.push_back(-dims + i);
            kernel[axis] = 1;
            strides[axis] = 1;
            pads[axis] = 0;
        } else if (oversized[axis] && traced_rank >= dims) {
            const auto spatial_dim = traced_shape[traced_rank - dims + i].get_length();
            const auto padded_dim = spatial_dim + static_cast<int64_t>(pads[axis]) * 2;
            if (static_cast<int64_t>(kernel[axis]) > padded_dim) {
                kernel[axis] = static_cast<size_t>(padded_dim);
            }
        }
    }
    if (!reduce_axes.empty()) {
        auto axes = v0::Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
        input = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, true));
    }

    auto res = context.mark_node(
        std::make_shared<v14::AvgPool>(input, strides, pads, pads, kernel, !count_include_pad, rounding_type));

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
