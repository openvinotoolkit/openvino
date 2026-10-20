// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {
// PyTorch accepts a kernel larger than an unpadded 1D/2D spatial axis with ceil_mode if
// that axis produces just one output. AvgPool requires the kernel to fit, so use ReduceMean
// for that axis only when it is short at runtime. Keep the other axes' fixed windows intact.
Output<Node> pool_with_short_axes(const NodeContext& context,
                                  const Output<Node>& input,
                                  const Strides& strides,
                                  const Shape& pads,
                                  const Shape& kernel,
                                  bool exclude_pad,
                                  RoundingType rounding_type,
                                  size_t axis) {
    if (axis == kernel.size()) {
        return context.mark_node(
            std::make_shared<v14::AvgPool>(input, strides, pads, pads, kernel, exclude_pad, rounding_type));
    }

    // Explicitly padded axes retain AvgPool's normal kernel-fit and divisor semantics. For stride 1,
    // no integer extent can be both shorter than the kernel and large enough for a ceil-mode window.
    if (pads[axis] != 0 || kernel[axis] == 1 || strides[axis] == 1) {
        return pool_with_short_axes(context, input, strides, pads, kernel, exclude_pad, rounding_type, axis + 1);
    }

    const auto shape = input.get_partial_shape();
    if (shape.rank().is_static()) {
        const auto& dim = shape[axis + 2];
        if (dim.get_min_length() >= static_cast<int64_t>(kernel[axis])) {
            return pool_with_short_axes(context, input, strides, pads, kernel, exclude_pad, rounding_type, axis + 1);
        }
        if (dim.is_static()) {
            // ceil((length - kernel) / stride) + 1 must be positive for a short axis.
            if (strides[axis] < kernel[axis] &&
                dim.get_length() <= static_cast<int64_t>(kernel[axis] - strides[axis])) {
                return pool_with_short_axes(context,
                                            input,
                                            strides,
                                            pads,
                                            kernel,
                                            exclude_pad,
                                            rounding_type,
                                            axis + 1);
            }
            auto axes =
                context.mark_node(v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(axis + 2)}));
            auto reduced = context.mark_node(std::make_shared<v1::ReduceMean>(input, axes, true));
            auto reduced_kernel = kernel;
            auto reduced_strides = strides;
            reduced_kernel[axis] = 1;
            reduced_strides[axis] = 1;
            return pool_with_short_axes(context,
                                        reduced,
                                        reduced_strides,
                                        pads,
                                        reduced_kernel,
                                        exclude_pad,
                                        rounding_type,
                                        axis + 1);
        }
    }

    auto shape_of = context.mark_node(std::make_shared<v3::ShapeOf>(input));
    auto index = context.mark_node(v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(axis + 2)}));
    auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto extent = context.mark_node(std::make_shared<v8::Gather>(shape_of, index, zero));
    auto size = context.mark_node(v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(kernel[axis])}));
    Output<Node> condition = context.mark_node(std::make_shared<v1::Less>(extent, size));
    if (strides[axis] < kernel[axis]) {
        auto min_valid = context.mark_node(
            v0::Constant::create(element::i64, Shape{}, {static_cast<int64_t>(kernel[axis] - strides[axis])}));
        auto has_window = context.mark_node(std::make_shared<v1::Greater>(extent, min_valid));
        condition = context.mark_node(std::make_shared<v1::LogicalAnd>(condition, has_window));
    }

    // Keep body dimensions dynamic so both branches can be constructed independently.
    const auto body_shape = PartialShape::dynamic(kernel.size() + 2);
    auto short_input = std::make_shared<v0::Parameter>(input.get_element_type(), body_shape);
    auto long_input = std::make_shared<v0::Parameter>(input.get_element_type(), body_shape);
    auto axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(axis + 2)}));
    auto reduced = context.mark_node(std::make_shared<v1::ReduceMean>(short_input, axes, true));
    auto reduced_kernel = kernel;
    auto reduced_strides = strides;
    reduced_kernel[axis] = 1;
    reduced_strides[axis] = 1;
    auto short_output = pool_with_short_axes(context,
                                             reduced,
                                             reduced_strides,
                                             pads,
                                             reduced_kernel,
                                             exclude_pad,
                                             rounding_type,
                                             axis + 1);
    auto long_output =
        pool_with_short_axes(context, long_input, strides, pads, kernel, exclude_pad, rounding_type, axis + 1);
    auto short_result = std::make_shared<v0::Result>(short_output);
    auto long_result = std::make_shared<v0::Result>(long_output);
    auto if_node = std::make_shared<v8::If>(condition);
    context.mark_node(if_node);
    if_node->set_then_body(std::make_shared<Model>(ResultVector{short_result}, ParameterVector{short_input}));
    if_node->set_else_body(std::make_shared<Model>(ResultVector{long_result}, ParameterVector{long_input}));
    if_node->set_input(input, short_input, long_input);
    return if_node->set_output(short_result, long_result);
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
    if (pads.empty()) {
        pads.resize(kernel.size(), 0);
    }
    // The short-axis helper indexes these attributes before AvgPool can validate them.
    PYTORCH_OP_CONVERSION_CHECK(kernel.size() == static_cast<size_t>(dims),
                                "Expected kernel size to match the number of spatial dimensions: ",
                                dims);
    PYTORCH_OP_CONVERSION_CHECK(strides.size() == kernel.size(),
                                "Expected strides size to match kernel size: ",
                                kernel.size());
    PYTORCH_OP_CONVERSION_CHECK(pads.size() == kernel.size(),
                                "Expected padding size to match kernel size: ",
                                kernel.size());
    ov::op::RoundingType rounding_type = ov::op::RoundingType::FLOOR;
    if (!(context.input_is_none(4))) {
        rounding_type = context.const_input<bool>(4) ? ov::op::RoundingType::CEIL_TORCH : ov::op::RoundingType::FLOOR;
    }
    if (!(context.input_is_none(5))) {
        count_include_pad = context.const_input<bool>(5);
    }
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(6),
                                "Translation for aten::avg_pool2d do not support divisor_override input.");

    Output<Node> res;
    const auto input_type = input.get_element_type();
    const bool use_short_axis_pooling = dims < 3 && rounding_type == RoundingType::CEIL_TORCH;
    // PyTorch rejects oversized kernels for avg_pool3d, and without ceil_mode a short
    // unpadded axis cannot produce a valid output. Leave those cases on the usual path.
    if (use_short_axis_pooling) {
        auto pool_input = input;
        if (input_type.is_integral()) {
            // Keep fractional intermediate averages until every spatial axis has been pooled.
            pool_input = context.mark_node(std::make_shared<v0::Convert>(input, element::f64));
        }
        res = pool_with_short_axes(context, pool_input, strides, pads, kernel, !count_include_pad, rounding_type, 0);
        if (input_type.is_integral()) {
            res = context.mark_node(std::make_shared<v0::Convert>(res, input_type));
        }
    } else {
        res = context.mark_node(
            std::make_shared<v14::AvgPool>(input, strides, pads, pads, kernel, !count_include_pad, rounding_type));
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
