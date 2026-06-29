// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
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
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

// Decompose a max_pool whose kernel_size is only known at runtime.
//
// OpenVINO's MaxPool takes the pooling window (kernel/strides/pads/dilations) as constructor
// attributes, so it cannot represent a kernel whose size is computed at runtime -- e.g. PyTorch's
// F.max_pool2d(x, kernel_size=[1, x.size(3)]). In practice such a kernel only appears as a
// "global" pool that spans the full extent of a spatial axis, which is exactly a ReduceMax over
// that axis (with keep_dims=True to preserve the pooled axis as size 1, matching MaxPool). Every
// other dynamic-kernel configuration is rejected with a clear message.
//
// elem_is_const / elem_const_val describe the kernel element of each of the `dims` spatial axes:
// a constant 1 means "window of 1" (identity along that axis), a non-constant element means
// "full-extent global pool" (reduce that axis). ReduceMax accepts arbitrary rank, so this works
// directly on the original input regardless of batch presence / dynamic rank, using negative axes.
OutputVector translate_max_pool_dynamic_kernel(const NodeContext& context,
                                               int dims,
                                               bool return_indices,
                                               const Output<Node>& input,
                                               const std::vector<bool>& elem_is_const,
                                               const std::vector<int64_t>& elem_const_val) {
    PYTORCH_OP_CONVERSION_CHECK(static_cast<int>(elem_is_const.size()) == dims,
                                "aten::max_pool",
                                dims,
                                "d: could not interpret the non-constant kernel_size (expected ",
                                dims,
                                " spatial entries).");
    PYTORCH_OP_CONVERSION_CHECK(!return_indices,
                                "aten::max_pool",
                                dims,
                                "d with a non-constant kernel_size and return_indices=True is not supported.");
    // The decomposition is exact only for a full-extent pool with the default window placement:
    // default stride (=kernel), zero padding, dilation 1, ceil_mode=False.
    PYTORCH_OP_CONVERSION_CHECK(context.input_is_none(2) || is_empty_list(context.get_input(2)),
                                "aten::max_pool",
                                dims,
                                "d with a non-constant kernel_size is only supported with the default stride "
                                "(stride=kernel_size).");
    if (!context.input_is_none(3)) {
        for (auto p : context.const_input<Shape>(3)) {
            PYTORCH_OP_CONVERSION_CHECK(p == 0,
                                        "aten::max_pool",
                                        dims,
                                        "d with a non-constant kernel_size is only supported with zero padding.");
        }
    }
    if (!context.input_is_none(4)) {
        for (auto d : context.const_input<Strides>(4)) {
            PYTORCH_OP_CONVERSION_CHECK(d == 1,
                                        "aten::max_pool",
                                        dims,
                                        "d with a non-constant kernel_size is only supported with dilation 1.");
        }
    }
    if (!context.input_is_none(5)) {
        PYTORCH_OP_CONVERSION_CHECK(!context.const_input<bool>(5),
                                    "aten::max_pool",
                                    dims,
                                    "d with a non-constant kernel_size is only supported with ceil_mode=False.");
    }

    // Decide, per spatial axis, whether it is reduced (full-extent) or left untouched (window 1).
    std::vector<int64_t> reduce_axes;
    for (int i = 0; i < dims; ++i) {
        const int64_t axis = static_cast<int64_t>(i) - dims;  // negative index of this spatial axis
        if (!elem_is_const[i]) {
            // Runtime kernel element: assume it spans the whole axis (global pool over that axis).
            reduce_axes.push_back(axis);
        } else if (elem_const_val[i] == 1) {
            // Window of 1 with default stride is identity along this axis -- nothing to reduce.
            continue;
        } else {
            // A static window > 1 next to a dynamic axis would be a genuine sliding-window pool,
            // which a ReduceMax cannot represent.
            PYTORCH_OP_CONVERSION_CHECK(false,
                                        "aten::max_pool",
                                        dims,
                                        "d with a non-constant kernel_size is only supported when every pooled axis "
                                        "spans its full extent (the kernel along statically-sized axes must be 1).");
        }
    }
    PYTORCH_OP_CONVERSION_CHECK(!reduce_axes.empty(),
                                "aten::max_pool",
                                dims,
                                "d: a non-constant kernel_size that pools no axis is unexpected.");

    auto axes = context.mark_node(v0::Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes));
    auto res = context.mark_node(std::make_shared<v1::ReduceMax>(input, axes, /*keep_dims=*/true));
    return {std::move(res)};
}

}  // namespace

OutputVector translate_max_pool_base(const NodeContext& context, int dims, bool return_indices) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);

    // The OpenVINO MaxPool kernel is a constructor attribute and must be known at conversion time.
    // PyTorch sometimes builds kernel_size from runtime values (e.g. kernel_size=[1, x.size(3)]);
    // probe the kernel list element-by-element without throwing, and if any element is not a
    // constant fall back to a ReduceMax-based decomposition (see translate_max_pool_dynamic_kernel).
    std::vector<bool> elem_is_const;
    std::vector<int64_t> elem_const_val;
    bool kernel_is_static = true;
    for (const auto& e : get_list_as_outputs(context.get_input(1))) {
        if (const auto c = ov::util::get_constant_from_source(e)) {
            for (auto val : c->cast_vector<int64_t>()) {
                elem_is_const.push_back(true);
                elem_const_val.push_back(val);
            }
        } else {
            elem_is_const.push_back(false);
            elem_const_val.push_back(-1);
            kernel_is_static = false;
        }
    }
    // A scalar kernel_size (e.g. kernel_size=k) applies to every spatial axis in PyTorch.
    if (elem_is_const.size() == 1 && dims > 1) {
        elem_is_const.assign(dims, elem_is_const[0]);
        elem_const_val.assign(dims, elem_const_val[0]);
    }
    if (!kernel_is_static) {
        return translate_max_pool_dynamic_kernel(context, dims, return_indices, input, elem_is_const, elem_const_val);
    }

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
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    }
    auto dilations = Strides(dims, 1);
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
    }

    auto res = context.mark_node(std::make_shared<v14::MaxPool>(input,
                                                                strides,
                                                                dilations,
                                                                pads,
                                                                pads,
                                                                kernel,
                                                                rounding_type,
                                                                PadType::EXPLICIT,
                                                                element::i64,
                                                                2));
    if (is_static) {
        if (no_batch_dim) {
            if (return_indices) {
                auto out1 = res->output(0);
                auto out2 = res->output(1);
                out1 = context.mark_node(std::make_shared<v0::Squeeze>(out1, const_0));
                out2 = context.mark_node(std::make_shared<v0::Squeeze>(out2, const_0));
                return {std::move(out1), std::move(out2)};
            } else {
                res = context.mark_node(std::make_shared<v0::Squeeze>(res, const_0));
                return {res};
            }
        } else {
            if (return_indices) {
                auto out1 = res->output(0);
                auto out2 = res->output(1);
                return {std::move(out1), std::move(out2)};
            } else {
                return {res};
            }
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
        if (return_indices) {
            auto out1 = res->output(0);
            auto out2 = res->output(1);
            out1 = context.mark_node(std::make_shared<v1::Reshape>(out1, concat_shape, true));
            out2 = context.mark_node(std::make_shared<v1::Reshape>(out2, concat_shape, true));
            return {std::move(out1), std::move(out2)};
        } else {
            res = context.mark_node(std::make_shared<v1::Reshape>(res, concat_shape, true));
            return {res};
        }
    }
};

OutputVector translate_max_pool1d(const NodeContext& context) {
    return translate_max_pool_base(context, 1, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d(const NodeContext& context) {
    return translate_max_pool_base(context, 2, context.get_output_size() == 2);
};

OutputVector translate_max_pool3d(const NodeContext& context) {
    return translate_max_pool_base(context, 3, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 2, true);
    return {context.mark_node(make_list_construct(output))};
};

OutputVector translate_max_pool3d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 3, true);
    return {context.mark_node(make_list_construct(output))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
