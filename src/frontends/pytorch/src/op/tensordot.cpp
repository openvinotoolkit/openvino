// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

// Helper structure to hold axes information (can be static or dynamic)
struct AxesInfo {
    bool is_static;
    std::vector<int64_t> static_axes;
    Output<Node> dynamic_axes;

    AxesInfo() : is_static(false) {}
    AxesInfo(const std::vector<int64_t>& axes) : is_static(true), static_axes(axes) {}
    AxesInfo(Output<Node> axes) : is_static(false), dynamic_axes(axes) {}
};

// Helper function to create a range [start, start+1, ..., start+count-1]
Output<Node> create_range(const NodeContext& ctx, Output<Node> start, Output<Node> count) {
    auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});
    auto end = ctx.mark_node(std::make_shared<v1::Add>(start, count));
    return ctx.mark_node(std::make_shared<v4::Range>(start, end, const_1, element::i32));
}

// Helper function to parse dims input - supports int and tuple-of-lists dims.
void parse_tensordot_dims(const NodeContext& ctx, Output<Node> a, Output<Node> b, AxesInfo& a_axes, AxesInfo& b_axes) {
    auto a_rank_ps = a.get_partial_shape().rank();
    auto b_rank_ps = b.get_partial_shape().rank();

    // Helper to normalize and validate a compile-time axes vector.
    auto validate_axes_vec = [](std::vector<int64_t>& vec, const Dimension& rank_ps, const std::string& tensor_name) {
        if (rank_ps.is_static()) {
            int64_t rank = rank_ps.get_length();
            std::set<int64_t> seen;
            for (auto& ax : vec) {
                if (ax < 0)
                    ax += rank;
                FRONT_END_GENERAL_CHECK(ax >= 0 && ax < rank,
                                        std::string("aten::tensordot: axis out of range for ") + tensor_name);
                FRONT_END_GENERAL_CHECK(seen.insert(ax).second,
                                        std::string("aten::tensordot: duplicate axis in ") + tensor_name);
            }
        } else {
            // Dynamic rank: cannot normalize negative axes at graph-build time.
            for (const auto& ax : vec) {
                FRONT_END_GENERAL_CHECK(
                    ax >= 0,
                    std::string("aten::tensordot: negative axes require static rank for tensor ") + tensor_name);
            }
        }
    };

    // ---- dims = ([...], [...]) ----
    // By the time a translator runs, all producer nodes have already been translated.
    // prim::ListConstruct is translated to v0::Concat (usually constant-folded to a
    // v0::Constant) by translate_list_construct before tensordot is reached.  This means
    // a tuple-of-two-lists dims arrives here as a 2D Constant of shape {2, num_axes}:
    //   row 0 = a_axes, row 1 = b_axes.
    // Detecting the FrameworkNode via cast_fw_node would always miss because the node
    // is no longer a prim::ListConstruct FrameworkNode at this point.
    auto dims_input = ctx.get_input_from_visible_context(2);
    if (auto dims_2d = ov::util::get_constant_from_source(dims_input)) {
        const auto& shape = dims_2d->get_shape();
        if (shape.size() == 2 && shape[0] == 2) {
            // shape is {2, num_axes}: first row = a_axes, second row = b_axes.
            size_t num_axes = shape[1];
            auto flat = dims_2d->cast_vector<int64_t>();

            std::vector<int64_t> a_axes_vec(flat.begin(), flat.begin() + num_axes);
            std::vector<int64_t> b_axes_vec(flat.begin() + num_axes, flat.end());

            FRONT_END_GENERAL_CHECK(a_axes_vec.size() == b_axes_vec.size(),
                                    "aten::tensordot: mismatched contraction axes sizes");

            validate_axes_vec(a_axes_vec, a_rank_ps, "a");
            validate_axes_vec(b_axes_vec, b_rank_ps, "b");

            a_axes = AxesInfo(a_axes_vec);
            b_axes = AxesInfo(b_axes_vec);
            return;
        }
    }

    // ---- dims = int ----
    // get_values_from_const_input uses get_constant_from_source internally and accepts
    // constant-foldable subgraphs, not only bare v0::Constant nodes.
    auto dims_const = ctx.get_values_from_const_input(2);
    FRONT_END_GENERAL_CHECK(dims_const.is<int64_t>(), "aten::tensordot: dims must be int or tuple of lists");
    int64_t k = dims_const.as<int64_t>();

    if (a_rank_ps.is_static() && b_rank_ps.is_static()) {
        int64_t a_rank = a_rank_ps.get_length();
        int64_t b_rank = b_rank_ps.get_length();

        FRONT_END_GENERAL_CHECK(k >= 0 && k <= a_rank && k <= b_rank, "aten::tensordot: invalid dims value");

        std::vector<int64_t> a_axes_vec, b_axes_vec;
        for (int64_t i = a_rank - k; i < a_rank; ++i)
            a_axes_vec.push_back(i);
        for (int64_t i = 0; i < k; ++i)
            b_axes_vec.push_back(i);

        a_axes = AxesInfo(a_axes_vec);
        b_axes = AxesInfo(b_axes_vec);
    } else {
        // Dynamic rank: compute contraction axes at runtime via Range ops.
        FRONT_END_GENERAL_CHECK(k >= 0, "aten::tensordot: dims must be non-negative");

        // Validate against whichever rank(s) are statically known, even when the
        // other is dynamic.  Without this, e.g. a static-rank-2 tensor with k=3
        // would produce a_start = -1 and generate invalid axis indices at runtime.
        if (a_rank_ps.is_static()) {
            FRONT_END_GENERAL_CHECK(k <= a_rank_ps.get_length(),
                                    "aten::tensordot: dims value exceeds rank of first tensor");
        }
        if (b_rank_ps.is_static()) {
            FRONT_END_GENERAL_CHECK(k <= b_rank_ps.get_length(),
                                    "aten::tensordot: dims value exceeds rank of second tensor");
        }

        Output<Node> a_rank;
        std::tie(std::ignore, a_rank) = get_shape_rank(ctx, a, true);

        auto const_k = v0::Constant::create(element::i32, Shape{}, {k});
        auto a_start = ctx.mark_node(std::make_shared<v1::Subtract>(a_rank, const_k));
        auto a_axes_dynamic = create_range(ctx, a_start, const_k);

        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto b_axes_dynamic = create_range(ctx, const_0, const_k);

        a_axes = AxesInfo(a_axes_dynamic);
        b_axes = AxesInfo(b_axes_dynamic);
    }
}

// Helper function to reshape to 2D.
// shape_source: original (pre-transpose) tensor whose axis sizes match left_axes/right_axes.
// input:        transposed tensor to actually reshape.
Output<Node> reshape_to_2d(const NodeContext& ctx,
                           Output<Node> shape_source,
                           Output<Node> input,
                           const AxesInfo& left_axes,
                           const AxesInfo& right_axes) {
    // Gather dimension sizes from the original tensor so original axis indices are valid.
    auto shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(shape_source));

    auto gather_dims = [&](const AxesInfo& axes_info) -> Output<Node> {
        if (axes_info.is_static) {
            auto idx = v0::Constant::create(element::i64, Shape{axes_info.static_axes.size()}, axes_info.static_axes);
            return ctx.mark_node(
                std::make_shared<v8::Gather>(shape, idx, v0::Constant::create(element::i64, Shape{}, {0})));
        } else {
            return ctx.mark_node(std::make_shared<v8::Gather>(shape,
                                                              axes_info.dynamic_axes,
                                                              v0::Constant::create(element::i64, Shape{}, {0})));
        }
    };

    auto left_dims = gather_dims(left_axes);
    auto right_dims = gather_dims(right_axes);

    auto M = ctx.mark_node(
        std::make_shared<v1::ReduceProd>(left_dims, v0::Constant::create(element::i64, Shape{}, {0}), true));

    auto K = ctx.mark_node(
        std::make_shared<v1::ReduceProd>(right_dims, v0::Constant::create(element::i64, Shape{}, {0}), true));

    auto new_shape = ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{M, K}, 0));

    // Reshape the transposed tensor (not shape_source) into [M, K].
    return ctx.mark_node(std::make_shared<v1::Reshape>(input, new_shape, false));
}

// Helper function to get complement axes
AxesInfo complement_axes(const NodeContext& ctx, const Output<Node>& tensor, const AxesInfo& axes) {
    auto rank_ps = tensor.get_partial_shape().rank();

    // Static/static: compute entirely at graph-build time.
    if (axes.is_static && rank_ps.is_static()) {
        int64_t r = rank_ps.get_length();
        std::vector<bool> used(r, false);

        for (auto ax : axes.static_axes)
            used[ax] = true;

        std::vector<int64_t> result;
        for (int64_t i = 0; i < r; ++i)
            if (!used[i])
                result.push_back(i);

        return AxesInfo(result);
    }

    // Static axes require a known rank to compute the complement at build time.
    // (Tuple-dims always provides explicit compile-time indices, so the rank must
    // be statically known too.  Dynamic axes only arise from the int-dims path,
    // which computes them at runtime via Range.)
    FRONT_END_GENERAL_CHECK(!axes.is_static, "aten::tensordot: static axes with dynamic rank not yet supported");

    // Dynamic axes: compute complement at runtime with OV ops.
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(ctx, tensor, true);

    // Build range [0, 1, ..., rank-1] as the universe of all axis indices.
    auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});
    auto all_axes = ctx.mark_node(std::make_shared<v4::Range>(const_0, rank, const_1, element::i32));

    // Create a mask of ones with the same length as all_axes, then zero out
    // the positions listed in axes.dynamic_axes.
    auto all_axes_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(all_axes, element::i32));
    auto ones = ctx.mark_node(
        std::make_shared<v3::Broadcast>(v0::Constant::create(element::i32, Shape{}, {1}), all_axes_shape));

    auto zeros_shape_node = ctx.mark_node(std::make_shared<v3::ShapeOf>(axes.dynamic_axes, element::i32));
    auto zeros = ctx.mark_node(
        std::make_shared<v3::Broadcast>(v0::Constant::create(element::i32, Shape{}, {0}), zeros_shape_node));
    auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
    auto mask = ctx.mark_node(std::make_shared<v3::ScatterElementsUpdate>(ones, axes.dynamic_axes, zeros, axis_0));

    // Gather indices of surviving (non-zero) positions via NonZero.
    auto not_equal =
        ctx.mark_node(std::make_shared<v1::NotEqual>(mask, v0::Constant::create(element::i32, Shape{}, {0})));
    auto complement_indices = ctx.mark_node(std::make_shared<v3::NonZero>(not_equal, element::i32));

    // NonZero returns shape [1, num_true]; squeeze the leading dim to get a 1D index vector.
    auto squeeze_axis = v0::Constant::create(element::i64, Shape{1}, {0});
    auto result = ctx.mark_node(std::make_shared<v0::Squeeze>(complement_indices, squeeze_axis));

    return AxesInfo(result);
}

// Helper function to concatenate shapes
Output<Node> concat_shapes(const NodeContext& ctx,
                           Output<Node> a,
                           Output<Node> b,
                           const AxesInfo& a_axes,
                           const AxesInfo& b_axes) {
    auto a_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(a));
    auto b_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(b));

    auto gather = [&](Output<Node> shape, const AxesInfo& axes_info) -> Output<Node> {
        if (axes_info.is_static) {
            return ctx.mark_node(std::make_shared<v8::Gather>(
                shape,
                v0::Constant::create(element::i64, Shape{axes_info.static_axes.size()}, axes_info.static_axes),
                v0::Constant::create(element::i64, Shape{}, {0})));
        } else {
            return ctx.mark_node(std::make_shared<v8::Gather>(shape,
                                                              axes_info.dynamic_axes,
                                                              v0::Constant::create(element::i64, Shape{}, {0})));
        }
    };

    auto a_dims = gather(a_shape, a_axes);
    auto b_dims = gather(b_shape, b_axes);

    return ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{a_dims, b_dims}, 0));
}

// Helper function to create permutation for transpose
Output<Node> create_permutation(const NodeContext& ctx, const AxesInfo& first, const AxesInfo& second) {
    if (first.is_static && second.is_static) {
        // Both static - concatenate vectors.
        // Use i32 to match the element type produced by Range in the dynamic paths.
        std::vector<int64_t> perm = first.static_axes;
        perm.insert(perm.end(), second.static_axes.begin(), second.static_axes.end());
        return v0::Constant::create(element::i32, Shape{perm.size()}, perm);
    } else if (first.is_static && !second.is_static) {
        // First static, second dynamic
        auto first_const = v0::Constant::create(element::i32, Shape{first.static_axes.size()}, first.static_axes);
        return ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{first_const, second.dynamic_axes}, 0));
    } else if (!first.is_static && second.is_static) {
        // First dynamic, second static
        auto second_const = v0::Constant::create(element::i32, Shape{second.static_axes.size()}, second.static_axes);
        return ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{first.dynamic_axes, second_const}, 0));
    } else {
        // Both dynamic
        return ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{first.dynamic_axes, second.dynamic_axes}, 0));
    }
}

OutputVector translate_tensordot(const NodeContext& context) {
    num_inputs_check(context, 3, 4);

    Output<Node> a, b;
    std::tie(a, b) = get_inputs_with_promoted_types(context, 0, 1);

    AxesInfo a_axes, b_axes;
    parse_tensordot_dims(context, a, b, a_axes, b_axes);

    auto a_remain = complement_axes(context, a, a_axes);
    auto b_remain = complement_axes(context, b, b_axes);

    auto a_perm = create_permutation(context, a_remain, a_axes);
    auto b_perm = create_permutation(context, b_axes, b_remain);

    auto a_t = context.mark_node(std::make_shared<v1::Transpose>(a, a_perm));
    auto b_t = context.mark_node(std::make_shared<v1::Transpose>(b, b_perm));

    // reshape_to_2d must gather dim sizes from the *original* tensors (a, b), not from
    // a_t/b_t, because after transposing the axis indices no longer match their positions.
    auto a_2d = reshape_to_2d(context, a, a_t, a_remain, a_axes);
    auto b_2d = reshape_to_2d(context, b, b_t, b_axes, b_remain);

    auto mm = context.mark_node(std::make_shared<v0::MatMul>(a_2d, b_2d, false, false));

    auto out_shape = concat_shapes(context, a, b, a_remain, b_remain);

    auto out = context.mark_node(std::make_shared<v1::Reshape>(mm, out_shape, false));

    if (!context.input_is_none(3)) {
        out = context.mark_node(std::make_shared<v1::ConvertLike>(out, context.get_input(3)));
        context.mutate_input(3, out);
    }

    return {out};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
