#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/squeeze.hpp"
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

// Helper function to parse dims input - now returns AxesInfo for dynamic rank support
void parse_tensordot_dims(
    const NodeContext& ctx,
    Output<Node> dims,
    Output<Node> a,
    Output<Node> b,
    AxesInfo& a_axes,
    AxesInfo& b_axes) {

    auto dims_const = ctx.get_values_from_const_input(2);

    auto a_rank_ps = a.get_partial_shape().rank();
    auto b_rank_ps = b.get_partial_shape().rank();

    // ---- dims = int ----
    if (dims_const.is<int64_t>()) {
        int64_t k = dims_const.as<int64_t>();

        // For the int case, we need to handle both static and dynamic rank
        if (a_rank_ps.is_static() && b_rank_ps.is_static()) {
            // Static rank case - use original logic
            int64_t a_rank = a_rank_ps.get_length();
            int64_t b_rank = b_rank_ps.get_length();

            FRONT_END_GENERAL_CHECK(k >= 0 && k <= a_rank && k <= b_rank,
                "aten::tensordot: invalid dims value");

            std::vector<int64_t> a_axes_vec, b_axes_vec;
            for (int64_t i = a_rank - k; i < a_rank; ++i)
                a_axes_vec.push_back(i);

            for (int64_t i = 0; i < k; ++i)
                b_axes_vec.push_back(i);

            a_axes = AxesInfo(a_axes_vec);
            b_axes = AxesInfo(b_axes_vec);
        } else {
            // Dynamic rank case - compute axes at runtime
            FRONT_END_GENERAL_CHECK(k >= 0, "aten::tensordot: dims must be non-negative");
            
            // Get ranks dynamically
            Output<Node> a_rank, b_rank;
            std::tie(std::ignore, a_rank) = get_shape_rank(ctx, a, true);
            std::tie(std::ignore, b_rank) = get_shape_rank(ctx, b, true);

            // a_axes = range(a_rank - k, a_rank) = range(a_rank - k, k)
            auto const_k = v0::Constant::create(element::i32, Shape{}, {k});
            auto a_start = ctx.mark_node(std::make_shared<v1::Subtract>(a_rank, const_k));
            auto a_axes_dynamic = create_range(ctx, a_start, const_k);

            // b_axes = range(0, k)
            auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto b_axes_dynamic = create_range(ctx, const_0, const_k);

            a_axes = AxesInfo(a_axes_dynamic);
            b_axes = AxesInfo(b_axes_dynamic);
        }

        return;
    }

    // ---- dims = ([...], [...]) ----
    FRONT_END_GENERAL_CHECK(dims_const.is<std::vector<std::vector<int64_t>>>(),
        "aten::tensordot: dims must be int or tuple of lists");

    auto axes = dims_const.as<std::vector<std::vector<int64_t>>>();
    FRONT_END_GENERAL_CHECK(axes.size() == 2,
        "aten::tensordot: dims tuple must have two elements");

    auto a_axes_vec = axes[0];
    auto b_axes_vec = axes[1];

    FRONT_END_GENERAL_CHECK(a_axes_vec.size() == b_axes_vec.size(),
        "aten::tensordot: mismatched contraction axes sizes");

    // For tuple case, we have explicit axes, so we can normalize them
    // If rank is static, normalize negative axes
    if (a_rank_ps.is_static()) {
        int64_t a_rank = a_rank_ps.get_length();
        std::set<int64_t> seen_a;

        for (auto& ax : a_axes_vec) {
            if (ax < 0) ax += a_rank;
            FRONT_END_GENERAL_CHECK(ax >= 0 && ax < a_rank,
                "aten::tensordot: axis out of range for a");
            FRONT_END_GENERAL_CHECK(seen_a.insert(ax).second,
                "aten::tensordot: duplicate axis in a");
        }
    } else {
        // For dynamic rank, use normalize_axis utility
        Output<Node> a_rank;
        std::tie(std::ignore, a_rank) = get_shape_rank(ctx, a, true);
        
        for (auto& ax : a_axes_vec) {
            auto ax_node = v0::Constant::create(element::i32, Shape{}, {ax});
            auto normalized = normalize_axis(ctx, ax_node, a_rank);
            // We still store as vector for tuple case since axes are explicit
        }
    }

    if (b_rank_ps.is_static()) {
        int64_t b_rank = b_rank_ps.get_length();
        std::set<int64_t> seen_b;

        for (auto& ax : b_axes_vec) {
            if (ax < 0) ax += b_rank;
            FRONT_END_GENERAL_CHECK(ax >= 0 && ax < b_rank,
                "aten::tensordot: axis out of range for b");
            FRONT_END_GENERAL_CHECK(seen_b.insert(ax).second,
                "aten::tensordot: duplicate axis in b");
        }
    } else {
        Output<Node> b_rank;
        std::tie(std::ignore, b_rank) = get_shape_rank(ctx, b, true);
        
        for (auto& ax : b_axes_vec) {
            auto ax_node = v0::Constant::create(element::i32, Shape{}, {ax});
            auto normalized = normalize_axis(ctx, ax_node, b_rank);
        }
    }

    a_axes = AxesInfo(a_axes_vec);
    b_axes = AxesInfo(b_axes_vec);
}

// Helper function to reshape to 2D
Output<Node> reshape_to_2d(
    const NodeContext& ctx,
    Output<Node> input,
    const AxesInfo& left_axes,
    const AxesInfo& right_axes) {

    using namespace ov::op;

    auto shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto gather_dims = [&](const AxesInfo& axes_info) -> Output<Node> {
        if (axes_info.is_static) {
            auto idx = v0::Constant::create(element::i64, Shape{axes_info.static_axes.size()}, axes_info.static_axes);
            return ctx.mark_node(std::make_shared<v8::Gather>(
                shape, idx, v0::Constant::create(element::i64, Shape{}, {0})));
        } else {
            return ctx.mark_node(std::make_shared<v8::Gather>(
                shape, axes_info.dynamic_axes, v0::Constant::create(element::i64, Shape{}, {0})));
        }
    };

    auto left_dims = gather_dims(left_axes);
    auto right_dims = gather_dims(right_axes);

    auto M = ctx.mark_node(std::make_shared<v1::ReduceProd>(
        left_dims, v0::Constant::create(element::i64, Shape{}, {0}), true));

    auto K = ctx.mark_node(std::make_shared<v1::ReduceProd>(
        right_dims, v0::Constant::create(element::i64, Shape{}, {0}), true));

    auto new_shape = ctx.mark_node(std::make_shared<v0::Concat>(
        OutputVector{M, K}, 0));

    return ctx.mark_node(
        std::make_shared<v1::Reshape>(input, new_shape, false));
}

// Helper function to get complement axes
AxesInfo complement_axes(
    const NodeContext& ctx,
    const Output<Node>& tensor,
    const AxesInfo& axes) {

    auto rank_ps = tensor.get_partial_shape().rank();
    
    if (axes.is_static && rank_ps.is_static()) {
        // Static case - use original logic
        int64_t r = rank_ps.get_length();
        std::vector<bool> used(r, false);

        for (auto ax : axes.static_axes)
            used[ax] = true;

        std::vector<int64_t> result;
        for (int64_t i = 0; i < r; ++i)
            if (!used[i])
                result.push_back(i);

        return AxesInfo(result);
    } else {
        // Dynamic case - compute complement at runtime
        Output<Node> rank;
        std::tie(std::ignore, rank) = get_shape_rank(ctx, tensor, true);
        
        // Create range [0, 1, ..., rank-1]
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});
        auto all_axes = ctx.mark_node(std::make_shared<v4::Range>(const_0, rank, const_1, element::i32));
        
        if (axes.is_static) {
            // Create a mask by scattering -1 at positions to exclude
            auto minus_one_shape = ctx.mark_node(v0::Constant::create(element::i32, Shape{axes.static_axes.size()}, 
                std::vector<int32_t>(axes.static_axes.size(), -1)));
            auto indices = v0::Constant::create(element::i32, Shape{axes.static_axes.size()}, axes.static_axes);
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto marked = ctx.mark_node(std::make_shared<v3::ScatterElementsUpdate>(
                all_axes, indices, minus_one_shape, axis_0));
            
            // Filter out -1 values using NonZero + Gather pattern
            // For simplicity, we'll use a different approach: gather only non-excluded indices
            // Since this is complex, for now we'll use the approach of building the complement directly
            
            // Actually, let's use a simpler approach: build the complement vector statically if possible
            // This handles the case where axes are static but rank is dynamic
            
            // We need a more sophisticated approach here
            // For now, let's assume if axes are static, rank should also be static
            // This is reasonable for the tuple case
            FRONT_END_GENERAL_CHECK(rank_ps.is_static(),
                "aten::tensordot: static axes with dynamic rank not yet supported");
                
            // Fallback to static computation
            int64_t r = rank_ps.get_length();
            std::vector<bool> used(r, false);
            for (auto ax : axes.static_axes)
                used[ax] = true;

            std::vector<int64_t> result;
            for (int64_t i = 0; i < r; ++i)
                if (!used[i])
                    result.push_back(i);

            return AxesInfo(result);
        } else {
            // Both axes and rank are dynamic
            // This is the most complex case
            // We need to filter all_axes to exclude axes.dynamic_axes
            
            // Create a mask tensor of size rank, initialized to 1
            auto ones = ctx.mark_node(std::make_shared<v3::Broadcast>(
                v0::Constant::create(element::i32, Shape{}, {1}),
                rank));
            
            // Scatter 0 at positions in axes.dynamic_axes
            auto zeros_shape_node = ctx.mark_node(std::make_shared<v3::ShapeOf>(axes.dynamic_axes, element::i32));
            auto zeros = ctx.mark_node(std::make_shared<v3::Broadcast>(
                v0::Constant::create(element::i32, Shape{}, {0}),
                zeros_shape_node));
            auto axis_0 = v0::Constant::create(element::i32, Shape{}, {0});
            auto mask = ctx.mark_node(std::make_shared<v3::ScatterElementsUpdate>(
                ones, axes.dynamic_axes, zeros, axis_0));
            
            // Now we need to gather indices where mask == 1
            // This requires NonZero operation
            auto not_equal = ctx.mark_node(std::make_shared<v1::NotEqual>(
                mask, v0::Constant::create(element::i32, Shape{}, {0})));
            auto complement_indices = ctx.mark_node(std::make_shared<v3::NonZero>(not_equal, element::i32));
            
            // NonZero returns shape [1, num_true], we need to squeeze it
            auto squeeze_axis = v0::Constant::create(element::i64, Shape{1}, {0});
            auto result = ctx.mark_node(std::make_shared<v0::Squeeze>(complement_indices, squeeze_axis));
            
            return AxesInfo(result);
        }
    }
}

// Helper function to concatenate shapes
Output<Node> concat_shapes(
    const NodeContext& ctx,
    Output<Node> a,
    Output<Node> b,
    const AxesInfo& a_axes,
    const AxesInfo& b_axes) {

    using namespace ov::op;

    auto a_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(a));
    auto b_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(b));

    auto gather = [&](Output<Node> shape, const AxesInfo& axes_info) -> Output<Node> {
        if (axes_info.is_static) {
            return ctx.mark_node(std::make_shared<v8::Gather>(
                shape, v0::Constant::create(element::i64, Shape{axes_info.static_axes.size()}, axes_info.static_axes),
                v0::Constant::create(element::i64, Shape{}, {0})));
        } else {
            return ctx.mark_node(std::make_shared<v8::Gather>(
                shape, axes_info.dynamic_axes, v0::Constant::create(element::i64, Shape{}, {0})));
        }
    };

    auto a_dims = gather(a_shape, a_axes);
    auto b_dims = gather(b_shape, b_axes);

    return ctx.mark_node(
        std::make_shared<v0::Concat>(OutputVector{a_dims, b_dims}, 0));
}

// Helper function to create permutation for transpose
Output<Node> create_permutation(const NodeContext& ctx, const AxesInfo& first, const AxesInfo& second) {
    if (first.is_static && second.is_static) {
        // Both static - concatenate vectors
        std::vector<int64_t> perm = first.static_axes;
        perm.insert(perm.end(), second.static_axes.begin(), second.static_axes.end());
        return v0::Constant::create(element::i64, Shape{perm.size()}, perm);
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

    auto a = context.get_input(0);
    auto b = context.get_input(1);

    FRONT_END_GENERAL_CHECK(ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(2).get_node_shared_ptr()),
        "aten::tensordot: dims must be constant");

    AxesInfo a_axes, b_axes;
    parse_tensordot_dims(context, context.get_input(2), a, b, a_axes, b_axes);

    auto a_remain = complement_axes(context, a, a_axes);
    auto b_remain = complement_axes(context, b, b_axes);

    auto a_perm = create_permutation(context, a_remain, a_axes);
    auto b_perm = create_permutation(context, b_axes, b_remain);

    auto a_t = context.mark_node(std::make_shared<v1::Transpose>(a, a_perm));
    auto b_t = context.mark_node(std::make_shared<v1::Transpose>(b, b_perm));

    auto a_2d = reshape_to_2d(context, a_t, a_remain, a_axes);
    auto b_2d = reshape_to_2d(context, b_t, b_axes, b_remain);

    auto mm = context.mark_node(
        std::make_shared<v0::MatMul>(a_2d, b_2d, false, false));

    auto out_shape = concat_shapes(context, a, b, a_remain, b_remain);

    auto out = context.mark_node(
        std::make_shared<v1::Reshape>(mm, out_shape, false));

    return {out};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
