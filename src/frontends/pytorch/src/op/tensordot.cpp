#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
//helper function to parse dims input
void parse_tensordot_dims(
    const NodeContext& ctx,
    Output<Node> dims,
    Output<Node> a,
    Output<Node> b,
    std::vector<int64_t>& a_axes,
    std::vector<int64_t>& b_axes) {

    auto dims_const = ctx.get_values_from_const_input(2);

    auto a_rank_ps = a.get_partial_shape().rank();
    auto b_rank_ps = b.get_partial_shape().rank();

    FRONT_END_GENERAL_CHECK(a_rank_ps.is_static() && b_rank_ps.is_static(),
        "aten::tensordot: static rank required");

    int64_t a_rank = a_rank_ps.get_length();
    int64_t b_rank = b_rank_ps.get_length();

    // ---- dims = int ----
    if (dims_const.is<int64_t>()) {
        int64_t k = dims_const.as<int64_t>();

        FRONT_END_GENERAL_CHECK(k >= 0 && k <= a_rank && k <= b_rank,
            "aten::tensordot: invalid dims value");

        for (int64_t i = a_rank - k; i < a_rank; ++i)
            a_axes.push_back(i);

        for (int64_t i = 0; i < k; ++i)
            b_axes.push_back(i);

        return;
    }

    // ---- dims = ([...], [...]) ----
    FRONT_END_GENERAL_CHECK(dims_const.is<std::vector<std::vector<int64_t>>>(),
        "aten::tensordot: dims must be int or tuple of lists");

    auto axes = dims_const.as<std::vector<std::vector<int64_t>>>();
    FRONT_END_GENERAL_CHECK(axes.size() == 2,
        "aten::tensordot: dims tuple must have two elements");

    a_axes = axes[0];
    b_axes = axes[1];

    FRONT_END_GENERAL_CHECK(a_axes.size() == b_axes.size(),
        "aten::tensordot: mismatched contraction axes sizes");

    // normalize negative axes + validate uniqueness
    std::set<int64_t> seen_a, seen_b;

    for (auto& ax : a_axes) {
        if (ax < 0) ax += a_rank;
        FRONT_END_GENERAL_CHECK(ax >= 0 && ax < a_rank,
            "aten::tensordot: axis out of range for a");
        FRONT_END_GENERAL_CHECK(seen_a.insert(ax).second,
            "aten::tensordot: duplicate axis in a");
    }

    for (auto& ax : b_axes) {
        if (ax < 0) ax += b_rank;
        FRONT_END_GENERAL_CHECK(ax >= 0 && ax < b_rank,
            "aten::tensordot: axis out of range for b");
        FRONT_END_GENERAL_CHECK(seen_b.insert(ax).second,
            "aten::tensordot: duplicate axis in b");
    }
}


//helper function to reshape to 2D
Output<Node> reshape_to_2d(
    const NodeContext& ctx,
    Output<Node> input,
    const std::vector<int64_t>& left_axes,
    const std::vector<int64_t>& right_axes) {

    using namespace ov::op;

    auto shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(input));

    auto gather_dims = [&](const std::vector<int64_t>& axes) {
        auto idx = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
        return ctx.mark_node(std::make_shared<v8::Gather>(
            shape, idx, v0::Constant::create(element::i64, Shape{}, {0})));
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
//helper function to get complement axes
std::vector<int64_t> complement_axes(
    const Output<Node>& tensor,
    const std::vector<int64_t>& axes) {

    auto rank_ps = tensor.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(rank_ps.is_static(),
        "aten::tensordot: static rank required");

    int64_t r = rank_ps.get_length();
    std::vector<bool> used(r, false);

    for (auto ax : axes)
        used[ax] = true;

    std::vector<int64_t> result;
    for (int64_t i = 0; i < r; ++i)
        if (!used[i])
            result.push_back(i);

    return result;
}

//helper function to concatenate shapes
Output<Node> concat_shapes(
    const NodeContext& ctx,
    Output<Node> a,
    Output<Node> b,
    const std::vector<int64_t>& a_axes,
    const std::vector<int64_t>& b_axes) {

    using namespace ov::op;

    auto a_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(a));
    auto b_shape = ctx.mark_node(std::make_shared<v3::ShapeOf>(b));

    auto gather = [&](Output<Node> shape, const std::vector<int64_t>& axes) {
        return ctx.mark_node(std::make_shared<v8::Gather>(
            shape, v0::Constant::create(element::i64, Shape{axes.size()}, axes), v0::Constant::create(element::i64, Shape{}, {0})));
    };

    auto a_dims = gather(a_shape, a_axes);
    auto b_dims = gather(b_shape, b_axes);

    return ctx.mark_node(
        std::make_shared<v0::Concat>(OutputVector{a_dims, b_dims}, 0));
}



OutputVector translate_tensordot(const NodeContext& context) {
    num_inputs_check(context, 3, 4);

    auto a = context.get_input(0);
    auto b = context.get_input(1);

    FRONT_END_GENERAL_CHECK(ov::as_type_ptr<v0::Constant>(context.get_input_from_visible_context(2).get_node_shared_ptr()),
        "aten::tensordot: dims must be constant");

    std::vector<int64_t> a_axes, b_axes;
    parse_tensordot_dims(context, context.get_input(2), a, b, a_axes, b_axes);

    auto a_remain = complement_axes(a, a_axes);
    auto b_remain = complement_axes(b, b_axes);

    std::vector<int64_t> a_perm = a_remain;
    a_perm.insert(a_perm.end(), a_axes.begin(), a_axes.end());

    std::vector<int64_t> b_perm = b_axes;
    b_perm.insert(b_perm.end(), b_remain.begin(), b_remain.end());

    auto a_t = context.mark_node(
        std::make_shared<v1::Transpose>(a, v0::Constant::create(element::i64, Shape{a_perm.size()}, a_perm)));

    auto b_t = context.mark_node(
        std::make_shared<v1::Transpose>(b, v0::Constant::create(element::i64, Shape{b_perm.size()}, b_perm)));

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
