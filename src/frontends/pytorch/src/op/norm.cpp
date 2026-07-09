// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> norm_vector(const NodeContext& context,
                         Output<Node> input_tensor,
                         Output<Node> dim,
                         float p,
                         bool keep_dim) {
    Output<Node> res;
    if (p == 1) {
        res = context.mark_node(std::make_shared<v4::ReduceL1>(input_tensor, dim, keep_dim));
    } else if (p == 2) {
        res = context.mark_node(std::make_shared<v4::ReduceL2>(input_tensor, dim, keep_dim));
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(abs, dim, keep_dim));
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(abs, dim, keep_dim));
    } else if (p == 0) {
        auto input_rank = input_tensor.get_partial_shape().rank();
        PYTORCH_OP_CONVERSION_CHECK(input_rank.is_dynamic() || input_rank.get_length() == 1,
                                    "ord=0 supported only for vector norm");
        auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input_tensor));
        auto cond = context.mark_node(std::make_shared<v1::NotEqual>(input_tensor, zero));
        cond = context.mark_node(std::make_shared<v1::ConvertLike>(cond, input_tensor));
        res = context.mark_node(std::make_shared<v1::ReduceSum>(cond, dim, keep_dim));
    } else {
        auto const_p = context.mark_node(v0::Constant::create(element::f32, Shape{}, {p}));
        const_p = context.mark_node(std::make_shared<v1::ConvertLike>(const_p, input_tensor));
        auto const_p_inv = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0 / p}));
        const_p_inv = context.mark_node(std::make_shared<v1::ConvertLike>(const_p_inv, input_tensor));
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto pow = context.mark_node(std::make_shared<v1::Power>(abs, const_p));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(pow, dim, keep_dim));
        res = context.mark_node(std::make_shared<v1::Power>(sum, const_p_inv));
    }
    return res;
};

// Singular values of the batched trailing (M, N) matrix of `x`, sorted descending, shape (..., K)
// with K = min(M, N). General MxN one-sided Jacobi (values only): orthogonalize the N columns by
// Givens rotations in a v5::Loop (dynamic upper-triangle pair list tiled SVD_SWEEPS times); the column
// norms are the singular values, and for N > M the extra columns converge to ~0 so the top-K norms are
// the true values. Batch dims are flattened to a single axis (a CPU-plugin loop body cannot have a
// dynamic rank). Used by the spectral (ord=+-2) and nuclear ("nuc") matrix norms.
Output<Node> matrix_svdvals(const NodeContext& context, const Output<Node>& x) {
    // Number of Jacobi sweeps: fixed (data-independent) so the loop trip count is static. Rectangular
    // / near-rank-deficient matrices need a few more sweeps than the square 3x3 case.
    constexpr int SVDVALS_SWEEPS = 24;
    auto i64_c = [&](const std::vector<int64_t>& v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{v.size()}, v));
    };
    auto i64_s = [&](int64_t v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{}, {v}));
    };
    auto fc = [&](float v) {
        return context.mark_node(v0::Constant::create(element::f32, Shape{}, {v}));
    };
    auto mul = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Multiply>(a, b));
    };
    auto add = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Add>(a, b));
    };
    auto sub = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Subtract>(a, b));
    };
    auto div = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Divide>(a, b));
    };

    auto x_f32 = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x_f32, element::i64));  // (rank,)
    auto m = context.mark_node(std::make_shared<v8::Gather>(shape, i64_s(-2), i64_s(0)));  // scalar M
    auto n = context.mark_node(std::make_shared<v8::Gather>(shape, i64_s(-1), i64_s(0)));  // scalar N
    auto k = context.mark_node(std::make_shared<v1::Minimum>(m, n));                        // K = min(M,N)
    auto m_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(m, i64_c({0})));
    auto n_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(n, i64_c({0})));

    // Flatten leading batch dims into one axis: working matrix (B, M, N).
    auto flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{i64_c({-1}), m_1d, n_1d}, 0));
    auto A_flat = context.mark_node(std::make_shared<v1::Reshape>(x_f32, flat_shape, /*special_zero=*/false));

    // Upper-triangle column pair list (p < q over N), tiled SVDVALS_SWEEPS times.
    auto seq = context.mark_node(std::make_shared<v4::Range>(i64_s(0), n, i64_s(1), element::i64));  // (N,)
    auto ii = context.mark_node(std::make_shared<v0::Unsqueeze>(seq, i64_c({1})));  // (N,1)
    auto jj = context.mark_node(std::make_shared<v0::Unsqueeze>(seq, i64_c({0})));  // (1,N)
    auto upper = context.mark_node(std::make_shared<v1::Less>(ii, jj));            // (N,N) p<q
    auto coords = context.mark_node(std::make_shared<v3::NonZero>(upper, element::i64));  // (2,P)
    auto p_list = context.mark_node(std::make_shared<v8::Gather>(coords, i64_s(0), i64_s(0)));  // (P,)
    auto q_list = context.mark_node(std::make_shared<v8::Gather>(coords, i64_s(1), i64_s(0)));  // (P,)
    auto sweeps = i64_c({SVDVALS_SWEEPS});
    auto p_all = context.mark_node(std::make_shared<v0::Tile>(p_list, sweeps));
    auto q_all = context.mark_node(std::make_shared<v0::Tile>(q_list, sweeps));
    auto total = context.mark_node(std::make_shared<v8::Gather>(
        context.mark_node(std::make_shared<v3::ShapeOf>(p_all, element::i64)),
        i64_s(0),
        i64_s(0)));  // SVDVALS_SWEEPS * P

    // ---- Loop body: one Givens rotation of columns p, q of the (B, M, N) working matrix ----
    auto A_b = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto p_s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto q_s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto p = std::make_shared<v0::Squeeze>(p_s, i64_c({0}));
    auto q = std::make_shared<v0::Squeeze>(q_s, i64_c({0}));
    auto ax_col = i64_s(2);
    auto ap = std::make_shared<v8::Gather>(A_b, p, ax_col);  // (B, M)
    auto aq = std::make_shared<v8::Gather>(A_b, q, ax_col);
    auto dot = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::ReduceSum>(mul(a, b), i64_c({1}), true));  // (B,1)
    };
    auto alpha = dot(ap, ap);
    auto beta = dot(aq, aq);
    auto gamma = dot(ap, aq);
    auto tiny = fc(1e-30f);
    auto denom = mul(fc(2.0f), gamma);
    auto is_orth = context.mark_node(std::make_shared<v1::Equal>(gamma, fc(0.0f)));
    auto safe_denom = context.mark_node(std::make_shared<v1::Select>(is_orth, tiny, denom));
    auto zeta = div(sub(beta, alpha), safe_denom);
    zeta = context.mark_node(std::make_shared<v0::Clamp>(zeta, -1e18, 1e18));
    auto abs_zeta = context.mark_node(std::make_shared<v0::Abs>(zeta));
    auto sgn = context.mark_node(std::make_shared<v0::Sign>(zeta));
    auto is_zero = context.mark_node(std::make_shared<v1::Equal>(zeta, fc(0.0f)));
    auto sign = context.mark_node(std::make_shared<v1::Select>(is_zero, fc(1.0f), sgn));
    auto t = mul(sign, div(fc(1.0f), add(abs_zeta, context.mark_node(std::make_shared<v0::Sqrt>(
                                                       add(fc(1.0f), mul(zeta, zeta)))))));
    auto cc = div(fc(1.0f), context.mark_node(std::make_shared<v0::Sqrt>(add(fc(1.0f), mul(t, t)))));
    auto ss = mul(cc, t);
    auto na_p = sub(mul(cc, ap), mul(ss, aq));
    auto na_q = add(mul(ss, ap), mul(cc, aq));
    auto pq = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{context.mark_node(std::make_shared<v0::Unsqueeze>(p, i64_c({0}))),
                     context.mark_node(std::make_shared<v0::Unsqueeze>(q, i64_c({0})))},
        0));
    auto pack = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{context.mark_node(std::make_shared<v0::Unsqueeze>(na_p, i64_c({2}))),
                     context.mark_node(std::make_shared<v0::Unsqueeze>(na_q, i64_c({2})))},
        2));  // (B, M, 2)
    auto A_new = context.mark_node(std::make_shared<v3::ScatterUpdate>(A_b, pq, pack, ax_col));

    auto body_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    auto body = std::make_shared<Model>(OutputVector{body_cond, A_new}, ParameterVector{A_b, p_s, q_s});

    auto exec_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    auto loop = std::make_shared<v5::Loop>(total, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(A_b, A_flat, A_new);
    loop->set_sliced_input(p_s, p_all, 0, 1, 1, 0, 0);
    loop->set_sliced_input(q_s, q_all, 0, 1, 1, 0, 0);
    Output<Node> A_res = loop->get_iter_value(A_new, -1);  // (B, M, N)
    context.mark_node(loop);

    // Column norms (B, N); the top-K descending are the singular values.
    auto sq_norms = context.mark_node(std::make_shared<v1::ReduceSum>(mul(A_res, A_res), i64_c({1}), false));  // (B,N)
    auto col_norms = context.mark_node(std::make_shared<v0::Sqrt>(sq_norms));  // (B, N)
    auto topk = context.mark_node(std::make_shared<v11::TopK>(col_norms,
                                                              k,
                                                              1,
                                                              v11::TopK::Mode::MAX,
                                                              v11::TopK::SortType::SORT_VALUES,
                                                              element::i64));
    auto sv_flat = topk->output(0);  // (B, K) descending

    // Restore the original leading batch dims (shape[:-2]) -> (..., K).
    auto k_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(k, i64_c({0})));
    auto batch_shape = context.mark_node(std::make_shared<v8::Slice>(shape, i64_c({0}), i64_c({-2}), i64_c({1})));
    auto sv_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_shape, k_1d}, 0));
    auto sv = context.mark_node(std::make_shared<v1::Reshape>(sv_flat, sv_shape, /*special_zero=*/false));
    return context.mark_node(std::make_shared<v1::ConvertLike>(sv, x));
}

// Move the two matrix axes (dim[0], dim[1]) to the trailing positions (-2, -1) so matrix_svdvals sees
// the matrix as the last two axes. `dim` is the 2-element axis list (may be negative).
Output<Node> move_matrix_axes_to_end(const NodeContext& context, const Output<Node>& x, const Output<Node>& dim) {
    // Normalize the two axes to non-negative, build a permutation that appends them after the rest.
    auto rank = context.mark_node(std::make_shared<v3::ShapeOf>(
        context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i64)),
        element::i64));  // (1,) = [rank]
    auto rank_s = context.mark_node(std::make_shared<v0::Squeeze>(rank));  // scalar rank
    auto dim_i64 = context.mark_node(std::make_shared<v0::Convert>(dim, element::i64));
    auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto zero_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    // norm axes normalized to [0, rank): (a + rank) % rank via Add + Mod.
    auto rank_b = context.mark_node(std::make_shared<v1::Add>(dim_i64, rank_s));
    auto norm_axes = context.mark_node(std::make_shared<v1::Mod>(rank_b, rank_s));  // (2,) in [0,rank)
    // full axis range [0, rank)
    auto all_axes = context.mark_node(std::make_shared<v4::Range>(zero, rank_s, one, element::i64));  // (rank,)
    // mask out the two matrix axes: keep axes not in norm_axes, then append the two matrix axes.
    auto a0 = context.mark_node(std::make_shared<v8::Gather>(norm_axes, zero_1d, zero));  // [ax0]
    auto a1 = context.mark_node(
        std::make_shared<v8::Gather>(norm_axes, v0::Constant::create(element::i64, Shape{1}, {1}), zero));  // [ax1]
    auto eq0 = context.mark_node(std::make_shared<v1::Equal>(all_axes, a0));
    auto eq1 = context.mark_node(std::make_shared<v1::Equal>(all_axes, a1));
    auto is_mat = context.mark_node(std::make_shared<v1::LogicalOr>(eq0, eq1));                 // (rank,)
    auto keep_mask = context.mark_node(std::make_shared<v1::LogicalNot>(is_mat));
    auto keep_idx = context.mark_node(std::make_shared<v3::NonZero>(keep_mask, element::i64));  // (1, rank-2)
    auto keep_axes = context.mark_node(std::make_shared<v8::Gather>(all_axes,
                                                                    context.mark_node(std::make_shared<v0::Squeeze>(
                                                                        keep_idx, zero_1d)),
                                                                    zero));  // (rank-2,)
    auto perm = context.mark_node(std::make_shared<v0::Concat>(OutputVector{keep_axes, a0, a1}, 0));
    return context.mark_node(std::make_shared<v1::Transpose>(x, perm));
}

// Singular-value matrix norms: spectral (mode "max"=ord 2 / "min"=ord -2) and nuclear ("sum"). Moves
// the two `dim` axes to the trailing positions, computes the singular values there, reduces them
// (max/min/sum), and restores the output layout honoring keep_dim.
Output<Node> svd_matrix_norm(const NodeContext& context,
                             const Output<Node>& input_tensor,
                             const Output<Node>& dim,
                             const std::string& mode,
                             bool keep_dim) {
    auto moved = move_matrix_axes_to_end(context, input_tensor, dim);
    auto sv = matrix_svdvals(context, moved);  // (rest..., K)
    auto last_axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    Output<Node> res;
    if (mode == "max") {
        res = context.mark_node(std::make_shared<v1::ReduceMax>(sv, last_axis, false));  // (rest...)
    } else if (mode == "min") {
        res = context.mark_node(std::make_shared<v1::ReduceMin>(sv, last_axis, false));
    } else {  // "sum" -> nuclear norm
        res = context.mark_node(std::make_shared<v1::ReduceSum>(sv, last_axis, false));
    }
    // `res` (the kept axes in original order) already matches torch's keepdim=False output. For
    // keepdim=True, reshape to the input shape with the two matrix axes pinned to 1 (a size-1 axis
    // reorders no data, so the reshape is valid).
    if (keep_dim) {
        auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i64));
        auto rank = context.mark_node(std::make_shared<v0::Squeeze>(
            context.mark_node(std::make_shared<v3::ShapeOf>(shape, element::i64))));  // scalar rank
        auto dim_i64 = context.mark_node(std::make_shared<v0::Convert>(dim, element::i64));
        auto norm_axes = context.mark_node(std::make_shared<v1::Mod>(
            context.mark_node(std::make_shared<v1::Add>(dim_i64, rank)), rank));  // (2,) in [0,rank)
        auto ones = context.mark_node(v0::Constant::create(element::i64, Shape{2}, {1, 1}));
        auto axis0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
        auto kd_shape = context.mark_node(std::make_shared<v3::ScatterUpdate>(shape, norm_axes, ones, axis0));
        res = context.mark_node(std::make_shared<v1::Reshape>(res, kd_shape, /*special_zero=*/false));
    }
    return res;
}

Output<Node> norm_matrix(const NodeContext& context,
                         Output<Node> input_tensor,
                         Output<Node> dim,
                         float p,
                         bool keep_dim) {
    Output<Node> res;
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto first_dim = context.mark_node(std::make_shared<v8::Gather>(dim, zero, zero));
    auto second_dim = context.mark_node(std::make_shared<v8::Gather>(dim, one, zero));
    if (p == 1) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, first_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(sum, second_dim, true));
    } else if (p == std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, second_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMax>(sum, first_dim, true));
    } else if (p == -std::numeric_limits<float>::infinity()) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, second_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(sum, first_dim, true));
    } else if (p == -1) {
        auto abs = context.mark_node(std::make_shared<v0::Abs>(input_tensor));
        auto sum = context.mark_node(std::make_shared<v1::ReduceSum>(abs, first_dim, true));
        res = context.mark_node(std::make_shared<v1::ReduceMin>(sum, second_dim, true));
    } else if (p == 2) {
        // Spectral norm: the largest singular value.
        return svd_matrix_norm(context, input_tensor, dim, "max", keep_dim);
    } else if (p == -2) {
        // Smallest singular value.
        return svd_matrix_norm(context, input_tensor, dim, "min", keep_dim);
    } else {
        PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p, " for matrix norm");
    }
    if (!keep_dim) {
        res = context.mark_node(std::make_shared<v0::Squeeze>(res, dim));
    }

    return res;
};

Output<Node> frobenius_norm(const NodeContext& context, Output<Node> x, Output<Node> dim, bool keep_dim) {
    auto sqr = context.mark_node(std::make_shared<v1::Multiply>(x, x));
    auto sumsqr = context.mark_node(std::make_shared<v1::ReduceSum>(sqr, dim, keep_dim));
    return context.mark_node(std::make_shared<v0::Sqrt>(sumsqr));
}
};  // namespace

OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 2, 6);
    auto input_tensor = context.get_input(0);
    auto p_node_type = context.get_input_type(1);
    bool keep_dim = false;
    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, input_tensor);
    } else {
        dim = concat_list_construct(context.get_input(2));
    }
    if (!context.input_is_none(3)) {
        keep_dim = context.const_input<bool>(3);
    }
    if (!context.input_is_none(4)) {
        input_tensor = apply_dtype(context, 4, input_tensor);
    }
    Output<Node> res;
    if (p_node_type.is<type::Str>()) {
        auto p_str = context.const_input<std::string>(1);
        if (p_str == "fro") {
            res = frobenius_norm(context, input_tensor, dim, keep_dim);
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
        }
    } else {
        auto p = context.const_input<float>(1);
        res = norm_vector(context, input_tensor, dim, p, keep_dim);
    }
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, res);
    }
    return {res};
};

OutputVector translate_weight_norm(const NodeContext& context) {
    // aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
    num_inputs_check(context, 3, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    Output<Node> dim;
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto rank = std::get<1>(get_shape_rank(context, x, true));
    if (context.input_is_none(2)) {
        dim = context.mark_node(std::make_shared<v0::Range>(zero, rank, one));
    } else {
        dim = get_input_as_i32(context, 2);
        auto dims_before = context.mark_node(std::make_shared<v0::Range>(zero, dim, one));
        auto dim_next = context.mark_node(std::make_shared<v1::Add>(dim, one));
        auto dims_after = context.mark_node(std::make_shared<v0::Range>(dim_next, rank, one));
        dim = context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, dims_after}, 0));
    }
    Output<Node> res;
    auto norm = context.mark_node(std::make_shared<v4::ReduceL2>(x, dim, true));
    auto y_norm = context.mark_node(std::make_shared<v1::Divide>(y, norm));
    return {context.mark_node(std::make_shared<v1::Multiply>(x, y_norm))};
};

OutputVector translate_linalg_vector_norm(const NodeContext& context) {
    // aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType?
    // dtype=None) -> Tensor
    // aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool
    // keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!):
    num_inputs_check(context, 3, 6);
    auto x = context.get_input(0);
    // ord defines the vector norm that is computed.
    auto ord = context.const_input<float>(1);
    bool keep_dim = false;
    if (!context.input_is_none(3)) {
        keep_dim = context.const_input<bool>(3);
    }
    Output<Node> dim;
    Output<Node> result;
    // If dim= None, x will be flattened before the norm is computed.
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, x);
    } else {
        dim = concat_list_construct(context.get_input(2));
    }
    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.vector_norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    result = norm_vector(context, x, dim, ord, keep_dim);
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_linalg_matrix_norm(const NodeContext& context) {
    // aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2, -1], bool keepdim=False, *, ScalarType?
    // dtype=None, Tensor(a!) out) -> Tensor(a!) aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2, -1],
    // bool keepdim=False, *, ScalarType? dtype=None) aten::linalg_matrix_norm.str_ord(Tensor self, str ord="fro", int[]
    // dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None)
    num_inputs_check(context, 5, 6);
    auto x = context.get_input(0);
    // ord defines the vector norm that is computed can be string or number
    auto ord_type = context.get_input_type(1);
    auto dim = concat_list_construct(context.get_input(2));
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> result;

    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.matrix_norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    if (ord_type.is<type::Str>()) {
        auto p_str = context.const_input<std::string>(1);
        if (p_str == "fro") {
            result = frobenius_norm(context, x, dim, keep_dim);
        } else if (p_str == "nuc") {
            // Nuclear norm: the sum of the singular values.
            result = svd_matrix_norm(context, x, dim, "sum", keep_dim);
        } else {
            PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
        }
    } else {
        auto p = context.const_input<float>(1);
        result = norm_matrix(context, x, dim, p, keep_dim);
    }
    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_linalg_norm(const NodeContext& context) {
    // aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)
    // aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None)
    // aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType?
    // dtype=None, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 5, 6);
    auto x = context.get_input(0);
    bool keep_dim = context.const_input<bool>(3);
    Output<Node> result;
    Output<Node> dim;
    // dtype may be used to perform the computation in a more precise dtype. It is semantically equivalent to calling
    // linalg.norm(x.to(dtype))
    if (!context.input_is_none(4)) {
        x = apply_dtype(context, 4, x);
    }
    // If dim=None apply for all dimensions
    if (context.input_is_none(2)) {
        dim = get_node_axes_range(context, x);
    } else {
        dim = concat_list_construct(context.get_input(2));
    }
    // ord=None: Frobenius and vector L2 are both sqrt(sum(x^2)) over `dim` for the real inputs here,
    // so a single L2 reduction is correct and rank-agnostic -- no static rank or foldable `dim` needed.
    if (context.input_is_none(1)) {
        result = norm_vector(context, x, dim, 2, keep_dim);
    } else {
        // ord defines the  norm that is computed can be string or number
        auto ord_type = context.get_input_type(1);
        if (ord_type.is<type::Str>()) {
            auto p_str = context.const_input<std::string>(1);
            if (p_str == "fro") {
                result = frobenius_norm(context, x, dim, keep_dim);
            } else if (p_str == "nuc") {
                // Nuclear norm: the sum of the singular values.
                result = svd_matrix_norm(context, x, dim, "sum", keep_dim);
            } else {
                PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported ord ", p_str);
            }
        } else {
            auto p = context.const_input<float>(1);
            if (!context.input_is_none(2)) {
                auto const_dim = context.const_input<std::vector<int64_t>>(2);
                if (const_dim.size() == 2) {
                    result = norm_matrix(context, x, dim, p, keep_dim);
                } else {
                    result = norm_vector(context, x, dim, p, keep_dim);
                }
            } else {
                result = norm_vector(context, x, dim, p, keep_dim);
            }
        }
    }

    // output tensor
    if (!context.input_is_none(5)) {
        context.mutate_input(5, result);
    }
    return {result};
};

OutputVector translate_frobenius_norm(const NodeContext& context) {
    // aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
    // aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    auto x = context.get_input(0);
    bool keep_dim = context.const_input<bool>(2);
    Output<Node> dim;
    if (context.input_is_none(1)) {
        dim = get_axes_range(context, 0);

    } else {
        dim = concat_list_construct(context.get_input(1));
    }
    auto result = frobenius_norm(context, x, dim, keep_dim);
    if (!context.input_is_none(3)) {
        context.mutate_input(3, result);
    }
    return {result};
}

OutputVector translate_rms_norm(const NodeContext& context) {
    // Tensor = aten::rms_norm(%input_data.1, %2, %4, %3)
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto normalized_shape = context.get_input(1);
    Output<Node> eps;
    if (!context.input_is_none(3)) {
        eps = context.get_input(3);
        if (eps.get_element_type().is_dynamic() || eps.get_element_type() != x.get_element_type())
            eps = std::make_shared<v1::ConvertLike>(eps, x);
    } else {
        switch (x.get_element_type()) {
        case element::bf16:
            eps = v0::Constant::create(ov::element::bf16, {}, {std::numeric_limits<bfloat16>::epsilon()});
            break;
        case element::f16:
            eps = v0::Constant::create(ov::element::f16, {}, {std::numeric_limits<float16>::epsilon()});
            break;
        case element::f64:
            eps = v0::Constant::create(ov::element::f64, {}, {std::numeric_limits<double>::epsilon()});
            break;
        case element::f32:
            eps = v0::Constant::create(ov::element::f32, {}, {std::numeric_limits<float>::epsilon()});
            break;
        default:
            eps = v0::Constant::create(ov::element::f32, {}, {std::numeric_limits<float>::epsilon()});
            eps = std::make_shared<v1::ConvertLike>(eps, x);
        }
    }
    context.mark_output(eps);

    // normalized shape represent D last dimensions to be normalized
    auto num_axes = context.mark_node(std::make_shared<v3::ShapeOf>(normalized_shape, element::i32));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    num_axes = context.mark_node(std::make_shared<v0::Squeeze>(num_axes, zero));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto axes_range = context.mark_node(std::make_shared<v4::Range>(num_axes, zero, minus_one, element::i32));
    auto axes = context.mark_node(std::make_shared<v1::Multiply>(axes_range, minus_one));

    // Build the RMSNorm decomposition via the shared helper.
    ov::Output<ov::Node> scale;
    if (!context.input_is_none(2)) {
        scale = context.get_input(2);
    }
    ov::pass::NodeRegistry reg;
    auto result = ov::decomposition::rms_norm(reg, x, axes, eps, scale);
    context.mark_nodes(reg.get());
    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
