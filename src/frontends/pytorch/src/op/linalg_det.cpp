// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace detail {

// Element x[..., i, j] of a batched matrix (trailing singleton axes removed); positional indexing
// works for static and dynamic shapes.
Output<Node> matrix_element(const NodeContext& context, const Output<Node>& x, int64_t i, int64_t j) {
    auto row_idx = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {i}));
    auto col_idx = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {j}));
    auto axis_row = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-2}));
    auto axis_col = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto row = context.mark_node(std::make_shared<v8::Gather>(x, row_idx, axis_row));     // (..., 1, n)
    auto elem = context.mark_node(std::make_shared<v8::Gather>(row, col_idx, axis_col));  // (..., 1, 1)
    auto squeeze_axes = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {-2, -1}));
    return context.mark_node(std::make_shared<v0::Squeeze>(elem, squeeze_axes));  // (...)
}

// Closed-form determinant of a batched 1x1 matrix.
Output<Node> det_1x1(const NodeContext& context, const Output<Node>& x) {
    return matrix_element(context, x, 0, 0);
}

// Closed-form determinant of a batched 2x2 matrix: ad - bc.
Output<Node> det_2x2(const NodeContext& context, const Output<Node>& x) {
    auto a = matrix_element(context, x, 0, 0);
    auto b = matrix_element(context, x, 0, 1);
    auto c = matrix_element(context, x, 1, 0);
    auto d = matrix_element(context, x, 1, 1);
    auto ad = context.mark_node(std::make_shared<v1::Multiply>(a, d));
    auto bc = context.mark_node(std::make_shared<v1::Multiply>(b, c));
    return context.mark_node(std::make_shared<v1::Subtract>(ad, bc));
}

// Closed-form determinant of a batched 3x3 matrix via cofactor expansion.
Output<Node> det_3x3(const NodeContext& context, const Output<Node>& x) {
    Output<Node> m[3][3];
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            m[i][j] = matrix_element(context, x, i, j);
        }
    }
    auto mul = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Multiply>(a, b));
    };
    auto sub = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Subtract>(a, b));
    };
    auto c0 = sub(mul(m[1][1], m[2][2]), mul(m[1][2], m[2][1]));
    auto c1 = sub(mul(m[1][0], m[2][2]), mul(m[1][2], m[2][0]));
    auto c2 = sub(mul(m[1][0], m[2][1]), mul(m[1][1], m[2][0]));
    auto t0 = mul(m[0][0], c0);
    auto t1 = mul(m[0][1], c1);
    auto t2 = mul(m[0][2], c2);
    return context.mark_node(std::make_shared<v1::Add>(sub(t0, t1), t2));
}

// General determinant of a batched NxN matrix via LU decomposition with partial pivoting,
// following PyTorch's own algorithm det = sign(P) * prod(diag(U)). The elimination runs in a
// v5::Loop (trip count = N), so it handles the dynamic trailing dims the TorchScript decoder
// produces (where the size is unknown at conversion time). Each iteration k:
//   - selects the pivot row p = argmax_{r>=k} |A[r, k]| (partial pivoting),
//   - swaps rows k and p (per-batch), flipping the running determinant sign when p != k,
//   - eliminates below the pivot with a rank-1 update A -= outer(L, A[k, :]),
//   - multiplies the accumulator by the pivot A[k, k].
// After N iterations the accumulator holds sign(P) * prod(diag(U)) = det(A).
Output<Node> det_lu(const NodeContext& context, const Output<Node>& x) {
    auto i64_c = [&](const std::vector<int64_t>& v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{v.size()}, v));
    };
    auto i64_s = [&](int64_t v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{}, {v}));
    };
    auto f32_s = [&](float v) {
        return context.mark_node(v0::Constant::create(element::f32, Shape{}, {v}));
    };

    // Work in f32 for numerical headroom (LU on integer/half inputs would be lossy); cast back at
    // the end via ConvertLike so the result matches the input element type.
    auto x_f32 = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));

    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x_f32, element::i64));  // (rank,)
    auto n = context.mark_node(std::make_shared<v8::Gather>(shape, i64_s(-1), i64_s(0)));  // scalar N
    // Row-index vector r = [0, 1, ..., N-1] of shape (1, N) for masked argmax / swap logic.
    auto seq = context.mark_node(std::make_shared<v4::Range>(i64_s(0), n, i64_s(1), element::i64));  // (N,)
    auto r_row = context.mark_node(std::make_shared<v0::Unsqueeze>(seq, i64_c({0})));                // (1, N)

    // Flatten all leading batch axes into a single axis so the loop body has a fixed rank (B, N, N).
    // A CPU-plugin body cannot have a dynamic rank; the batch shape is restored on the result.
    // This reshape to [-1, N, N] doubles as the squareness guard: it changes rank (so no shape pass
    // can fold it to a no-op) and its element-count check fails loudly on any non-N x N trailing pair
    // (e.g. [1, 9] or [3, 4]). The op-labeled name surfaces the op in the CPU runtime error.
    auto n_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(n, i64_c({0})));  // (1,) = [N]
    auto flat_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{i64_c({-1}), n_1d, n_1d}, 0));
    auto x_flat = context.mark_node(std::make_shared<v1::Reshape>(x_f32, flat_shape, /*special_zero=*/false));
    x_flat->set_friendly_name("aten::det/linalg_det/requires_square");
    auto b_flat = context.mark_node(std::make_shared<v3::ShapeOf>(x_flat, element::i64));
    auto b_dim = context.mark_node(std::make_shared<v8::Slice>(b_flat, i64_c({0}), i64_c({1}), i64_c({1})));  // [B]
    auto acc_init = context.mark_node(std::make_shared<v3::Broadcast>(f32_s(1.0f), b_dim));  // (B,) ones

    // ---- Loop body: operates on the flattened rank-3 (B, N, N) working matrix ----
    auto A_body = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, -1, -1});
    auto acc_body = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1});
    auto k_slice = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});    // sliced iter index
    auto r_body = std::make_shared<v0::Parameter>(element::i64, PartialShape{1, -1});  // invariant (1, N)

    auto k = std::make_shared<v0::Squeeze>(k_slice, i64_c({0}));  // scalar current pivot index

    // -- partial pivot selection: argmax over rows r>=k of |A[:, :, k]| --
    auto ax1 = i64_s(1), ax2 = i64_s(2);
    auto col_k = std::make_shared<v8::Gather>(A_body, k, ax2);      // (B, N)
    auto abs_col = std::make_shared<v0::Abs>(col_k);
    auto ge = std::make_shared<v1::GreaterEqual>(r_body, k);        // (1, N) bool -> broadcast
    auto neg_big = f32_s(-3.0e38f);
    auto masked = std::make_shared<v1::Select>(ge, abs_col, neg_big);
    auto topk = std::make_shared<v11::TopK>(masked,
                                            i64_s(1),
                                            1,
                                            v11::TopK::Mode::MAX,
                                            v11::TopK::SortType::SORT_INDICES,
                                            element::i64);
    auto p = topk->output(1);  // (B, 1) i64 pivot row index

    // -- per-batch row permutation swapping k <-> p, then row-gather A --
    auto eq_k = std::make_shared<v1::Equal>(r_body, k);  // (1, N)
    auto eq_p = std::make_shared<v1::Equal>(r_body, p);  // (B, N)
    auto perm = std::make_shared<v1::Select>(eq_k, p, std::make_shared<v1::Select>(eq_p, k, r_body));  // (B, N)
    // Permute rows on axis 1 via GatherElements (index[b,i,:] = perm[b,i]) rather than
    // Gather(batch_dims=1): the batch_dims Gather hits a CPU-plugin AVX2 JIT kernel bug when the Loop
    // body compiles with static shapes; GatherElements takes a different, correct kernel path.
    auto a_shape = std::make_shared<v3::ShapeOf>(A_body, element::i64);
    auto perm_idx = std::make_shared<v3::Broadcast>(std::make_shared<v0::Unsqueeze>(perm, ax2), a_shape);
    auto A_sw = std::make_shared<v6::GatherElements>(A_body, perm_idx, 1);  // (B, N, N) rows permuted

    // -- sign flip when a real swap happened (p != k) --
    auto neq = std::make_shared<v1::NotEqual>(p, k);  // (B, 1) bool
    auto sign = std::make_shared<v1::Select>(neq, f32_s(-1.0f), f32_s(1.0f));  // (B, 1)

    // -- pivot, multipliers, rank-1 elimination update --
    auto row_k = std::make_shared<v8::Gather>(A_sw, k, ax1);          // (B, N) pivot row
    auto pivot = std::make_shared<v8::Gather>(row_k, k, ax1);         // (B,) pivot value
    auto col_k_new = std::make_shared<v8::Gather>(A_sw, k, ax2);      // (B, N) column k
    auto gt = std::make_shared<v1::Greater>(r_body, k);              // (1, N) rows below the diagonal
    auto pivot_col = std::make_shared<v0::Unsqueeze>(pivot, i64_c({1}));  // (B, 1)
    auto ratio = std::make_shared<v1::Divide>(col_k_new, pivot_col);
    auto mult = std::make_shared<v1::Select>(gt, ratio, f32_s(0.0f));     // (B, N) multipliers
    auto mult_c = std::make_shared<v0::Unsqueeze>(mult, ax2);             // (B, N, 1)
    auto row_r = std::make_shared<v0::Unsqueeze>(row_k, ax1);            // (B, 1, N)
    auto update = std::make_shared<v1::Multiply>(mult_c, row_r);         // (B, N, N)
    auto A_new = std::make_shared<v1::Subtract>(A_sw, update);           // (B, N, N)

    // -- accumulate det *= pivot * sign --
    auto factor = std::make_shared<v1::Multiply>(pivot, std::make_shared<v0::Squeeze>(sign, i64_c({1})));
    auto acc_new = std::make_shared<v1::Multiply>(acc_body, factor);

    auto body_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    auto body = std::make_shared<Model>(OutputVector{body_cond, A_new, acc_new},
                                        ParameterVector{A_body, acc_body, k_slice, r_body});

    // ---- Loop ----
    auto trip_count = std::make_shared<v0::Convert>(n, element::i64);  // N iterations
    auto exec_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    auto loop = std::make_shared<v5::Loop>(trip_count, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});  // body condition is output 0
    loop->set_merged_input(A_body, x_flat, A_new);
    loop->set_merged_input(acc_body, acc_init, acc_new);
    loop->set_sliced_input(k_slice, seq, 0, 1, 1, 0, 0);  // slice one index per iteration along axis 0
    loop->set_invariant_input(r_body, r_row);
    auto det_flat = loop->get_iter_value(acc_new, -1);  // (B,) determinant per flattened batch element

    // Restore the original batch shape (shape[:-2]).
    auto det_marked = context.mark_node(det_flat.get_node_shared_ptr());
    auto batch_shape =
        context.mark_node(std::make_shared<v8::Slice>(shape, i64_c({0}), i64_c({-2}), i64_c({1})));
    auto det = context.mark_node(std::make_shared<v1::Reshape>(det_marked, batch_shape, /*special_zero=*/false));
    return context.mark_node(std::make_shared<v1::ConvertLike>(det, x));
}

// Determinant of a batched square matrix. When the trailing dims are statically known and small
// (1x1 / 2x2 / 3x3) a closed-form expansion is used (reachable on the FX / convert_model(input=...)
// paths); the size-n static path guards non-square inputs via ensure_trailing_square. Otherwise —
// including the dynamic-dims TorchScript trace path — the general LU decomposition handles any square
// size, and its own [-1, N, N] flatten reshape enforces squareness (a rank-changing reshape cannot be
// folded to a no-op, unlike a rank-preserving [.., N, N] guard).
Output<Node> det_dispatch(const NodeContext& context, const Output<Node>& x) {
    const auto& ps = x.get_partial_shape();
    const auto rank = ps.rank();
    if (rank.is_static()) {
        const auto& last = ps[rank.get_length() - 1];
        if (last.is_static()) {
            const auto n = last.get_length();
            if (n == 1)
                return det_1x1(context, ensure_trailing_square(context, x, 1, "aten::det/linalg_det"));
            if (n == 2)
                return det_2x2(context, ensure_trailing_square(context, x, 2, "aten::det/linalg_det"));
            if (n == 3)
                return det_3x3(context, ensure_trailing_square(context, x, 3, "aten::det/linalg_det"));
        }
    }
    return det_lu(context, x);
}

}  // namespace detail

OutputVector translate_linalg_det(const NodeContext& context) {
    // aten::linalg_det(Tensor A) -> Tensor
    // aten::det(Tensor self) -> Tensor
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    return {detail::det_dispatch(context, x)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
