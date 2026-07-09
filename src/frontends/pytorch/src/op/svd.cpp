// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"
// After utils.hpp: op/jacobi_svd.hpp opens ns ...::pytorch::op, but utils.hpp uses unqualified op::.
#include "op/jacobi_svd.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

// One-sided Jacobi converges in ~5 sweeps for small matrices; more sweeps add margin for larger N.
// The count is fixed (data-independent) so the sweep loop has a static trip count.
constexpr int SVD_SWEEPS = 16;
// The static 3x3 fast path converges in ~5 sweeps; 6 adds margin (matches torch.svd to ~1e-5).
constexpr int SVD_SWEEPS_3X3 = 6;
// Column index pairs swept by the 3x3 one-sided Jacobi fast path.
constexpr int PAIRS_3X3[3][2] = {{0, 1}, {0, 2}, {1, 2}};

// SVD of batched 3x3 matrices via ONE-SIDED JACOBI, fully unrolled at build time (no Loop). Used as
// a fast path when the trailing dims are statically 3x3 (the SAM-6D Kabsch regime): it avoids the
// v5::Loop the general path builds, and its fixed-sweep sequence is fp32-accurate for rank-deficient
// 3x3 matrices. Operating on A directly (not A^T A) keeps that accuracy. Returns {U, S, V}: U
// (...,3,3), S (...,3) descending, V (...,3,3) with the right singular vectors as columns (torch.svd).
class JacobiSvd3x3 {
public:
    JacobiSvd3x3(const NodeContext& context, element::Type et) : m_ctx(context), m_et(et), m_tiny(cf(1e-30f)) {}

    std::tuple<Output<Node>, Output<Node>, Output<Node>> build(const Output<Node>& A_in) {
        // Working columns of A and of V (V starts as identity columns).
        Output<Node> a[3] = {mcol(A_in, 0), mcol(A_in, 1), mcol(A_in, 2)};
        Output<Node> v[3] = {ident_col(A_in, 0), ident_col(A_in, 1), ident_col(A_in, 2)};

        for (int sweep = 0; sweep < SVD_SWEEPS_3X3; ++sweep) {
            for (auto& pr : PAIRS_3X3) {
                jacobi_rotate(a[pr[0]], a[pr[1]], v[pr[0]], v[pr[1]]);
            }
        }

        // Singular values = column norms; left singular vectors = normalized columns.
        Output<Node> sig[3], u[3];
        for (int k = 0; k < 3; ++k) {
            sig[k] = sqrt(dot(a[k], a[k]));  // (...,1,1)
            u[k] = div(a[k], add(sig[k], m_tiny));
        }

        // Sort columns by descending singular value (3-element sorting network): (0,1),(1,2),(0,1).
        cmp_exchange(sig[0], sig[1], u[0], u[1], v[0], v[1]);
        cmp_exchange(sig[1], sig[2], u[1], u[2], v[1], v[2]);
        cmp_exchange(sig[0], sig[1], u[0], u[1], v[0], v[1]);

        auto U = colstack(u[0], u[1], u[2]);
        auto V = colstack(v[0], v[1], v[2]);
        // sig[k] are (...,1,1); concat along -1 -> (...,1,3); squeeze -2 -> (...,3).
        auto S_kd = m_ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{sig[0], sig[1], sig[2]}, -1));
        auto S = m_ctx.mark_node(std::make_shared<v0::Squeeze>(S_kd, ci_s(-2)));
        return {U, S, V};
    }

private:
    Output<Node> cf(float v) {
        return m_ctx.mark_node(v0::Constant::create(m_et, Shape{}, {v}));
    }
    static std::shared_ptr<v0::Constant> ci(int64_t v) {
        return v0::Constant::create(element::i32, Shape{1}, {v});
    }
    static std::shared_ptr<v0::Constant> ci_s(int64_t v) {
        return v0::Constant::create(element::i32, Shape{}, {v});
    }
    Output<Node> mul(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::Multiply>(a, b));
    }
    Output<Node> add(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::Add>(a, b));
    }
    Output<Node> sub(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::Subtract>(a, b));
    }
    Output<Node> div(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::Divide>(a, b));
    }
    Output<Node> sqrt(const Output<Node>& a) {
        return m_ctx.mark_node(std::make_shared<v0::Sqrt>(a));
    }
    Output<Node> sel(const Output<Node>& c, const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::Select>(c, a, b));
    }
    // Column k of a (...,3,3) matrix as a (...,3,1) keepdim vector (trailing axis = component).
    Output<Node> mcol(const Output<Node>& m, int64_t k) {
        return m_ctx.mark_node(std::make_shared<v8::Gather>(m, ci(k), ci_s(-1)));  // (...,3,1)
    }
    // Identity column k, broadcast to the batch of `like`, as a (...,3,1) vector.
    Output<Node> ident_col(const Output<Node>& like, int64_t k) {
        std::vector<float> e(3, 0.0f);
        e[static_cast<size_t>(k)] = 1.0f;
        auto col = m_ctx.mark_node(v0::Constant::create(m_et, Shape{3, 1}, e));
        auto zero_like = mul(mcol(like, 0), cf(0.0f));
        return add(col, zero_like);
    }
    // Reduce over the vector axis (-2) keeping it: dot of two (...,3,1) vectors -> (...,1,1).
    Output<Node> dot(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::ReduceSum>(mul(a, b), ci_s(-2), true));
    }
    // One Jacobi rotation orthogonalizing columns (ap, aq); the same rotation is applied to (vp, vq).
    void jacobi_rotate(Output<Node>& ap, Output<Node>& aq, Output<Node>& vp, Output<Node>& vq) {
        auto alpha = dot(ap, ap);  // (...,1,1)
        auto beta = dot(aq, aq);
        auto gamma = dot(ap, aq);
        auto denom = mul(cf(2.0f), gamma);
        auto zeta = div(sub(beta, alpha), add(denom, m_tiny));
        zeta = m_ctx.mark_node(std::make_shared<v0::Clamp>(zeta, -1e18, 1e18));
        auto azeta = absval(zeta);
        auto t_mag = div(cf(1.0f), add(azeta, sqrt(add(cf(1.0f), mul(zeta, zeta)))));
        auto t = mul(signum(zeta), t_mag);
        auto orthogonal = m_ctx.mark_node(std::make_shared<v1::Less>(absval(gamma), m_tiny));
        t = sel(orthogonal, cf(0.0f), t);
        auto c = div(cf(1.0f), sqrt(add(cf(1.0f), mul(t, t))));
        auto s = mul(c, t);
        rotate_pair(ap, aq, c, s);
        rotate_pair(vp, vq, c, s);
    }
    void rotate_pair(Output<Node>& p, Output<Node>& q, const Output<Node>& c, const Output<Node>& s) {
        auto np = sub(mul(c, p), mul(s, q));
        auto nq = add(mul(s, p), mul(c, q));
        p = np;
        q = nq;
    }
    // Compare-exchange so sig_p >= sig_q after the call, swapping the U and V columns to match.
    void cmp_exchange(Output<Node>& sp,
                      Output<Node>& sq,
                      Output<Node>& up,
                      Output<Node>& uq,
                      Output<Node>& vp,
                      Output<Node>& vq) {
        auto swap = m_ctx.mark_node(std::make_shared<v1::Less>(sp, sq));  // swap if sp < sq
        auto nsp = sel(swap, sq, sp);
        auto nsq = sel(swap, sp, sq);
        auto nup = sel(swap, uq, up);
        auto nuq = sel(swap, up, uq);
        auto nvp = sel(swap, vq, vp);
        auto nvq = sel(swap, vp, vq);
        sp = nsp;
        sq = nsq;
        up = nup;
        uq = nuq;
        vp = nvp;
        vq = nvq;
    }
    Output<Node> absval(const Output<Node>& x) {
        return m_ctx.mark_node(std::make_shared<v0::Abs>(x));
    }
    Output<Node> signum(const Output<Node>& x) {
        // sign(x) forcing sign(0) = +1: v0::Sign(0) = 0 would zero t and skip the +-45 degree rotation
        // when two equal-norm columns are non-orthogonal (zeta == 0), leaving them un-orthogonalized.
        auto s = m_ctx.mark_node(std::make_shared<v0::Sign>(x));
        auto is_zero = m_ctx.mark_node(std::make_shared<v1::Equal>(x, cf(0.0f)));
        return sel(is_zero, cf(1.0f), s);
    }
    // Stack three (...,3,1) column vectors into a (...,3,3) matrix (columns).
    Output<Node> colstack(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c) {
        return m_ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{a, b, c}, -1));
    }

    const NodeContext& m_ctx;
    element::Type m_et;
    Output<Node> m_tiny;
};

// General NxN one-sided Jacobi SVD, size-agnostic (works when N is dynamic, i.e. the TorchScript
// trace path). The columns are orthogonalized by run_jacobi_column_loop (Givens rotations
// accumulated into V); the singular values are then the final column norms and the left singular
// vectors are the normalized columns. Leading batch dims are flattened to rank-3 first (a CPU-plugin
// loop body cannot have a dynamic rank).
//
// Returns {U, S, V}: U (..., N, N), S (..., N) descending, V (..., N, N) with the right singular
// vectors as columns (the torch.svd convention).
std::tuple<Output<Node>, Output<Node>, Output<Node>> jacobi_svd(const NodeContext& context,
                                                                const Output<Node>& A_in,
                                                                element::Type et) {
    auto i64_c = [&](const std::vector<int64_t>& v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{v.size()}, v));
    };
    auto i64_s = [&](int64_t v) {
        return context.mark_node(v0::Constant::create(element::i64, Shape{}, {v}));
    };
    auto fc = [&](float v) {
        return context.mark_node(v0::Constant::create(et, Shape{}, {v}));
    };
    auto mul = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Multiply>(a, b));
    };
    auto add = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Add>(a, b));
    };
    auto div = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Divide>(a, b));
    };

    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(A_in, element::i64));     // (rank,)
    auto n = context.mark_node(std::make_shared<v8::Gather>(shape, i64_s(-1), i64_s(0)));  // scalar N
    auto n_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(n, i64_c({0})));         // [N]

    // Flatten to (B, N, N); the reshape doubles as the squareness guard (aten::svd/requires_square).
    auto A_flat = flatten_batch_to_square(context, A_in, n_1d, "aten::svd");
    auto jac = run_jacobi_column_loop(context, A_flat, n, SVD_SWEEPS, et, /*accumulate_v=*/true);

    // Singular values = column norms (B, N); left singular vectors U = A_col / sigma.
    auto tiny = fc(1e-30f);
    auto sq_norms = context.mark_node(std::make_shared<v1::ReduceSum>(mul(jac.a, jac.a), i64_c({1}), true));  // (B,1,N)
    auto sig = context.mark_node(std::make_shared<v0::Sqrt>(sq_norms));  // (B, 1, N)
    auto U_flat = div(jac.a, add(sig, tiny));                            // (B, N, N)
    auto S_flat = context.mark_node(std::make_shared<v0::Squeeze>(sig, i64_c({1})));  // (B, N)

    // Sort columns by descending singular value: TopK on S gives the permutation; gather U/V columns
    // and S entries by it.
    auto topk = context.mark_node(std::make_shared<v11::TopK>(S_flat,
                                                              n,
                                                              1,
                                                              v11::TopK::Mode::MAX,
                                                              v11::TopK::SortType::SORT_VALUES,
                                                              element::i64));
    auto S_sorted = topk->output(0);  // (B, N) descending
    auto order = topk->output(1);     // (B, N) column permutation
    // Permute columns on axis 2 via GatherElements (index[b,:,k] = order[b,k]) rather than
    // Gather(batch_dims=1): the batch_dims Gather hits a CPU-plugin AVX2 JIT kernel bug when the graph
    // compiles with static shapes; GatherElements takes a different, correct kernel path.
    auto u_full_shape = context.mark_node(std::make_shared<v3::ShapeOf>(U_flat, element::i64));
    auto order_idx = context.mark_node(
        std::make_shared<v3::Broadcast>(context.mark_node(std::make_shared<v0::Unsqueeze>(order, i64_c({1}))),
                                        u_full_shape));                                             // (B, N, N)
    auto U_sorted = context.mark_node(std::make_shared<v6::GatherElements>(U_flat, order_idx, 2));  // (B,N,N)
    auto V_sorted = context.mark_node(std::make_shared<v6::GatherElements>(jac.v, order_idx, 2));   // (B,N,N)

    // Restore the original leading batch dims (shape[:-2]) on U, S, V.
    auto U = restore_leading_batch(context, U_sorted, shape, {n_1d, n_1d});
    auto V = restore_leading_batch(context, V_sorted, shape, {n_1d, n_1d});
    auto S = restore_leading_batch(context, S_sorted, shape, {n_1d});
    return {U, S, V};
}

// Transpose the trailing two axes of a batched matrix (V -> Vh = V^T for the linalg_svd convention).
Output<Node> transpose_last2(const NodeContext& context, const Output<Node>& x) {
    // Build the permutation [0, 1, ..., rank-3, rank-1, rank-2] dynamically so it works for any rank.
    // Transpose requires non-negative axes, so the last two are computed as rank-1 and rank-2.
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i64));
    auto rank = context.mark_node(std::make_shared<v3::ShapeOf>(shape, element::i64));  // (1,) = [rank]
    auto rank_s = context.mark_node(std::make_shared<v0::Squeeze>(rank));               // scalar rank
    auto zero = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto two = context.mark_node(v0::Constant::create(element::i64, Shape{}, {2}));
    auto head_end = context.mark_node(std::make_shared<v1::Subtract>(rank_s, two));                // rank-2
    auto seq = context.mark_node(std::make_shared<v4::Range>(zero, head_end, one, element::i64));  // [0..rank-3]
    auto rm1 = context.mark_node(std::make_shared<v1::Subtract>(rank_s, one));                     // rank-1
    auto rm2 = head_end;                                                                           // rank-2
    auto last2 = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{context.mark_node(std::make_shared<v0::Unsqueeze>(rm1, zero)),
                                                  context.mark_node(std::make_shared<v0::Unsqueeze>(rm2, zero))},
                                     0));  // [rank-1, rank-2]
    auto perm = context.mark_node(std::make_shared<v0::Concat>(OutputVector{seq, last2}, 0));
    return context.mark_node(std::make_shared<v1::Transpose>(x, perm));
}

OutputVector svd_common(const NodeContext& context, bool return_vh) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);

    auto in_et = x.get_element_type();
    // Compute in f32 (f16/bf16 are too coarse for the rotations); keep f64 when statically requested.
    auto compute_et = (in_et == element::f64) ? element::f64 : element::f32;
    Output<Node> A = x;
    if (in_et != compute_et) {
        A = context.mark_node(std::make_shared<v0::Convert>(x, compute_et));
    }

    // Fast path: statically-known 3x3 trailing dims (the SAM-6D Kabsch regime) use the unrolled 3x3
    // Jacobi (no v5::Loop). Any other/dynamic size uses the general NxN Jacobi (Loop).
    const auto& ps = x.get_partial_shape();
    const auto rank = ps.rank();
    bool static_3x3 = false;
    if (rank.is_static() && rank.get_length() >= 2) {
        const auto& m_dim = ps[rank.get_length() - 2];
        const auto& n_dim = ps[rank.get_length() - 1];
        static_3x3 = m_dim.is_static() && n_dim.is_static() && m_dim.get_length() == 3 && n_dim.get_length() == 3;
    }

    Output<Node> U, S, V;
    if (static_3x3) {
        JacobiSvd3x3 builder(context, compute_et);
        std::tie(U, S, V) = builder.build(A);
    } else {
        std::tie(U, S, V) = jacobi_svd(context, A, compute_et);
    }

    // aten::linalg_svd returns Vh = V^T (right singular vectors as rows); aten::svd returns V.
    if (return_vh) {
        V = transpose_last2(context, V);
    }

    // Cast outputs back to the input type. ConvertLike resolves it from `x` even when `x`'s type
    // is dynamic at conversion time (a direct Convert would fail).
    if (in_et != compute_et) {
        U = context.mark_node(std::make_shared<v1::ConvertLike>(U, x));
        S = context.mark_node(std::make_shared<v1::ConvertLike>(S, x));
        V = context.mark_node(std::make_shared<v1::ConvertLike>(V, x));
    }
    return {U, S, V};
}

}  // namespace

OutputVector translate_svd(const NodeContext& context) {
    // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
    // PyTorch zero-fills U/V when compute_uv=False, but this translator always produces real U/V,
    // so reject a constant-false (or non-constant, possibly-false) compute_uv. input_is_none checks
    // the input size internally, so no separate get_input_size guard is needed.
    if (!context.input_is_none(2)) {
        const auto c = ov::util::get_constant_from_source(context.get_input(2));
        PYTORCH_OP_CONVERSION_CHECK(c, "aten::svd with a non-constant compute_uv is not supported.");
        const auto vals = c->cast_vector<bool>();
        PYTORCH_OP_CONVERSION_CHECK(vals.empty() || vals[0], "aten::svd with compute_uv=False is not supported.");
    }
    return svd_common(context, /*return_vh=*/false);
};

OutputVector translate_linalg_svd(const NodeContext& context) {
    // aten::linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (U, S, Vh)
    // Square-only (rectangular A fails the squareness guard); for a square matrix full_matrices makes
    // no shape difference (U, Vh are always N x N), so the flag is accepted but not read. Vh = V^T of
    // the torch.svd convention.
    return svd_common(context, /*return_vh=*/true);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
