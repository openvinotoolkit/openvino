// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
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
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

// One-sided Jacobi for 3x3 converges in ~5 sweeps; 6 adds margin (matches torch.svd to ~1e-5,
// incl. rank-deficient matrices).
constexpr int SVD_SWEEPS = 6;
// Column index pairs swept by one-sided Jacobi.
constexpr int PAIRS[3][2] = {{0, 1}, {0, 2}, {1, 2}};

// SVD of batched 3x3 matrices via ONE-SIDED JACOBI. Unlike normal equations (A^T A
// eigendecomposition), it operates on A directly and stays fp32-accurate for rank-deficient
// matrices (the 3-point Weighted-Procrustes / Kabsch matrices in SAM-6D pose estimation): it
// orthogonalizes A's columns over a fixed number of rotation sweeps (accumulated into V); the
// singular values are then the column norms and U = A_col / sigma, sorted descending like torch.
class JacobiSvd3x3 {
public:
    JacobiSvd3x3(const NodeContext& context, element::Type et) : m_ctx(context), m_et(et), m_tiny(cf(1e-30f)) {}

    // Returns {U, S, V} with shapes U:(...,3,3), S:(...,3) descending, V:(...,3,3).
    // Each column of U/V is a (...,3) singular vector; V holds the right singular
    // vectors as columns (the torch.svd convention).
    std::tuple<Output<Node>, Output<Node>, Output<Node>> build(const Output<Node>& A_in) {
        // Working columns of A and of V (V starts as identity columns).
        Output<Node> a[3] = {mcol(A_in, 0), mcol(A_in, 1), mcol(A_in, 2)};
        Output<Node> v[3] = {ident_col(A_in, 0), ident_col(A_in, 1), ident_col(A_in, 2)};

        for (int sweep = 0; sweep < SVD_SWEEPS; ++sweep) {
            for (auto& pr : PAIRS) {
                jacobi_rotate(a[pr[0]], a[pr[1]], v[pr[0]], v[pr[1]]);
            }
        }

        // Singular values = column norms; left singular vectors = normalized columns.
        Output<Node> sig[3], u[3];
        for (int k = 0; k < 3; ++k) {
            sig[k] = sqrt(dot(a[k], a[k]));  // (...,1,1)
            u[k] = div(a[k], add(sig[k], m_tiny));
        }

        // Sort columns by descending singular value with a 3-element sorting
        // network of compare-exchanges: (0,1),(1,2),(0,1).
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
    // --- constant helpers ---
    Output<Node> cf(float v) {
        return m_ctx.mark_node(v0::Constant::create(m_et, Shape{}, {v}));
    }
    static std::shared_ptr<v0::Constant> ci(int64_t v) {
        return v0::Constant::create(element::i32, Shape{1}, {v});
    }
    static std::shared_ptr<v0::Constant> ci_s(int64_t v) {
        return v0::Constant::create(element::i32, Shape{}, {v});
    }

    // --- elementwise wrappers ---
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

    // Column k of a (...,3,3) matrix as a (...,3,1) keepdim vector (so the
    // trailing reduce/broadcast ops keep their last axis = vector component).
    Output<Node> mcol(const Output<Node>& m, int64_t k) {
        return m_ctx.mark_node(std::make_shared<v8::Gather>(m, ci(k), ci_s(-1)));  // (...,3,1)
    }
    // Same as mcol but for the V columns of an identity-like start. We build the
    // identity columns from constants broadcast to the batch of `like`.
    Output<Node> ident_col(const Output<Node>& like, int64_t k) {
        std::vector<float> e(3, 0.0f);
        e[static_cast<size_t>(k)] = 1.0f;
        auto col = m_ctx.mark_node(v0::Constant::create(m_et, Shape{3, 1}, e));
        // Broadcast to batch via add with 0*like-column (keeps the (...,3,1) shape).
        auto zero_like = mul(mcol(like, 0), cf(0.0f));
        return add(col, zero_like);
    }
    // Reduce over the vector axis (-2) keeping it: dot of two (...,3,1) vectors -> (...,1,1).
    Output<Node> dot(const Output<Node>& a, const Output<Node>& b) {
        return m_ctx.mark_node(std::make_shared<v1::ReduceSum>(mul(a, b), ci_s(-2), true));
    }

    // One Jacobi rotation that orthogonalizes columns (ap, aq); the same rotation
    // is applied to (vp, vq). All operands are (...,3,1) vectors.
    void jacobi_rotate(Output<Node>& ap, Output<Node>& aq, Output<Node>& vp, Output<Node>& vq) {
        auto alpha = dot(ap, ap);  // (...,1,1)
        auto beta = dot(aq, aq);
        auto gamma = dot(ap, aq);
        auto denom = mul(cf(2.0f), gamma);
        // zeta = (beta - alpha) / (2 gamma); guard the divide and clamp to keep
        // zeta*zeta finite in fp32.
        auto zeta = div(sub(beta, alpha), add(denom, m_tiny));
        zeta = m_ctx.mark_node(std::make_shared<v0::Clamp>(zeta, -1e18, 1e18));
        auto azeta = absval(zeta);
        // t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)); |t| <= 1 (overflow-free).
        auto t_mag = div(cf(1.0f), add(azeta, sqrt(add(cf(1.0f), mul(zeta, zeta)))));
        auto t = mul(signum(zeta), t_mag);
        // If the columns are already orthogonal (gamma ~ 0), use t = 0 (identity).
        auto orthogonal = m_ctx.mark_node(std::make_shared<v1::Less>(absval(gamma), m_tiny));
        t = sel(orthogonal, cf(0.0f), t);
        auto c = div(cf(1.0f), sqrt(add(cf(1.0f), mul(t, t))));
        auto s = mul(c, t);
        // Apply the rotation to A columns and V columns.
        rotate_pair(ap, aq, c, s);
        rotate_pair(vp, vq, c, s);
    }

    void rotate_pair(Output<Node>& p, Output<Node>& q, const Output<Node>& c, const Output<Node>& s) {
        auto np = sub(mul(c, p), mul(s, q));
        auto nq = add(mul(s, p), mul(c, q));
        p = np;
        q = nq;
    }

    // Compare-exchange so that sig_p >= sig_q after the call, swapping the
    // associated U and V columns accordingly. sig_* are (...,1,1); cols are (...,3,1).
    void cmp_exchange(Output<Node>& sp,
                      Output<Node>& sq,
                      Output<Node>& up,
                      Output<Node>& uq,
                      Output<Node>& vp,
                      Output<Node>& vq) {
        auto swap = m_ctx.mark_node(std::make_shared<v1::Less>(sp, sq));  // swap if sp < sq
        // swap broadcast for (...,3,1) vectors uses the same (...,1,1) mask.
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
        // sign(x) with sign(0) = +1. v0::Sign(0) = 0 would set t = 0 and skip the +-45 degree
        // rotation exactly when two equal-norm columns are non-orthogonal (zeta == 0), leaving
        // rank-deficient columns un-orthogonalized. Force +1 at 0 so the rotation still fires.
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

OutputVector svd_common(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    // The Jacobi decomposition below is written for 3x3; ensure_trailing_square validates/guards
    // the trailing axes to 3x3 (a non-3x3 input fails at conversion or loudly at runtime).
    x = ensure_trailing_square(context, x, 3, "aten::svd");

    auto in_et = x.get_element_type();
    // Compute in f32 (f16/bf16 are too coarse for the rotations); keep f64 when statically requested.
    auto compute_et = (in_et == element::f64) ? element::f64 : element::f32;
    Output<Node> A = x;
    if (in_et != compute_et) {
        A = context.mark_node(std::make_shared<v0::Convert>(x, compute_et));
    }

    JacobiSvd3x3 builder(context, compute_et);
    Output<Node> U, S, V;
    std::tie(U, S, V) = builder.build(A);

    // Cast outputs back to the input type. ConvertLike resolves it from `x` even when it is
    // dynamic at conversion time (a direct Convert to a dynamic type would fail).
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
    // PyTorch returns zero-filled U/V when compute_uv=False; this translator always produces real
    // U/V, so compute_uv must be a constant true. A constant false, or a non-constant flag that
    // could be false at runtime, is rejected -- otherwise a caller expecting zeros gets real vectors.
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        const auto c = ov::util::get_constant_from_source(context.get_input(2));
        PYTORCH_OP_CONVERSION_CHECK(c, "aten::svd with a non-constant compute_uv is not supported.");
        const auto vals = c->cast_vector<bool>();
        PYTORCH_OP_CONVERSION_CHECK(vals.empty() || vals[0], "aten::svd with compute_uv=False is not supported.");
    }
    return svd_common(context);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
