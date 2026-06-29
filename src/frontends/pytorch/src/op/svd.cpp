// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
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

// Number of Jacobi sweeps. One-sided Jacobi for 3x3 converges in ~5 sweeps; 6 adds
// margin (matches torch.svd to ~1e-5 across many seeds, including rank-deficient
// matrices).
constexpr int SVD_SWEEPS = 6;
// Column index pairs swept by one-sided Jacobi.
constexpr int PAIRS[3][2] = {{0, 1}, {0, 2}, {1, 2}};

// Builds an OpenVINO subgraph computing the SVD of batched 3x3 matrices via the
// ONE-SIDED JACOBI method. Unlike the normal-equations (A^T A eigendecomposition)
// approach, Jacobi operates on A directly and stays fp32-accurate for rank-
// deficient matrices (e.g. the 3-point Weighted-Procrustes / Kabsch matrices used
// in SAM-6D pose estimation). It orthogonalizes the columns of A by a fixed number
// of sweeps of Jacobi rotations (accumulated into V); afterwards the singular
// values are the column norms and U = A_col / sigma. Columns are sorted by
// descending singular value to match torch's convention.
class JacobiSvd3x3 {
public:
    JacobiSvd3x3(const NodeContext& context, element::Type et)
        : m_ctx(context),
          m_et(et),
          m_tiny(cf(1e-30f)) {}

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
            sig[k] = sqrt(dot(a[k], a[k]));      // (...,1)
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
        auto alpha = dot(ap, ap);   // (...,1,1)
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
    void cmp_exchange(Output<Node>& sp, Output<Node>& sq,
                      Output<Node>& up, Output<Node>& uq,
                      Output<Node>& vp, Output<Node>& vq) {
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
        // |x| via Select(x < 0, -x, x); avoids pulling in an Abs op and keeps full
        // precision (no sqrt rounding).
        auto neg = mul(x, cf(-1.0f));
        return sel(m_ctx.mark_node(std::make_shared<v1::Less>(x, cf(0.0f))), neg, x);
    }
    Output<Node> signum(const Output<Node>& x) {
        // sign with sign(0) = +1 (the value is irrelevant when gamma ~ 0).
        return sel(m_ctx.mark_node(std::make_shared<v1::Less>(x, cf(0.0f))), cf(-1.0f), cf(1.0f));
    }

    // Stack three (...,3,1) column vectors into a (...,3,3) matrix (columns).
    Output<Node> colstack(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c) {
        return m_ctx.mark_node(std::make_shared<v0::Concat>(OutputVector{a, b, c}, -1));
    }

    const NodeContext& m_ctx;
    element::Type m_et;
    Output<Node> m_tiny;
};

// The Jacobi decomposition below is written for 3x3 matrices. When the trailing
// matrix dimensions are statically known, reject anything that is not 3x3 with a
// clear conversion-time message. When they are dynamic (on the TorchScript path the
// decoder forces all dims dynamic, so this is the common case), the size cannot be
// checked at conversion time; instead insert a runtime square-3x3 guard: reshape
// the trailing two axes to a fixed [3, 3] while preserving the batch axes
// (new_shape = concat(shape_of(x)[:-2], [3, 3])). For a genuine 3x3 input this is an
// identity; any other size cannot match the element count and raises a runtime
// Reshape error -- turning an otherwise silent wrong result into a loud failure. 3x3
// is the supported / expected runtime size (e.g. the Kabsch rigid-transform block in
// pose-estimation models). Returns the (possibly reshape-guarded) matrix to use.
Output<Node> check_square_3x3(const NodeContext& context, const Output<Node>& x) {
    const auto& ps = x.get_partial_shape();
    const auto rank = ps.rank();
    if (rank.is_static() && rank.get_length() >= 2) {
        auto n = ps[rank.get_length() - 1];
        auto m = ps[rank.get_length() - 2];
        if (n.is_static() && m.is_static()) {
            PYTORCH_OP_CONVERSION_CHECK(
                n.get_length() == 3 && m.get_length() == 3,
                "aten::svd is only supported for 3x3 matrices, got trailing dimensions ",
                m.get_length(),
                "x",
                n.get_length(),
                ".");
            return x;
        }
    }
    // Dynamic trailing dimension(s): guard the assumed 3x3 size at runtime.
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i64));
    auto start = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto stop = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto step = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto batch_shape = context.mark_node(std::make_shared<v8::Slice>(shape, start, stop, step, axis));
    auto nn = context.mark_node(v0::Constant::create(element::i64, Shape{2}, {3, 3}));
    auto new_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{batch_shape, nn}, 0));
    return context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, /*special_zero=*/false));
}

OutputVector svd_common(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    // Runtime/conversion-time 3x3 guard; on the dynamic path this returns x wrapped
    // in a reshape-to-[...,3,3] so a non-3x3 input fails loudly at runtime.
    x = check_square_3x3(context, x);

    auto in_et = x.get_element_type();
    // Compute at least in f32 (f16/bf16 are too coarse for the Jacobi rotations).
    // f64 is preserved when statically requested. When the input type is dynamic
    // at conversion time, default the compute type to f32.
    auto compute_et = (in_et == element::f64) ? element::f64 : element::f32;
    Output<Node> A = x;
    if (in_et != compute_et) {
        A = context.mark_node(std::make_shared<v0::Convert>(x, compute_et));
    }

    JacobiSvd3x3 builder(context, compute_et);
    Output<Node> U, S, V;
    std::tie(U, S, V) = builder.build(A);

    // Cast the outputs back to the input element type. ConvertLike resolves the
    // target type from `x` even when it is dynamic at conversion time (a direct
    // Convert to a dynamic type would fail at runtime).
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
    // With compute_uv=False PyTorch returns zero-filled U and V (only S is meaningful); this
    // translator always produces real U/V, so reject a statically-false compute_uv loudly rather
    // than returning non-zero singular vectors a caller would treat as zeros. Probe without
    // throwing so a (non-constant) runtime flag does not break conversion.
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        if (const auto c = ov::util::get_constant_from_source(context.get_input(2))) {
            const auto vals = c->cast_vector<bool>();
            PYTORCH_OP_CONVERSION_CHECK(vals.empty() || vals[0],
                                        "aten::svd with compute_uv=False is not supported.");
        }
    }
    return svd_common(context);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
