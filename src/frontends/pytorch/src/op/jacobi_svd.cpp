// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/jacobi_svd.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/eye.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

JacobiLoopResult run_jacobi_column_loop(const NodeContext& context,
                                        const Output<Node>& A_flat,
                                        const Output<Node>& n,
                                        int sweeps,
                                        element::Type et,
                                        bool accumulate_v) {
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
    auto sub = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Subtract>(a, b));
    };
    auto div = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::Divide>(a, b));
    };

    auto n_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(n, i64_c({0})));  // [cols]
    auto b_dim = context.mark_node(
        std::make_shared<v8::Slice>(context.mark_node(std::make_shared<v3::ShapeOf>(A_flat, element::i64)),
                                    i64_c({0}),
                                    i64_c({1}),
                                    i64_c({1})));  // [B]

    // Identity V (B, cols, cols) to accumulate the right rotations (only when requested).
    Output<Node> V_flat;
    if (accumulate_v) {
        auto eye = context.mark_node(std::make_shared<v9::Eye>(n, n, i64_s(0), et));  // (cols, cols)
        auto v_bshape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{b_dim, n_1d, n_1d}, 0));
        V_flat = context.mark_node(
            std::make_shared<v3::Broadcast>(context.mark_node(std::make_shared<v0::Unsqueeze>(eye, i64_c({0}))),
                                            v_bshape));
    }

    // Upper-triangle pair list (p < q): NonZero of the (cols, cols) strictly-upper mask, tiled
    // `sweeps` times into one flat schedule of length sweeps * cols(cols-1)/2.
    auto seq = context.mark_node(std::make_shared<v4::Range>(i64_s(0), n, i64_s(1), element::i64));  // (cols,)
    auto ii = context.mark_node(std::make_shared<v0::Unsqueeze>(seq, i64_c({1})));                   // (cols, 1)
    auto jj = context.mark_node(std::make_shared<v0::Unsqueeze>(seq, i64_c({0})));                   // (1, cols)
    auto upper = context.mark_node(std::make_shared<v1::Less>(ii, jj));                         // (cols, cols) p < q
    auto coords = context.mark_node(std::make_shared<v3::NonZero>(upper, element::i64));        // (2, P)
    auto p_list = context.mark_node(std::make_shared<v8::Gather>(coords, i64_s(0), i64_s(0)));  // (P,)
    auto q_list = context.mark_node(std::make_shared<v8::Gather>(coords, i64_s(1), i64_s(0)));  // (P,)
    auto sweeps_c = i64_c({sweeps});
    auto p_all = context.mark_node(std::make_shared<v0::Tile>(p_list, sweeps_c));  // (sweeps*P,)
    auto q_all = context.mark_node(std::make_shared<v0::Tile>(q_list, sweeps_c));
    auto total = context.mark_node(
        std::make_shared<v8::Gather>(context.mark_node(std::make_shared<v3::ShapeOf>(p_all, element::i64)),
                                     i64_s(0),
                                     i64_s(0)));  // scalar sweeps*P

    // ---- Loop body: one Givens rotation of columns p, q of A (and V) ----
    auto A_b = std::make_shared<v0::Parameter>(et, PartialShape{-1, -1, -1});
    std::shared_ptr<v0::Parameter> V_b;
    if (accumulate_v) {
        V_b = std::make_shared<v0::Parameter>(et, PartialShape{-1, -1, -1});
    }
    auto p_s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto q_s = std::make_shared<v0::Parameter>(element::i64, PartialShape{1});
    auto p = std::make_shared<v0::Squeeze>(p_s, i64_c({0}));  // scalar column index p
    auto q = std::make_shared<v0::Squeeze>(q_s, i64_c({0}));  // scalar column index q

    auto ax_col = i64_s(2);
    auto ap = std::make_shared<v8::Gather>(A_b, p, ax_col);  // (B, rows) column p
    auto aq = std::make_shared<v8::Gather>(A_b, q, ax_col);  // (B, rows) column q
    auto dot = [&](const Output<Node>& a, const Output<Node>& b) {
        return context.mark_node(std::make_shared<v1::ReduceSum>(mul(a, b), i64_c({1}), true));  // (B, 1)
    };
    // Jacobi angle from alpha=<ap,ap>, beta=<aq,aq>, gamma=<ap,aq>.
    auto alpha = dot(ap, ap);
    auto beta = dot(aq, aq);
    auto gamma = dot(ap, aq);
    auto tiny = fc(1e-30f);
    // zeta = (beta - alpha) / (2 gamma), with a guarded divisor and a clamp so zeta^2 stays finite.
    auto denom = mul(fc(2.0f), gamma);
    auto is_orth = context.mark_node(std::make_shared<v1::Equal>(gamma, fc(0.0f)));
    auto safe_denom = context.mark_node(std::make_shared<v1::Select>(is_orth, tiny, denom));
    auto zeta = div(sub(beta, alpha), safe_denom);
    zeta = context.mark_node(std::make_shared<v0::Clamp>(zeta, -1e18, 1e18));
    // t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)); sign(0) = +1 so equal-norm non-orthogonal
    // columns still get the +-45 degree rotation (v0::Sign(0) = 0 would skip it).
    auto abs_zeta = context.mark_node(std::make_shared<v0::Abs>(zeta));
    auto sgn = context.mark_node(std::make_shared<v0::Sign>(zeta));
    auto is_zero = context.mark_node(std::make_shared<v1::Equal>(zeta, fc(0.0f)));
    auto sign = context.mark_node(std::make_shared<v1::Select>(is_zero, fc(1.0f), sgn));
    auto t = mul(
        sign,
        div(fc(1.0f), add(abs_zeta, context.mark_node(std::make_shared<v0::Sqrt>(add(fc(1.0f), mul(zeta, zeta)))))));
    auto cc = div(fc(1.0f), context.mark_node(std::make_shared<v0::Sqrt>(add(fc(1.0f), mul(t, t)))));
    auto ss = mul(cc, t);

    auto rot_p = [&](const Output<Node>& cp, const Output<Node>& cq) {
        return sub(mul(cc, cp), mul(ss, cq));
    };
    auto rot_q = [&](const Output<Node>& cp, const Output<Node>& cq) {
        return add(mul(ss, cp), mul(cc, cq));
    };
    auto na_p = rot_p(ap, aq);
    auto na_q = rot_q(ap, aq);

    // Scatter the two rotated columns back on the column axis (-1).
    auto pq = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{context.mark_node(std::make_shared<v0::Unsqueeze>(p, i64_c({0}))),
                                                  context.mark_node(std::make_shared<v0::Unsqueeze>(q, i64_c({0})))},
                                     0));  // (2,)
    auto pack = [&](const Output<Node>& cp, const Output<Node>& cq) {
        return context.mark_node(std::make_shared<v0::Concat>(
            OutputVector{context.mark_node(std::make_shared<v0::Unsqueeze>(cp, i64_c({2}))),
                         context.mark_node(std::make_shared<v0::Unsqueeze>(cq, i64_c({2})))},
            2));  // (B, rows, 2)
    };
    auto A_new = context.mark_node(std::make_shared<v3::ScatterUpdate>(A_b, pq, pack(na_p, na_q), ax_col));

    auto body_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    OutputVector body_results{body_cond, A_new};
    ParameterVector body_params{A_b};
    std::shared_ptr<Node> V_new;
    if (accumulate_v) {
        auto vp = std::make_shared<v8::Gather>(V_b, p, ax_col);
        auto vq = std::make_shared<v8::Gather>(V_b, q, ax_col);
        V_new =
            context.mark_node(std::make_shared<v3::ScatterUpdate>(V_b, pq, pack(rot_p(vp, vq), rot_q(vp, vq)), ax_col));
        body_results.push_back(V_new);
        body_params.push_back(V_b);
    }
    body_params.push_back(p_s);
    body_params.push_back(q_s);
    auto body = std::make_shared<Model>(body_results, body_params);

    auto exec_cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
    auto loop = std::make_shared<v5::Loop>(total, exec_cond);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(A_b, A_flat, A_new);
    if (accumulate_v) {
        loop->set_merged_input(V_b, V_flat, V_new);
    }
    loop->set_sliced_input(p_s, p_all, 0, 1, 1, 0, 0);
    loop->set_sliced_input(q_s, q_all, 0, 1, 1, 0, 0);
    // Keep each get_iter_value as an Output (with its port index): reading the node instead would
    // collapse both A and V to port 0.
    JacobiLoopResult result;
    result.a = loop->get_iter_value(A_new, -1);  // (B, rows, cols)
    if (accumulate_v) {
        result.v = loop->get_iter_value(V_new, -1);  // (B, cols, cols)
    }
    context.mark_node(loop);
    return result;
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
