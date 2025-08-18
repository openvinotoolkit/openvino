// Copyright (C) 2018-2025 Intel
// SPDX-License-Identifier: Apache-2.0
//
// GridSample decomposition split into STATIC and DYNAMIC shape paths.
// - No use of ov::reference::* (explicitly forbidden).
// - Static path builds a smaller, const-shape graph (no ShapeOf/Gather over shapes).
// - Dynamic path is the original fully-generic builder over ShapeOf/Gather.
// - Three interpolation modes are supported: NEAREST, BILINEAR, BICUBIC.
// - Padding modes: ZEROS / BORDER / REFLECTION. align_corners is respected.
//

#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

// ============================================================================
// INTERNAL SHARED: graph-level builders (reused by both static & dynamic paths)
// ============================================================================

namespace {

struct Ctx {
    // Scalars with sizes (might be Constants in static path or ShapeOf-slices in dynamic)
    std::shared_ptr<Node> n_dim, c_dim, h_in, w_in, h_out, w_out;
    // Types
    element::Type element_type;
    element::Type calc_type;  // f32 for math
    // Float constants
    std::shared_ptr<Node> c0, c1, c2, c0_5;
    // Integer constants
    std::shared_ptr<Node> i32_0, i32_1, i32_2, i32_3, i32_neg1;
    // Common axes
    std::shared_ptr<Node> axis_0, axis_3;
    // NHWC transpose pattern
    std::shared_ptr<Node> to_nhwc_perm;
    // Pixels coords after normalization
    std::shared_ptr<Node> x, y;
    // Normalized coords (from grid, in [-1, 1])
    std::shared_ptr<Node> x_norm, y_norm;
    // Data as NHWC for GatherND
    std::shared_ptr<Node> data_nhwc;
};

// ---- small helpers ----
std::shared_ptr<Node> to_f32(const std::shared_ptr<Node>& i) {
    return std::make_shared<op::v0::Convert>(i, element::f32);
}

// Normalize grid [-1, 1] to pixel coordinates (no padding/reflection here)
void normalize_grid_to_pixels(const Ctx& ctx,
                              const std::shared_ptr<Node>& x_grid,
                              const std::shared_ptr<Node>& y_grid,
                              const std::shared_ptr<Node>& w_in_f,
                              const std::shared_ptr<Node>& h_in_f,
                              bool align_corners,
                              std::shared_ptr<Node>& x_out,
                              std::shared_ptr<Node>& y_out) {
    if (align_corners) {
        auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
        auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
        x_out = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(x_grid, ctx.c1), ctx.c2),
            w_m1);
        y_out = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(y_grid, ctx.c1), ctx.c2),
            h_m1);
    } else {
        // ((t + 1) * size - 1) / 2
        auto x_p1_w = std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(x_grid, ctx.c1), w_in_f);
        auto y_p1_h = std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(y_grid, ctx.c1), h_in_f);
        x_out = std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Subtract>(x_p1_w, ctx.c1), ctx.c2);
        y_out = std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Subtract>(y_p1_h, ctx.c1), ctx.c2);
    }
}

// Create [N, H_out, W_out] batch indices tensor for GatherND (broadcasted)
std::shared_ptr<Node> create_batch_indices(const Ctx& ctx) {
    auto range = std::make_shared<op::v4::Range>(ctx.i32_0,
                                                 ctx.n_dim,
                                                 ctx.i32_1,
                                                 element::i32);

    auto n_dim_1d_bs = std::make_shared<op::v0::Unsqueeze>(ctx.n_dim, ctx.i32_0);
    auto batch_shape = std::make_shared<op::v0::Concat>(OutputVector{n_dim_1d_bs,
                                                                     op::v0::Constant::create(element::i32, {1}, {1}),
                                                                     op::v0::Constant::create(element::i32, {1}, {1})},
                                                        0);

    auto batch_indices = std::make_shared<op::v1::Reshape>(range, batch_shape, false);

    auto n_dim_1d = std::make_shared<op::v0::Unsqueeze>(ctx.n_dim, ctx.i32_0);
    auto h_out_1d = std::make_shared<op::v0::Unsqueeze>(ctx.h_out, ctx.i32_0);
    auto w_out_1d = std::make_shared<op::v0::Unsqueeze>(ctx.w_out, ctx.i32_0);
    auto bshape = std::make_shared<op::v0::Concat>(OutputVector{n_dim_1d, h_out_1d, w_out_1d}, 0);
    auto bcast = std::make_shared<op::v3::Broadcast>(batch_indices, bshape);
    return std::make_shared<op::v0::Convert>(bcast, element::i32);
}

// Gather from NHWC by (b, y, x)
std::shared_ptr<Node> gather_hw_nhwc(const Ctx& ctx,
                                     const std::shared_ptr<Node>& x_coord,
                                     const std::shared_ptr<Node>& y_coord) {
    auto x_i32 = std::make_shared<op::v0::Convert>(x_coord, element::i32);
    auto y_i32 = std::make_shared<op::v0::Convert>(y_coord, element::i32);
    auto bidx = create_batch_indices(ctx);

    auto b_exp = std::make_shared<op::v0::Unsqueeze>(bidx, ctx.i32_neg1);
    auto y_exp = std::make_shared<op::v0::Unsqueeze>(y_i32, ctx.i32_neg1);
    auto x_exp = std::make_shared<op::v0::Unsqueeze>(x_i32, ctx.i32_neg1);

    auto indices = std::make_shared<op::v0::Concat>(OutputVector{b_exp, y_exp, x_exp}, -1);
    return std::make_shared<op::v8::GatherND>(ctx.data_nhwc, indices, 0);
}

// Reflection helpers (continuous/indexed)
std::shared_ptr<Node> reflect_coord(const Ctx& ctx,
                                    const std::shared_ptr<Node>& coord,
                                    const std::shared_ptr<Node>& size_f,
                                    bool align_corners) {
    auto size_is_one = std::make_shared<op::v1::Equal>(size_f, ctx.c1);
    if (align_corners) {
        auto size_m1 = std::make_shared<op::v1::Subtract>(size_f, ctx.c1);
        auto period = std::make_shared<op::v1::Multiply>(ctx.c2, size_m1);
        auto period_ok = std::make_shared<op::v1::Maximum>(period, ctx.c1);
        auto r = std::make_shared<op::v1::FloorMod>(coord, period_ok);
        auto d = std::make_shared<op::v1::Subtract>(r, size_m1);
        auto ad = std::make_shared<op::v0::Abs>(d);
        auto ref = std::make_shared<op::v1::Subtract>(size_m1, ad);
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, ref);
    }
    auto is_negative = std::make_shared<op::v1::Less>(coord, ctx.c0);
    auto neg_coord = std::make_shared<op::v0::Negative>(coord);
    auto neg_minus_one = std::make_shared<op::v1::Subtract>(neg_coord, ctx.c1);
    auto coord_positive = std::make_shared<op::v1::Select>(is_negative, neg_minus_one, coord);

    auto two_size = std::make_shared<op::v1::Multiply>(ctx.c2, size_f);
    auto two_size_safe = std::make_shared<op::v1::Maximum>(two_size, ctx.c1);
    auto mod = std::make_shared<op::v1::FloorMod>(coord_positive, two_size_safe);

    auto need_ref = std::make_shared<op::v1::GreaterEqual>(mod, size_f);
    auto two_size_m1 = std::make_shared<op::v1::Subtract>(two_size_safe, ctx.c1);
    auto ref_val = std::make_shared<op::v1::Subtract>(two_size_m1, mod);
    auto result = std::make_shared<op::v1::Select>(need_ref, ref_val, mod);

    return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, result);
}

std::shared_ptr<Node> reflect_index(const Ctx& ctx,
                                    const std::shared_ptr<Node>& idx,
                                    const std::shared_ptr<Node>& size_f,
                                    bool align_corners) {
    auto size_is_one = std::make_shared<op::v1::Equal>(size_f, ctx.c1);
    if (align_corners) {
        auto size_m1 = std::make_shared<op::v1::Subtract>(size_f, ctx.c1);
        auto two_sm1 = std::make_shared<op::v1::Multiply>(ctx.c2, size_m1);
        auto period_safe = std::make_shared<op::v1::Maximum>(two_sm1, ctx.c1);
        auto mod1 = std::make_shared<op::v1::FloorMod>(idx, period_safe);
        auto mod2 = std::make_shared<op::v1::FloorMod>(std::make_shared<op::v1::Add>(mod1, period_safe), period_safe);
        auto flipped = std::make_shared<op::v1::Subtract>(two_sm1, mod2);
        auto cond = std::make_shared<op::v1::GreaterEqual>(mod2, size_f);
        auto ref = std::make_shared<op::v1::Select>(cond, flipped, mod2);
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, ref);
    }
    auto two_s = std::make_shared<op::v1::Multiply>(ctx.c2, size_f);
    auto two_s_safe = std::make_shared<op::v1::Maximum>(two_s, ctx.c1);
    auto mod1 = std::make_shared<op::v1::FloorMod>(idx, two_s_safe);
    auto mod2 = std::make_shared<op::v1::FloorMod>(std::make_shared<op::v1::Add>(mod1, two_s_safe), two_s_safe);
    auto need_rf = std::make_shared<op::v1::GreaterEqual>(mod2, size_f);
    auto two_s_m1 = std::make_shared<op::v1::Subtract>(two_s_safe, ctx.c1);
    auto ref_val = std::make_shared<op::v1::Subtract>(two_s_m1, mod2);
    auto res = std::make_shared<op::v1::Select>(need_rf, ref_val, mod2);
    return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, res);
}

// Clamp integer-ish indices to [0..size-1] (safe)
void clamp_indices_inplace(const Ctx& ctx,
                           std::shared_ptr<Node>& xi,
                           std::shared_ptr<Node>& yi,
                           const std::shared_ptr<Node>& w_in_f,
                           const std::shared_ptr<Node>& h_in_f) {
    auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
    auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
    xi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(xi, ctx.c0), w_m1);
    yi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(yi, ctx.c0), h_m1);
}

// Inside mask for ZEROS padding (check BEFORE clamp)
std::shared_ptr<Node> inside_mask_indexed(const Ctx& ctx,
                                          const std::shared_ptr<Node>& xi,
                                          const std::shared_ptr<Node>& yi,
                                          const std::shared_ptr<Node>& w_in_f,
                                          const std::shared_ptr<Node>& h_in_f) {
    auto x_ge_0 = std::make_shared<op::v1::GreaterEqual>(xi, ctx.c0);
    auto y_ge_0 = std::make_shared<op::v1::GreaterEqual>(yi, ctx.c0);
    auto x_lt_w = std::make_shared<op::v1::Less>(xi, w_in_f);
    auto y_lt_h = std::make_shared<op::v1::Less>(yi, h_in_f);
    return std::make_shared<op::v1::LogicalAnd>(std::make_shared<op::v1::LogicalAnd>(x_ge_0, x_lt_w),
                                                std::make_shared<op::v1::LogicalAnd>(y_ge_0, y_lt_h));
}

// ---- builders for interpolation modes ----

// NEAREST (HALF_TO_EVEN rounding; reflect after rounding; always clamp)
std::shared_ptr<Node> build_nearest_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_zeros = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);

    std::shared_ptr<Node> x_idx = std::make_shared<op::v5::Round>(ctx.x, op::v5::Round::RoundMode::HALF_TO_EVEN);
    std::shared_ptr<Node> y_idx = std::make_shared<op::v5::Round>(ctx.y, op::v5::Round::RoundMode::HALF_TO_EVEN);

    std::shared_ptr<Node> x_unclamped = x_idx;
    std::shared_ptr<Node> y_unclamped = y_idx;

    if (is_reflection) {
        x_idx = reflect_index(ctx, x_idx, w_in_f, attrs.align_corners);
        y_idx = reflect_index(ctx, y_idx, h_in_f, attrs.align_corners);
    }
    clamp_indices_inplace(ctx, x_idx, y_idx, w_in_f, h_in_f);

    std::shared_ptr<Node> out = gather_hw_nhwc(ctx, x_idx, y_idx);

    if (is_zeros) {
        auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
        auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
        auto x_ge_0 = std::make_shared<op::v1::GreaterEqual>(x_unclamped, ctx.c0);
        auto x_le_w1 = std::make_shared<op::v1::LessEqual>(x_unclamped, w_m1);
        auto y_ge_0 = std::make_shared<op::v1::GreaterEqual>(y_unclamped, ctx.c0);
        auto y_le_h1 = std::make_shared<op::v1::LessEqual>(y_unclamped, h_m1);
        auto ok = std::make_shared<op::v1::LogicalAnd>(std::make_shared<op::v1::LogicalAnd>(x_ge_0, x_le_w1),
                                                       std::make_shared<op::v1::LogicalAnd>(y_ge_0, y_le_h1));
        auto mask = std::make_shared<op::v0::Convert>(ok, ctx.calc_type);
        auto maske = std::make_shared<op::v0::Unsqueeze>(mask, ctx.i32_neg1);
        if (ctx.element_type != ctx.calc_type) {
            out = std::make_shared<op::v0::Convert>(out, ctx.calc_type);
        }
        out = std::make_shared<op::v1::Multiply>(out, maske);
        if (ctx.element_type != ctx.calc_type) {
            out = std::make_shared<op::v0::Convert>(out, ctx.element_type);
        }
    }

    return out;
}

// BILINEAR
std::shared_ptr<Node> build_bilinear_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_border = attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER;
    const bool is_zeros = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);

    std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
    if (is_reflection) {
        x_pad = reflect_coord(ctx, ctx.x, w_in_f, attrs.align_corners);
        y_pad = reflect_coord(ctx, ctx.y, h_in_f, attrs.align_corners);
    }

    std::shared_ptr<Node> x0 = std::make_shared<op::v0::Floor>(x_pad);
    std::shared_ptr<Node> y0 = std::make_shared<op::v0::Floor>(y_pad);
    std::shared_ptr<Node> x1 = std::make_shared<op::v1::Add>(x0, ctx.c1);
    std::shared_ptr<Node> y1 = std::make_shared<op::v1::Add>(y0, ctx.c1);

    auto dx = std::make_shared<op::v1::Subtract>(x_pad, x0);
    auto dy = std::make_shared<op::v1::Subtract>(y_pad, y0);
    auto omdx = std::make_shared<op::v1::Subtract>(ctx.c1, dx);
    auto omdy = std::make_shared<op::v1::Subtract>(ctx.c1, dy);

    std::vector<std::shared_ptr<Node>> weights{std::make_shared<op::v1::Multiply>(omdx, omdy),
                                               std::make_shared<op::v1::Multiply>(dx, omdy),
                                               std::make_shared<op::v1::Multiply>(omdx, dy),
                                               std::make_shared<op::v1::Multiply>(dx, dy)};

    if (is_reflection) {
        x0 = reflect_index(ctx, x0, w_in_f, attrs.align_corners);
        x1 = reflect_index(ctx, x1, w_in_f, attrs.align_corners);
        y0 = reflect_index(ctx, y0, h_in_f, attrs.align_corners);
        y1 = reflect_index(ctx, y1, h_in_f, attrs.align_corners);
    }

    std::vector<std::shared_ptr<Node>> xs{x0, x1, x0, x1};
    std::vector<std::shared_ptr<Node>> ys{y0, y0, y1, y1};

    std::shared_ptr<Node> res;
    for (int i = 0; i < 4; ++i) {
        auto xi = xs[i], yi = ys[i];
        if (is_zeros || is_border) {
            clamp_indices_inplace(ctx, xi, yi, w_in_f, h_in_f);
        }
        auto v = gather_hw_nhwc(ctx, xi, yi);

        if (ctx.element_type != ctx.calc_type) {
            v = std::make_shared<op::v0::Convert>(v, ctx.calc_type);
        }

        if (is_zeros) {
            auto ok = inside_mask_indexed(ctx, xs[i], ys[i], w_in_f, h_in_f);
            auto mask = std::make_shared<op::v0::Convert>(ok, ctx.calc_type);
            v = std::make_shared<op::v1::Multiply>(
                v,
                std::make_shared<op::v0::Unsqueeze>(mask, ctx.i32_neg1));
        }

        auto wexp = std::make_shared<op::v0::Unsqueeze>(weights[i], ctx.i32_neg1);
        auto tap = std::make_shared<op::v1::Multiply>(v, wexp);
        res = res ? std::shared_ptr<Node>(std::make_shared<op::v1::Add>(res, tap)) : tap;
    }
    if (ctx.element_type != ctx.calc_type) {
        res = std::make_shared<op::v0::Convert>(res, ctx.element_type);
    }
    return res;
}

// BICUBIC
std::shared_ptr<Node> build_bicubic_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_border = attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER;
    const bool is_zeros = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);

    std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
    if (is_reflection && attrs.align_corners) {
        x_pad = reflect_coord(ctx, ctx.x, w_in_f, true);
        y_pad = reflect_coord(ctx, ctx.y, h_in_f, true);
    }

    auto x0 = std::make_shared<op::v0::Floor>(x_pad);
    auto y0 = std::make_shared<op::v0::Floor>(y_pad);
    auto dx_raw = std::make_shared<op::v1::Subtract>(x_pad, x0);
    auto dy_raw = std::make_shared<op::v1::Subtract>(y_pad, y0);

    std::shared_ptr<Node> dx = dx_raw;
    std::shared_ptr<Node> dy = dy_raw;
    if (!is_zeros) {
        auto eps = op::v0::Constant::create(ctx.calc_type, {}, {1e-7F});
        auto one_m_eps = std::make_shared<op::v1::Subtract>(ctx.c1, eps);
        dx = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(dx_raw, ctx.c0), one_m_eps);
        dy = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(dy_raw, ctx.c0), one_m_eps);
    }

    auto w_is_one = std::make_shared<op::v1::Equal>(w_in_f, ctx.c1);
    auto h_is_one = std::make_shared<op::v1::Equal>(h_in_f, ctx.c1);
    dx = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, dx);
    dy = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, dy);

    // cubic weights (a = -0.75)
    auto a = op::v0::Constant::create(ctx.calc_type, {}, {-0.75F});
    auto c3 = op::v0::Constant::create(ctx.calc_type, {}, {3.0F});
    auto c4 = op::v0::Constant::create(ctx.calc_type, {}, {4.0F});
    auto c5 = op::v0::Constant::create(ctx.calc_type, {}, {5.0F});
    auto c8 = op::v0::Constant::create(ctx.calc_type, {}, {8.0F});

    auto cubic_weight = [&](const std::shared_ptr<Node>& t) -> std::shared_ptr<Node> {
        auto t0 = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(t, ctx.c0), ctx.c2);
        auto t2 = std::make_shared<op::v1::Multiply>(t0, t0);
        auto t3 = std::make_shared<op::v1::Multiply>(t2, t0);
        auto ap2 = std::make_shared<op::v1::Add>(a, ctx.c2);
        auto ap3 = std::make_shared<op::v1::Add>(a, c3);

        auto f01_in = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, t0), ap3);
        auto f01 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(f01_in, t2), ctx.c1);

        auto five_a = std::make_shared<op::v1::Multiply>(c5, a);
        auto eight_a = std::make_shared<op::v1::Multiply>(c8, a);
        auto four_a = std::make_shared<op::v1::Multiply>(c4, a);
        auto f12_in1 = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(a, t0), five_a);
        auto f12_in2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(f12_in1, t0), eight_a);
        auto f12 = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(f12_in2, t0), four_a);

        auto cond1 = std::make_shared<op::v1::LessEqual>(t0, ctx.c1);
        auto cond2 = std::make_shared<op::v1::Less>(t0, ctx.c2);
        return std::make_shared<op::v1::Select>(cond1, f01, std::make_shared<op::v1::Select>(cond2, f12, ctx.c0));
    };

    auto t_x0 = std::make_shared<op::v1::Add>(ctx.c1, dx);
    auto t_x1 = dx;
    auto t_x2 = std::make_shared<op::v1::Subtract>(ctx.c1, dx);
    auto t_x3 = std::make_shared<op::v1::Subtract>(ctx.c2, dx);

    auto w_x0 = cubic_weight(t_x0);
    auto w_x1 = cubic_weight(t_x1);
    auto w_x2 = cubic_weight(t_x2);
    auto w_x3 = cubic_weight(t_x3);

    w_x0 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x0);
    w_x1 = std::make_shared<op::v1::Select>(w_is_one, ctx.c1, w_x1);
    w_x2 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x2);
    w_x3 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x3);

    auto t_y0 = std::make_shared<op::v1::Add>(ctx.c1, dy);
    auto t_y1 = dy;
    auto t_y2 = std::make_shared<op::v1::Subtract>(ctx.c1, dy);
    auto t_y3 = std::make_shared<op::v1::Subtract>(ctx.c2, dy);

    auto w_y0 = cubic_weight(t_y0);
    auto w_y1 = cubic_weight(t_y1);
    auto w_y2 = cubic_weight(t_y2);
    auto w_y3 = cubic_weight(t_y3);

    w_y0 = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, w_y0);
    w_y1 = std::make_shared<op::v1::Select>(h_is_one, ctx.c1, w_y1);
    w_y2 = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, w_y2);
    w_y3 = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, w_y3);

    std::vector<std::shared_ptr<Node>> x_idx{
        std::make_shared<op::v1::Subtract>(x0, ctx.c1),
        x0,
        std::make_shared<op::v1::Add>(x0, ctx.c1),
        std::make_shared<op::v1::Add>(x0, ctx.c2),
    };
    std::vector<std::shared_ptr<Node>> y_idx{
        std::make_shared<op::v1::Subtract>(y0, ctx.c1),
        y0,
        std::make_shared<op::v1::Add>(y0, ctx.c1),
        std::make_shared<op::v1::Add>(y0, ctx.c2),
    };
    std::vector<std::shared_ptr<Node>> wx{w_x0, w_x1, w_x2, w_x3};
    std::vector<std::shared_ptr<Node>> wy{w_y0, w_y1, w_y2, w_y3};

    // validity masks for ZEROS (before clamping)
    std::vector<std::shared_ptr<Node>> x_ok(4), y_ok(4);
    if (is_zeros) {
        for (int i = 0; i < 4; ++i) {
            x_ok[i] = std::make_shared<op::v1::LogicalAnd>(std::make_shared<op::v1::GreaterEqual>(x_idx[i], ctx.c0),
                                                           std::make_shared<op::v1::Less>(x_idx[i], w_in_f));
            y_ok[i] = std::make_shared<op::v1::LogicalAnd>(std::make_shared<op::v1::GreaterEqual>(y_idx[i], ctx.c0),
                                                           std::make_shared<op::v1::Less>(y_idx[i], h_in_f));
        }
    }

    if (is_reflection) {
        for (int i = 0; i < 4; ++i) {
            x_idx[i] = reflect_index(ctx, x_idx[i], w_in_f, attrs.align_corners);
            y_idx[i] = reflect_index(ctx, y_idx[i], h_in_f, attrs.align_corners);
        }
    } else {
        auto w_is_one_chk = std::make_shared<op::v1::Equal>(w_in_f, ctx.c1);
        auto h_is_one_chk = std::make_shared<op::v1::Equal>(h_in_f, ctx.c1);
        for (int i = 0; i < 4; ++i) {
            x_idx[i] = std::make_shared<op::v1::Select>(w_is_one_chk, ctx.c0, x_idx[i]);
            y_idx[i] = std::make_shared<op::v1::Select>(h_is_one_chk, ctx.c0, y_idx[i]);
        }
    }

    std::shared_ptr<Node> sum;
    for (int iy = 0; iy < 4; ++iy) {
        std::shared_ptr<Node> row;
        for (int ix = 0; ix < 4; ++ix) {
            auto xi = x_idx[ix];
            auto yi = y_idx[iy];

            if (is_zeros || is_border) {
                clamp_indices_inplace(ctx, xi, yi, w_in_f, h_in_f);
            }

            auto val = gather_hw_nhwc(ctx, xi, yi);
            if (ctx.element_type != ctx.calc_type) {
                val = std::make_shared<op::v0::Convert>(val, ctx.calc_type);
            }

            auto wxy = std::make_shared<op::v1::Multiply>(wx[ix], wy[iy]);
            if (is_zeros) {
                auto ok = std::make_shared<op::v1::LogicalAnd>(x_ok[ix], y_ok[iy]);
                auto mask = std::make_shared<op::v0::Convert>(ok, ctx.calc_type);
                wxy = std::make_shared<op::v1::Multiply>(wxy, mask);
            }

            auto wexp = std::make_shared<op::v0::Unsqueeze>(wxy, ctx.i32_neg1);
            auto tap = std::make_shared<op::v1::Multiply>(val, wexp);

            row = row ? std::shared_ptr<Node>(std::make_shared<op::v1::Add>(row, tap)) : tap;
        }
        sum = sum ? std::shared_ptr<Node>(std::make_shared<op::v1::Add>(sum, row)) : row;
    }

    if (ctx.element_type != ctx.calc_type) {
        sum = std::make_shared<op::v0::Convert>(sum, ctx.element_type);
    }
    return sum;
}

// ---- ctx builders ----

// Helper to initialize common constants that are shared between static and dynamic paths
void init_common_constants(Ctx& ctx) {
    // Float constants
    ctx.c0 = op::v0::Constant::create(ctx.calc_type, {}, {0.0F});
    ctx.c1 = op::v0::Constant::create(ctx.calc_type, {}, {1.0F});
    ctx.c2 = op::v0::Constant::create(ctx.calc_type, {}, {2.0F});
    ctx.c0_5 = op::v0::Constant::create(ctx.calc_type, {}, {0.5F});
    
    // Integer constants
    ctx.i32_0 = op::v0::Constant::create(element::i32, {}, {0});
    ctx.i32_1 = op::v0::Constant::create(element::i32, {}, {1});
    ctx.i32_2 = op::v0::Constant::create(element::i32, {}, {2});
    ctx.i32_3 = op::v0::Constant::create(element::i32, {}, {3});
    ctx.i32_neg1 = op::v0::Constant::create(element::i32, {}, {-1});
    
    // Common axes (just references to integer constants)
    ctx.axis_0 = ctx.i32_0;
    ctx.axis_3 = ctx.i32_3;
    
    // NHWC transpose pattern
    ctx.to_nhwc_perm = op::v0::Constant::create(element::i32, {4}, {0, 2, 3, 1});
}

// Dynamic: extract all size scalars via ShapeOf/Gather
bool build_ctx_dynamic(const std::shared_ptr<ov::op::v9::GridSample>& gs, Ctx& ctx) {
    const auto& data = gs->input_value(0);
    const auto& grid = gs->input_value(1);

    if (data.get_partial_shape().rank().get_length() != 4 || grid.get_partial_shape().rank().get_length() != 4) {
        return false;
    }

    ctx.element_type = data.get_element_type();
    ctx.calc_type = element::f32;

    // Initialize all common constants
    init_common_constants(ctx);

    auto dshape = std::make_shared<op::v3::ShapeOf>(data, element::i32);
    ctx.n_dim = std::make_shared<op::v8::Gather>(dshape, ctx.i32_0, ctx.axis_0);
    ctx.c_dim = std::make_shared<op::v8::Gather>(dshape, ctx.i32_1, ctx.axis_0);
    ctx.h_in = std::make_shared<op::v8::Gather>(dshape, ctx.i32_2, ctx.axis_0);
    ctx.w_in = std::make_shared<op::v8::Gather>(dshape, ctx.i32_3, ctx.axis_0);

    auto gshape = std::make_shared<op::v3::ShapeOf>(grid, element::i32);
    ctx.h_out = std::make_shared<op::v8::Gather>(gshape, ctx.i32_1, ctx.axis_0);
    ctx.w_out = std::make_shared<op::v8::Gather>(gshape, ctx.i32_2, ctx.axis_0);

    // split grid channels (x,y)
    auto grid_conv = grid.get_element_type() == ctx.calc_type
                         ? grid
                         : std::shared_ptr<Node>(std::make_shared<op::v0::Convert>(grid, ctx.calc_type));
    ctx.x_norm = std::make_shared<op::v8::Gather>(grid_conv, ctx.i32_0, ctx.axis_3);
    ctx.y_norm = std::make_shared<op::v8::Gather>(grid_conv, ctx.i32_1, ctx.axis_3);

    // NCHW -> NHWC
    ctx.data_nhwc = std::make_shared<op::v1::Transpose>(data, ctx.to_nhwc_perm);
    return true;
}

// Static: sizes are direct constants, no ShapeOf/Gather on shapes
bool build_ctx_static(const std::shared_ptr<ov::op::v9::GridSample>& gs, Ctx& ctx) {
    const auto& data_iv = gs->input_value(0);
    const auto& grid_iv = gs->input_value(1);
    const auto& dps = data_iv.get_partial_shape();
    const auto& gps = grid_iv.get_partial_shape();
    if (!dps.is_static() || !gps.is_static()) {
        return false;
    }

    auto data_shape = dps.to_shape();  // N, C, H_in, W_in
    auto grid_shape = gps.to_shape();  // N, H_out, W_out, 2
    if (data_shape.size() != 4 || grid_shape.size() != 4) {
        return false;
    }

    ctx.element_type = data_iv.get_element_type();
    ctx.calc_type = element::f32;

    // Initialize all common constants
    init_common_constants(ctx);

    // sizes as i32 constants
    auto cN = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(data_shape[0])});
    auto cC = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(data_shape[1])});
    auto cHi = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(data_shape[2])});
    auto cWi = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(data_shape[3])});
    auto cHo = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(grid_shape[1])});
    auto cWo = op::v0::Constant::create(element::i32, {}, {static_cast<int32_t>(grid_shape[2])});

    ctx.n_dim = cN;
    ctx.c_dim = cC;
    ctx.h_in = cHi;
    ctx.w_in = cWi;
    ctx.h_out = cHo;
    ctx.w_out = cWo;

    // grid channels split (keep as graph ops; shapes are const so it folds nicely)
    auto grid = grid_iv.get_node_shared_ptr();
    auto grid_conv = grid_iv.get_element_type() == ctx.calc_type
                         ? grid_iv.get_node_shared_ptr()
                         : std::shared_ptr<Node>(std::make_shared<op::v0::Convert>(grid_iv, ctx.calc_type));
    ctx.x_norm = std::make_shared<op::v8::Gather>(grid_conv, ctx.i32_0, ctx.axis_3);
    ctx.y_norm = std::make_shared<op::v8::Gather>(grid_conv, ctx.i32_1, ctx.axis_3);

    // NCHW -> NHWC
    ctx.data_nhwc = std::make_shared<op::v1::Transpose>(data_iv, ctx.to_nhwc_perm);
    return true;
}

// Common glue: build ctx, normalize, build mode, NHWC->NCHW, replace
using BuildModeFn = std::function<std::shared_ptr<Node>(const Ctx&, const ov::op::v9::GridSample::Attributes&)>;
bool decompose_impl(const std::shared_ptr<ov::op::v9::GridSample>& gs,
                    const BuildModeFn& build_mode_result_nhwc,
                    bool use_static_ctx) {
    Ctx ctx{};
    if (!(use_static_ctx ? build_ctx_static(gs, ctx) : build_ctx_dynamic(gs, ctx))) {
        return false;
    }

    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);

    normalize_grid_to_pixels(ctx,
                             ctx.x_norm,
                             ctx.y_norm,
                             w_in_f,
                             h_in_f,
                             gs->get_attributes().align_corners,
                             ctx.x,
                             ctx.y);

    auto result_nhwc = build_mode_result_nhwc(ctx, gs->get_attributes());
    if (!result_nhwc) {
        return false;
    }

    auto to_nchw = op::v0::Constant::create(element::i32, {4}, {0, 3, 1, 2});
    auto result = std::make_shared<op::v1::Transpose>(result_nhwc, to_nchw);

    result->set_friendly_name(gs->get_friendly_name());
    ov::copy_runtime_info(gs, result);
    ov::replace_node_update_name(gs, result);
    return true;
}

}  // unnamed namespace

// ============================================================================
// PASSES: split into STATIC and DYNAMIC paths (no reference path).
// ============================================================================

// If your header previously declared only GridSampleDecomposition + 3 matchers,
// you can either:
//  - update the header to declare the 6 new matchers below, or
//  - replace your old 3 matchers with the new behavior by keeping the same names.
// Here we expose 6 fine-grained matchers and one composite GraphRewrite.

class GridSampleDecompositionNearestStatic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionNearestStatic() {
        // Match GridSample with 4D inputs and static shapes
        auto data_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto grid_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::NEAREST;
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_nearest_nhwc(ctx, attrs);
                },
                true);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionNearestStatic"), cb);
    }
};

class GridSampleDecompositionBilinearStatic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBilinearStatic() {
        // Match GridSample with 4D inputs and static shapes
        auto data_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto grid_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::BILINEAR;
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_bilinear_nhwc(ctx, attrs);
                },
                true);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBilinearStatic"),
                         cb);
    }
};

class GridSampleDecompositionBicubicStatic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBicubicStatic() {
        // Match GridSample with 4D inputs and static shapes
        auto data_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto grid_pattern = ov::pass::pattern::any_input(
            [](const Output<Node>& output) {
                return output.get_partial_shape().rank().is_static() &&
                       output.get_partial_shape().rank().get_length() == 4 &&
                       output.get_partial_shape().is_static();
            });
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::BICUBIC;
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_bicubic_nhwc(ctx, attrs);
                },
                true);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBicubicStatic"), cb);
    }
};

class GridSampleDecompositionNearestDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionNearestDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::NEAREST) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || 
                       !gs->get_input_partial_shape(1).is_static();
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_nearest_nhwc(ctx, attrs);
                },
                false);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionNearestDynamic"),
                         cb);
    }
};

class GridSampleDecompositionBilinearDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBilinearDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BILINEAR) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || 
                       !gs->get_input_partial_shape(1).is_static();
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_bilinear_nhwc(ctx, attrs);
                },
                false);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBilinearDynamic"),
                         cb);
    }
};

class GridSampleDecompositionBicubicDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBicubicDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>({data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BICUBIC) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || 
                       !gs->get_input_partial_shape(1).is_static();
            });
        
        matcher_pass_callback cb = [this](ov::pass::pattern::Matcher& m) {
            auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
            if (!gs || transformation_callback(gs)) {
                return false;
            }
            return decompose_impl(
                gs,
                [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) {
                    return build_bicubic_nhwc(ctx, attrs);
                },
                false);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBicubicDynamic"),
                         cb);
    }
};

// Composite GraphRewrite that installs the 6 matchers
GridSampleDecomposition::GridSampleDecomposition() {
    // Static first (cheaper graphs), then dynamic fallback
    add_matcher<GridSampleDecompositionNearestStatic>();
    add_matcher<GridSampleDecompositionBilinearStatic>();
    add_matcher<GridSampleDecompositionBicubicStatic>();

    add_matcher<GridSampleDecompositionNearestDynamic>();
    add_matcher<GridSampleDecompositionBilinearDynamic>();
    add_matcher<GridSampleDecompositionBicubicDynamic>();
}

}  // namespace ov::intel_cpu
