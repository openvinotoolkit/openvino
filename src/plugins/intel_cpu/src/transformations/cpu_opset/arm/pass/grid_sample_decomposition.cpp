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

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
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
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

// ============================================================================
// INTERNAL SHARED: graph-level builders (reused by both static & dynamic paths)
// ============================================================================

namespace {

struct Ctx {
    // Scalars with sizes (might be Constants in static path or ShapeOf-slices in dynamic)
    std::shared_ptr<Node> n_dim = nullptr;
    std::shared_ptr<Node> c_dim = nullptr;
    std::shared_ptr<Node> h_in = nullptr;
    std::shared_ptr<Node> w_in = nullptr;
    std::shared_ptr<Node> h_out = nullptr;
    std::shared_ptr<Node> w_out = nullptr;
    // Types
    element::Type element_type = element::dynamic;
    element::Type calc_type = element::dynamic;  // set to f32 for math
    // Float constants
    std::shared_ptr<Node> c0 = nullptr;
    std::shared_ptr<Node> c1 = nullptr;
    std::shared_ptr<Node> c2 = nullptr;
    std::shared_ptr<Node> c0_5 = nullptr;
    // Integer constants
    std::shared_ptr<Node> i32_0 = nullptr;
    std::shared_ptr<Node> i32_1 = nullptr;
    std::shared_ptr<Node> i32_2 = nullptr;
    std::shared_ptr<Node> i32_3 = nullptr;
    std::shared_ptr<Node> i32_neg1 = nullptr;
    // Common axes
    std::shared_ptr<Node> axis_0 = nullptr;
    std::shared_ptr<Node> axis_3 = nullptr;
    // NHWC transpose pattern
    std::shared_ptr<Node> to_nhwc_perm = nullptr;
    // Pixels coords after normalization
    std::shared_ptr<Node> x = nullptr;
    std::shared_ptr<Node> y = nullptr;
    // Normalized coords (from grid, in [-1, 1])
    std::shared_ptr<Node> x_norm = nullptr;
    std::shared_ptr<Node> y_norm = nullptr;
    // Data as NHWC for GatherND
    std::shared_ptr<Node> data_nhwc = nullptr;
    // Original data (NCHW)
    std::shared_ptr<Node> data_nchw = nullptr;
};

// ---- small helpers ----
std::shared_ptr<Node> to_f32(const std::shared_ptr<Node>& input_node) {
    return std::make_shared<op::v0::Convert>(input_node, element::f32);
}

// Copy runtime info from the original node to entire subgraph feeding `output`.
void copy_rt_to_subgraph(const std::shared_ptr<Node>& from, const std::shared_ptr<Node>& output) {
    std::vector<std::shared_ptr<Node>> stack;
    stack.push_back(output);
    std::unordered_set<const Node*> visited;
    std::vector<std::shared_ptr<Node>> subgraph_nodes;
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node || visited.count(node.get()) > 0) {
            continue;
        }
        visited.insert(node.get());
        subgraph_nodes.push_back(node);
        for (const auto& input : node->inputs()) {
            auto src = input.get_source_output().get_node_shared_ptr();
            if (src) {
                stack.push_back(src);
            }
        }
    }
    ov::copy_runtime_info(NodeVector{from}, subgraph_nodes);
}

// Normalize grid [-1, 1] to pixel coordinates (no padding/reflection here)
void normalize_grid_to_pixels(const Ctx& ctx,
                              const std::shared_ptr<Node>& x_grid_coord,
                              const std::shared_ptr<Node>& y_grid_coord,
                              const std::shared_ptr<Node>& width_in_float,
                              const std::shared_ptr<Node>& height_in_float,
                              bool align_corners,
                              std::shared_ptr<Node>& x_pixel_out,
                              std::shared_ptr<Node>& y_pixel_out) {
    if (align_corners) {
        auto w_m1 = std::make_shared<op::v1::Subtract>(width_in_float, ctx.c1);
        auto h_m1 = std::make_shared<op::v1::Subtract>(height_in_float, ctx.c1);
        x_pixel_out = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(x_grid_coord, ctx.c1), ctx.c2),
            w_m1);
        y_pixel_out = std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(y_grid_coord, ctx.c1), ctx.c2),
            h_m1);
    } else {
        // ((t + 1) * size - 1) / 2
        auto x_p1_w =
            std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(x_grid_coord, ctx.c1), width_in_float);
        auto y_p1_h =
            std::make_shared<op::v1::Multiply>(std::make_shared<op::v1::Add>(y_grid_coord, ctx.c1), height_in_float);
        x_pixel_out = std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Subtract>(x_p1_w, ctx.c1), ctx.c2);
        y_pixel_out = std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Subtract>(y_p1_h, ctx.c1), ctx.c2);
    }
}

// Create [N, H_out, W_out] batch indices tensor for GatherND (broadcasted)
std::shared_ptr<Node> create_batch_indices(const Ctx& ctx) {
    auto range = std::make_shared<op::v4::Range>(ctx.i32_0, ctx.n_dim, ctx.i32_1, element::i32);

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

    // For REFLECTION, match reference semantics:
    // - when align_corners == true: reflect continuous coordinates prior to flooring
    // - when align_corners == false: do NOT reflect continuous coords here; indices are
    //   reflected later via reflect_index (post-floor) like in the reference path
    std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
    if (is_reflection && attrs.align_corners) {
        x_pad = reflect_coord(ctx, ctx.x, w_in_f, true);
        y_pad = reflect_coord(ctx, ctx.y, h_in_f, true);
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
            v = std::make_shared<op::v1::Multiply>(v, std::make_shared<op::v0::Unsqueeze>(mask, ctx.i32_neg1));
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

    // cubic weights (a = -0.75), match OV reference cubic_coeffs formulation exactly
    auto a = op::v0::Constant::create(ctx.calc_type, {}, {-0.75F});
    auto c3 = op::v0::Constant::create(ctx.calc_type, {}, {3.0F});
    auto c4 = op::v0::Constant::create(ctx.calc_type, {}, {4.0F});
    auto c5 = op::v0::Constant::create(ctx.calc_type, {}, {5.0F});
    auto c8 = op::v0::Constant::create(ctx.calc_type, {}, {8.0F});

    auto dx_p1 = std::make_shared<op::v1::Add>(dx, ctx.c1);          // dx + 1
    auto one_m_dx = std::make_shared<op::v1::Subtract>(ctx.c1, dx);  // 1 - dx
    auto two_m_dx = std::make_shared<op::v1::Subtract>(ctx.c2, dx);  // 2 - dx

    auto dx2 = std::make_shared<op::v1::Multiply>(dx, dx);
    auto dx_p1_2 = std::make_shared<op::v1::Multiply>(dx_p1, dx_p1);
    auto one_m_dx_2 = std::make_shared<op::v1::Multiply>(one_m_dx, one_m_dx);
    auto two_m_dx_2 = std::make_shared<op::v1::Multiply>(two_m_dx, two_m_dx);

    // v0 = ((A*(dx+1) - 5A) * (dx+1) + 8A) * (dx+1) - 4A
    auto a_dx_p1 = std::make_shared<op::v1::Multiply>(a, dx_p1);
    auto a5 = std::make_shared<op::v1::Multiply>(c5, a);
    auto term_v0_1 = std::make_shared<op::v1::Subtract>(a_dx_p1, a5);
    auto term_v0_2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(term_v0_1, dx_p1),
                                                   std::make_shared<op::v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_x0 =
        std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(term_v0_2, dx_p1),
                                           std::make_shared<op::v1::Multiply>(c4, a));

    // v1 = ((A + 2) * dx - (A + 3)) * dx * dx + 1
    auto ap2 = std::make_shared<op::v1::Add>(a, ctx.c2);
    auto ap3 = std::make_shared<op::v1::Add>(a, c3);
    std::shared_ptr<Node> w_x1 = std::make_shared<op::v1::Add>(
        std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, dx), ap3),
            dx2),
        ctx.c1);

    // v2 = ((A + 2) * (1 - dx) - (A + 3)) * (1 - dx)^2 + 1
    std::shared_ptr<Node> w_x2 = std::make_shared<op::v1::Add>(
        std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, one_m_dx), ap3),
            one_m_dx_2),
        ctx.c1);

    // v3 = ((A * (2 - dx) - 5A) * (2 - dx) + 8A) * (2 - dx) - 4A
    auto a_two_m_dx = std::make_shared<op::v1::Multiply>(a, two_m_dx);
    auto term_v3_1 = std::make_shared<op::v1::Subtract>(a_two_m_dx, a5);
    auto term_v3_2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(term_v3_1, two_m_dx),
                                                   std::make_shared<op::v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_x3 =
        std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(term_v3_2, two_m_dx),
                                           std::make_shared<op::v1::Multiply>(c4, a));

    w_x0 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x0);
    w_x1 = std::make_shared<op::v1::Select>(w_is_one, ctx.c1, w_x1);
    w_x2 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x2);
    w_x3 = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, w_x3);

    auto dy_p1 = std::make_shared<op::v1::Add>(dy, ctx.c1);          // dy + 1
    auto one_m_dy = std::make_shared<op::v1::Subtract>(ctx.c1, dy);  // 1 - dy
    auto two_m_dy = std::make_shared<op::v1::Subtract>(ctx.c2, dy);  // 2 - dy

    auto dy2 = std::make_shared<op::v1::Multiply>(dy, dy);
    auto dy_p1_2 = std::make_shared<op::v1::Multiply>(dy_p1, dy_p1);
    auto one_m_dy_2 = std::make_shared<op::v1::Multiply>(one_m_dy, one_m_dy);
    auto two_m_dy_2 = std::make_shared<op::v1::Multiply>(two_m_dy, two_m_dy);

    // y weights per OV reference cubic_coeffs(dy)
    auto a_dy_p1 = std::make_shared<op::v1::Multiply>(a, dy_p1);
    auto term_wy0_1 = std::make_shared<op::v1::Subtract>(a_dy_p1, a5);
    auto term_wy0_2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(term_wy0_1, dy_p1),
                                                    std::make_shared<op::v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_y0 =
        std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(term_wy0_2, dy_p1),
                                           std::make_shared<op::v1::Multiply>(c4, a));

    std::shared_ptr<Node> w_y1 = std::make_shared<op::v1::Add>(
        std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, dy), ap3),
            dy2),
        ctx.c1);

    std::shared_ptr<Node> w_y2 = std::make_shared<op::v1::Add>(
        std::make_shared<op::v1::Multiply>(
            std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, one_m_dy), ap3),
            one_m_dy_2),
        ctx.c1);

    auto a_two_m_dy = std::make_shared<op::v1::Multiply>(a, two_m_dy);
    auto term_wy3_1 = std::make_shared<op::v1::Subtract>(a_two_m_dy, a5);
    auto term_wy3_2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(term_wy3_1, two_m_dy),
                                                    std::make_shared<op::v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_y3 =
        std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(term_wy3_2, two_m_dy),
                                           std::make_shared<op::v1::Multiply>(c4, a));

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
                val = std::make_shared<op::v1::Multiply>(val, std::make_shared<op::v0::Unsqueeze>(mask, ctx.i32_neg1));
            }

            // Expand weight on last axis (NHWC)
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
void init_common_constants(Ctx& context) {
    // Float constants
    context.c0 = op::v0::Constant::create(context.calc_type, {}, {0.0F});
    context.c1 = op::v0::Constant::create(context.calc_type, {}, {1.0F});
    context.c2 = op::v0::Constant::create(context.calc_type, {}, {2.0F});
    context.c0_5 = op::v0::Constant::create(context.calc_type, {}, {0.5F});

    // Integer constants
    context.i32_0 = op::v0::Constant::create(element::i32, {}, {0});
    context.i32_1 = op::v0::Constant::create(element::i32, {}, {1});
    context.i32_2 = op::v0::Constant::create(element::i32, {}, {2});
    context.i32_3 = op::v0::Constant::create(element::i32, {}, {3});
    context.i32_neg1 = op::v0::Constant::create(element::i32, {}, {-1});

    // Common axes (just references to integer constants)
    context.axis_0 = context.i32_0;
    context.axis_3 = context.i32_3;

    // NHWC transpose pattern
    context.to_nhwc_perm = op::v0::Constant::create(element::i32, {4}, {0, 2, 3, 1});
}

// Unified context building: uses ShapeOf/Gather which can be folded to constants for static shapes
bool build_ctx(const std::shared_ptr<ov::op::v9::GridSample>& grid_sample_op, Ctx& context) {
    const auto& data = grid_sample_op->input_value(0);
    const auto& grid = grid_sample_op->input_value(1);

    if (data.get_partial_shape().rank().get_length() != 4 || grid.get_partial_shape().rank().get_length() != 4) {
        return false;
    }

    context.element_type = data.get_element_type();
    context.calc_type = element::f32;

    // Initialize all common constants
    init_common_constants(context);

    // Build shape extraction subgraphs using make_try_fold (automatically folds to constants if shapes are static)
    auto dshape = ov::op::util::make_try_fold<op::v3::ShapeOf>(data, element::i32);
    context.n_dim = ov::op::util::make_try_fold<op::v8::Gather>(dshape, context.i32_0, context.axis_0);
    context.c_dim = ov::op::util::make_try_fold<op::v8::Gather>(dshape, context.i32_1, context.axis_0);
    context.h_in = ov::op::util::make_try_fold<op::v8::Gather>(dshape, context.i32_2, context.axis_0);
    context.w_in = ov::op::util::make_try_fold<op::v8::Gather>(dshape, context.i32_3, context.axis_0);

    auto gshape = ov::op::util::make_try_fold<op::v3::ShapeOf>(grid, element::i32);
    context.h_out = ov::op::util::make_try_fold<op::v8::Gather>(gshape, context.i32_1, context.axis_0);
    context.w_out = ov::op::util::make_try_fold<op::v8::Gather>(gshape, context.i32_2, context.axis_0);

    // split grid channels (x,y)
    auto grid_conv = grid.get_element_type() == context.calc_type
                         ? grid
                         : std::shared_ptr<Node>(std::make_shared<op::v0::Convert>(grid, context.calc_type));
    context.x_norm = std::make_shared<op::v8::Gather>(grid_conv, context.i32_0, context.axis_3);
    context.y_norm = std::make_shared<op::v8::Gather>(grid_conv, context.i32_1, context.axis_3);

    // Keep references to data in both layouts for different gather strategies
    // NCHW original
    context.data_nchw = data.get_node_shared_ptr();
    // NHWC via transpose
    context.data_nhwc = std::make_shared<op::v1::Transpose>(data, context.to_nhwc_perm);
    return true;
}

// Common glue: build ctx, normalize, build mode, NHWC->NCHW, replace
using BuildModeFn = std::function<std::shared_ptr<Node>(const Ctx&, const ov::op::v9::GridSample::Attributes&)>;
bool decompose_impl(const std::shared_ptr<ov::op::v9::GridSample>& grid_sample_op,
                    const BuildModeFn& build_mode_result_nhwc) {
    Ctx context{};
    if (!build_ctx(grid_sample_op, context)) {
        return false;
    }

    auto w_in_f = to_f32(context.w_in);
    auto h_in_f = to_f32(context.h_in);

    normalize_grid_to_pixels(context,
                             context.x_norm,
                             context.y_norm,
                             w_in_f,
                             h_in_f,
                             grid_sample_op->get_attributes().align_corners,
                             context.x,
                             context.y);

    auto result_nhwc = build_mode_result_nhwc(context, grid_sample_op->get_attributes());
    if (!result_nhwc) {
        return false;
    }

    auto to_nchw = op::v0::Constant::create(element::i32, {4}, {0, 3, 1, 2});
    auto result = std::make_shared<op::v1::Transpose>(result_nhwc, to_nchw);

    result->set_friendly_name(grid_sample_op->get_friendly_name());
    copy_rt_to_subgraph(grid_sample_op, result);
    ov::replace_node_update_name(grid_sample_op, result);
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
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::NEAREST;
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            // Use reference path for known-problematic dynamic combos to avoid accuracy issues
            const auto& attrs = grid_sample->get_attributes();
            const bool is_f32_data = grid_sample->get_input_element_type(0) == element::f32;
            const bool is_f32_grid = grid_sample->get_input_element_type(1) == element::f32;
            // NEAREST + REFLECTION + align_corners=false (restrict to f32/f32 to not break f16 nightly)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION && !attrs.align_corners) {
                return false;  // keep original GridSample -> plugin will fallback to Reference
            }
            // NEAREST + BORDER + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER && !attrs.align_corners) {
                return false;
            }
            // NEAREST + ZEROS + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS && !attrs.align_corners) {
                return false;
            }
            // For non-f32 cases in these problematic modes, explicitly convert to f32 -> GridSample -> convert back.
            if ((!is_f32_data || !is_f32_grid) && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                (attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION ||
                 attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER ||
                 attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS) &&
                !attrs.align_corners) {
                auto data_f32 = std::make_shared<op::v0::Convert>(grid_sample->input_value(0), element::f32);
                auto grid_f32 = std::make_shared<op::v0::Convert>(grid_sample->input_value(1), element::f32);
                auto gs_f32 = std::make_shared<op::v9::GridSample>(data_f32, grid_f32, attrs);
                auto out = std::make_shared<op::v0::Convert>(gs_f32, grid_sample->get_output_element_type(0));
                out->set_friendly_name(grid_sample->get_friendly_name());
                copy_rt_to_subgraph(grid_sample, out);
                ov::replace_node_update_name(grid_sample, out);
                return true;
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_nearest_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionNearestStatic"),
                         callback);
    }
};

class GridSampleDecompositionBilinearStatic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBilinearStatic() {
        // Match GridSample with 4D inputs and static shapes
        auto data_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::BILINEAR;
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_bilinear_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBilinearStatic"),
                         callback);
    }
};

class GridSampleDecompositionBicubicStatic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBicubicStatic() {
        // Match GridSample with 4D inputs and static shapes
        auto data_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto grid_pattern = ov::pass::pattern::any_input(
            ov::pass::pattern::all_of({ov::pass::pattern::rank_equals(4), ov::pass::pattern::has_static_shape()}));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                return gs && gs->get_attributes().mode == op::v9::GridSample::InterpolationMode::BICUBIC;
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            // Use reference path for known-problematic dynamic combos to avoid accuracy issues
            const auto& attrs = grid_sample->get_attributes();
            const bool is_f32_data = grid_sample->get_input_element_type(0) == element::f32;
            const bool is_f32_grid = grid_sample->get_input_element_type(1) == element::f32;
            // BICUBIC + ZEROS + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::BICUBIC &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS && !attrs.align_corners) {
                return false;  // keep original GridSample -> plugin will fallback to Reference
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_bicubic_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBicubicStatic"),
                         callback);
    }
};

class GridSampleDecompositionNearestDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionNearestDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::NEAREST) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || !gs->get_input_partial_shape(1).is_static();
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            // Use reference path for known-problematic combos to avoid accuracy issues
            const auto& attrs = grid_sample->get_attributes();
            const bool is_f32_data = grid_sample->get_input_element_type(0) == element::f32;
            const bool is_f32_grid = grid_sample->get_input_element_type(1) == element::f32;
            // NEAREST + REFLECTION + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION && !attrs.align_corners) {
                return false;  // keep original GridSample
            }
            // NEAREST + BORDER + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER && !attrs.align_corners) {
                return false;
            }
            // NEAREST + ZEROS + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS && !attrs.align_corners) {
                return false;
            }
            // For non-f32 cases in these problematic modes, explicitly convert to f32 -> GridSample -> convert back.
            if ((!is_f32_data || !is_f32_grid) && attrs.mode == op::v9::GridSample::InterpolationMode::NEAREST &&
                (attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION ||
                 attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER ||
                 attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS) &&
                !attrs.align_corners) {
                auto data_f32 = std::make_shared<op::v0::Convert>(grid_sample->input_value(0), element::f32);
                auto grid_f32 = std::make_shared<op::v0::Convert>(grid_sample->input_value(1), element::f32);
                auto gs_f32 = std::make_shared<op::v9::GridSample>(data_f32, grid_f32, attrs);
                auto out = std::make_shared<op::v0::Convert>(gs_f32, grid_sample->get_output_element_type(0));
                out->set_friendly_name(grid_sample->get_friendly_name());
                copy_rt_to_subgraph(grid_sample, out);
                ov::replace_node_update_name(grid_sample, out);
                return true;
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_nearest_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionNearestDynamic"),
                         callback);
    }
};

class GridSampleDecompositionBilinearDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBilinearDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BILINEAR) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || !gs->get_input_partial_shape(1).is_static();
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_bilinear_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBilinearDynamic"),
                         callback);
    }
};

class GridSampleDecompositionBicubicDynamic : public ov::pass::MatcherPass {
public:
    GridSampleDecompositionBicubicDynamic() {
        // Match GridSample with 4D inputs (can be dynamic shapes)
        auto data_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto grid_pattern = ov::pass::pattern::any_input(ov::pass::pattern::rank_equals(4));
        auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>(
            {data_pattern, grid_pattern},
            [](const Output<Node>& output) {
                auto gs = ov::as_type_ptr<op::v9::GridSample>(output.get_node_shared_ptr());
                if (!gs || gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BICUBIC) {
                    return false;
                }
                // Only match if at least one shape is dynamic
                return !gs->get_input_partial_shape(0).is_static() || !gs->get_input_partial_shape(1).is_static();
            });

        matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
            auto grid_sample = std::dynamic_pointer_cast<op::v9::GridSample>(matcher.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            // Use reference path for known-problematic combos to avoid accuracy issues
            const auto& attrs = grid_sample->get_attributes();
            const bool is_f32_data = grid_sample->get_input_element_type(0) == element::f32;
            const bool is_f32_grid = grid_sample->get_input_element_type(1) == element::f32;
            // BICUBIC + ZEROS + align_corners=false (restrict to f32/f32)
            if (is_f32_data && is_f32_grid && attrs.mode == op::v9::GridSample::InterpolationMode::BICUBIC &&
                attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS && !attrs.align_corners) {
                return false;  // keep original GridSample
            }
            return decompose_impl(grid_sample, [&](const Ctx& context, const op::v9::GridSample::Attributes& attrs) {
                return build_bicubic_nhwc(context, attrs);
            });
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBicubicDynamic"),
                         callback);
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
