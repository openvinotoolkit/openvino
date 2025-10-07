// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/validation_util.hpp"
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
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov::intel_cpu;

// ========== Helper function implementations ==========
namespace {
using namespace ov;
using namespace ov::op;

struct Ctx {
    std::shared_ptr<Node> n_dim, c_dim, h_in, w_in, h_out, w_out;
    element::Type element_type = element::dynamic;
    element::Type calc_type = element::f32;
    std::shared_ptr<Node> c0, c1, c2, c0_5;
    std::shared_ptr<Node> i32_0, i32_1, i32_2, i32_3, i32_neg1;
    std::shared_ptr<Node> axis_0, axis_3;
    std::shared_ptr<Node> to_nhwc_perm;
    std::shared_ptr<Node> x, y, x_norm, y_norm;
    std::shared_ptr<Node> data_nhwc, data_nchw;
};

std::shared_ptr<Node> to_f32(const std::shared_ptr<Node>& n) {
    return std::make_shared<v0::Convert>(n, element::f32);
}

void init_common_constants(Ctx& ctx) {
    ctx.c0 = v0::Constant::create(ctx.calc_type, {}, {0.0f});
    ctx.c1 = v0::Constant::create(ctx.calc_type, {}, {1.0f});
    ctx.c2 = v0::Constant::create(ctx.calc_type, {}, {2.0f});
    ctx.c0_5 = v0::Constant::create(ctx.calc_type, {}, {0.5f});
    ctx.i32_0 = v0::Constant::create(element::i32, {}, {0});
    ctx.i32_1 = v0::Constant::create(element::i32, {}, {1});
    ctx.i32_2 = v0::Constant::create(element::i32, {}, {2});
    ctx.i32_3 = v0::Constant::create(element::i32, {}, {3});
    ctx.i32_neg1 = v0::Constant::create(element::i32, {}, {-1});
    ctx.axis_0 = ctx.i32_0;
    ctx.axis_3 = ctx.i32_3;
    ctx.to_nhwc_perm = v0::Constant::create(element::i32, {4}, {0, 2, 3, 1});
}

bool build_ctx(const std::shared_ptr<v0::Parameter>& data, const std::shared_ptr<v0::Parameter>& grid, Ctx& ctx) {
    const auto& data_ps = data->get_partial_shape();
    const auto& grid_ps = grid->get_partial_shape();
    if ((data_ps.rank().is_static() && data_ps.rank().get_length() != 4) ||
        (grid_ps.rank().is_static() && grid_ps.rank().get_length() != 4)) {
        return false;
    }
    if (grid_ps.rank().is_static() && grid_ps.rank().get_length() == 4) {
        const auto& last_dim = grid_ps[3];
        if (last_dim.is_static() && last_dim.get_length() != 2) {
            return false;
        }
    }
    ctx.element_type = data->get_element_type();
    ctx.calc_type = element::f32;
    init_common_constants(ctx);
    if (data->get_partial_shape().is_static()) {
        const auto& s = data->get_partial_shape().to_shape();
        ctx.n_dim = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[0])});
        ctx.c_dim = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[1])});
        ctx.h_in = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[2])});
        ctx.w_in = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[3])});
    } else {
        auto dshape = std::make_shared<v3::ShapeOf>(data, element::i32);
        ctx.n_dim = std::make_shared<v8::Gather>(dshape, ctx.i32_0, ctx.axis_0);
        ctx.c_dim = std::make_shared<v8::Gather>(dshape, ctx.i32_1, ctx.axis_0);
        ctx.h_in = std::make_shared<v8::Gather>(dshape, ctx.i32_2, ctx.axis_0);
        ctx.w_in = std::make_shared<v8::Gather>(dshape, ctx.i32_3, ctx.axis_0);
    }
    if (grid->get_partial_shape().is_static()) {
        const auto& s = grid->get_partial_shape().to_shape();
        ctx.h_out = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[1])});
        ctx.w_out = v0::Constant::create(element::i32, {}, {static_cast<int32_t>(s[2])});
    } else {
        auto gshape = std::make_shared<v3::ShapeOf>(grid, element::i32);
        ctx.h_out = std::make_shared<v8::Gather>(gshape, ctx.i32_1, ctx.axis_0);
        ctx.w_out = std::make_shared<v8::Gather>(gshape, ctx.i32_2, ctx.axis_0);
    }
    std::shared_ptr<Node> grid_any = grid;
    if (grid->get_element_type() != ctx.calc_type) {
        grid_any = std::make_shared<v0::Convert>(grid, ctx.calc_type);
    }
    ctx.x_norm = std::make_shared<v8::Gather>(grid_any, ctx.i32_0, ctx.axis_3);
    ctx.y_norm = std::make_shared<v8::Gather>(grid_any, ctx.i32_1, ctx.axis_3);
    ctx.data_nchw = data;
    ctx.data_nhwc = std::make_shared<v1::Transpose>(data, ctx.to_nhwc_perm);
    return true;
}

void normalize_grid_to_pixels(const Ctx& ctx,
                              const std::shared_ptr<Node>& xg,
                              const std::shared_ptr<Node>& yg,
                              const std::shared_ptr<Node>& w,
                              const std::shared_ptr<Node>& h,
                              bool align_corners,
                              std::shared_ptr<Node>& xo,
                              std::shared_ptr<Node>& yo) {
    if (align_corners) {
        auto wm1 = std::make_shared<v1::Subtract>(w, ctx.c1);
        auto hm1 = std::make_shared<v1::Subtract>(h, ctx.c1);
        xo = std::make_shared<v1::Multiply>(std::make_shared<v1::Divide>(std::make_shared<v1::Add>(xg, ctx.c1), ctx.c2),
                                            wm1);
        yo = std::make_shared<v1::Multiply>(std::make_shared<v1::Divide>(std::make_shared<v1::Add>(yg, ctx.c1), ctx.c2),
                                            hm1);
    } else {
        auto x_p1_w = std::make_shared<v1::Multiply>(std::make_shared<v1::Add>(xg, ctx.c1), w);
        auto y_p1_h = std::make_shared<v1::Multiply>(std::make_shared<v1::Add>(yg, ctx.c1), h);
        xo = std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(x_p1_w, ctx.c1), ctx.c2);
        yo = std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(y_p1_h, ctx.c1), ctx.c2);
    }
}

std::shared_ptr<Node> create_batch_indices(const Ctx& ctx) {
    auto range = std::make_shared<v4::Range>(ctx.i32_0, ctx.n_dim, ctx.i32_1, element::i32);
    auto n_dim_1d_bs = std::make_shared<v0::Unsqueeze>(ctx.n_dim, ctx.i32_0);
    auto batch_shape = std::make_shared<v0::Concat>(OutputVector{n_dim_1d_bs,
                                                                 v0::Constant::create(element::i32, {1}, {1}),
                                                                 v0::Constant::create(element::i32, {1}, {1})},
                                                    0);
    auto batch_indices = std::make_shared<v1::Reshape>(range, batch_shape, false);
    auto n_dim_1d = std::make_shared<v0::Unsqueeze>(ctx.n_dim, ctx.i32_0);
    auto h_out_1d = std::make_shared<v0::Unsqueeze>(ctx.h_out, ctx.i32_0);
    auto w_out_1d = std::make_shared<v0::Unsqueeze>(ctx.w_out, ctx.i32_0);
    auto bshape = std::make_shared<v0::Concat>(OutputVector{n_dim_1d, h_out_1d, w_out_1d}, 0);
    auto bcast = std::make_shared<v3::Broadcast>(batch_indices, bshape);
    return std::make_shared<v0::Convert>(bcast, element::i32);
}

std::shared_ptr<Node> gather_hw_nhwc(const Ctx& ctx, const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& y) {
    auto xi = std::make_shared<v0::Convert>(x, element::i32);
    auto yi = std::make_shared<v0::Convert>(y, element::i32);
    auto bidx = create_batch_indices(ctx);
    auto b = std::make_shared<v0::Unsqueeze>(bidx, ctx.i32_neg1);
    auto yy = std::make_shared<v0::Unsqueeze>(yi, ctx.i32_neg1);
    auto xx = std::make_shared<v0::Unsqueeze>(xi, ctx.i32_neg1);
    auto indices = std::make_shared<v0::Concat>(OutputVector{b, yy, xx}, -1);
    return std::make_shared<v8::GatherND>(ctx.data_nhwc, indices, 0);
}

std::shared_ptr<Node> reflect_coord(const Ctx& ctx,
                                    const std::shared_ptr<Node>& coord,
                                    const std::shared_ptr<Node>& size_f,
                                    bool align_corners) {
    auto size_is_one = std::make_shared<v1::Equal>(size_f, ctx.c1);
    if (const auto size_flag = ov::util::get_constant_from_source(size_is_one)) {
        const auto mask = size_flag->cast_vector<bool>();
        if (!mask.empty() && mask.front()) {
            return ctx.c0;
        }
    }
    if (align_corners) {
        auto size_m1 = std::make_shared<v1::Subtract>(size_f, ctx.c1);
        auto period = std::make_shared<v1::Multiply>(ctx.c2, size_m1);
        auto period_ok = std::make_shared<v1::Maximum>(period, ctx.c1);
        auto r = std::make_shared<v1::FloorMod>(coord, period_ok);
        auto d = std::make_shared<v1::Subtract>(r, size_m1);
        auto ad = std::make_shared<v0::Abs>(d);
        auto ref = std::make_shared<v1::Subtract>(size_m1, ad);
        return std::make_shared<v1::Select>(size_is_one, ctx.c0, ref);
    }
    auto is_negative = std::make_shared<v1::Less>(coord, ctx.c0);
    auto neg_coord = std::make_shared<v0::Negative>(coord);
    auto neg_minus_one = std::make_shared<v1::Subtract>(neg_coord, ctx.c1);
    auto coord_positive = std::make_shared<v1::Select>(is_negative, neg_minus_one, coord);
    auto two_size = std::make_shared<v1::Multiply>(ctx.c2, size_f);
    auto two_size_safe = std::make_shared<v1::Maximum>(two_size, ctx.c1);
    auto mod = std::make_shared<v1::FloorMod>(coord_positive, two_size_safe);
    auto need_ref = std::make_shared<v1::GreaterEqual>(mod, size_f);
    auto two_size_m1 = std::make_shared<v1::Subtract>(two_size_safe, ctx.c1);
    auto ref_val = std::make_shared<v1::Subtract>(two_size_m1, mod);
    auto res = std::make_shared<v1::Select>(need_ref, ref_val, mod);
    return std::make_shared<v1::Select>(size_is_one, ctx.c0, res);
}

std::shared_ptr<Node> reflect_index(const Ctx& ctx,
                                    const std::shared_ptr<Node>& idx,
                                    const std::shared_ptr<Node>& size_f,
                                    bool align_corners) {
    auto size_is_one = std::make_shared<v1::Equal>(size_f, ctx.c1);
    if (const auto size_flag = ov::util::get_constant_from_source(size_is_one)) {
        const auto mask = size_flag->cast_vector<bool>();
        if (!mask.empty() && mask.front()) {
            return ctx.c0;
        }
    }
    if (align_corners) {
        auto size_m1 = std::make_shared<v1::Subtract>(size_f, ctx.c1);
        auto two_sm1 = std::make_shared<v1::Multiply>(ctx.c2, size_m1);
        auto period_safe = std::make_shared<v1::Maximum>(two_sm1, ctx.c1);
        auto mod1 = std::make_shared<v1::FloorMod>(idx, period_safe);
        auto mod2 = std::make_shared<v1::FloorMod>(std::make_shared<v1::Add>(mod1, period_safe), period_safe);
        auto flipped = std::make_shared<v1::Subtract>(two_sm1, mod2);
        auto cond = std::make_shared<v1::GreaterEqual>(mod2, size_f);
        auto ref = std::make_shared<v1::Select>(cond, flipped, mod2);
        return std::make_shared<v1::Select>(size_is_one, ctx.c0, ref);
    }
    auto two_s = std::make_shared<v1::Multiply>(ctx.c2, size_f);
    auto two_s_safe = std::make_shared<v1::Maximum>(two_s, ctx.c1);
    auto mod1 = std::make_shared<v1::FloorMod>(idx, two_s_safe);
    auto mod2 = std::make_shared<v1::FloorMod>(std::make_shared<v1::Add>(mod1, two_s_safe), two_s_safe);
    auto need_rf = std::make_shared<v1::GreaterEqual>(mod2, size_f);
    auto two_s_m1 = std::make_shared<v1::Subtract>(two_s_safe, ctx.c1);
    auto ref_val = std::make_shared<v1::Subtract>(two_s_m1, mod2);
    auto res = std::make_shared<v1::Select>(need_rf, ref_val, mod2);
    return std::make_shared<v1::Select>(size_is_one, ctx.c0, res);
}

static std::shared_ptr<Node> inside_mask_indexed(const Ctx& ctx,
                                                 const std::shared_ptr<Node>& xi,
                                                 const std::shared_ptr<Node>& yi,
                                                 const std::shared_ptr<Node>& w_in_f,
                                                 const std::shared_ptr<Node>& h_in_f);

void clamp_indices_inplace(const Ctx& ctx,
                           std::shared_ptr<Node>& xi,
                           std::shared_ptr<Node>& yi,
                           const std::shared_ptr<Node>& w_in_f,
                           const std::shared_ptr<Node>& h_in_f) {
    auto w_m1 = std::make_shared<v1::Subtract>(w_in_f, ctx.c1);
    auto h_m1 = std::make_shared<v1::Subtract>(h_in_f, ctx.c1);
    xi = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(xi, ctx.c0), w_m1);
    yi = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(yi, ctx.c0), h_m1);
}

std::shared_ptr<Node> build_nearest_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_zeros = attrs.padding_mode == v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == v9::GridSample::PaddingMode::REFLECTION;
    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);
    std::shared_ptr<Node> x_idx = std::make_shared<v5::Round>(ctx.x, v5::Round::RoundMode::HALF_TO_EVEN);
    std::shared_ptr<Node> y_idx = std::make_shared<v5::Round>(ctx.y, v5::Round::RoundMode::HALF_TO_EVEN);
    auto x_unclamped = x_idx;
    auto y_unclamped = y_idx;
    if (is_reflection) {
        x_idx = reflect_index(ctx, x_idx, w_in_f, attrs.align_corners);
        y_idx = reflect_index(ctx, y_idx, h_in_f, attrs.align_corners);
    }
    clamp_indices_inplace(ctx, x_idx, y_idx, w_in_f, h_in_f);
    auto out = gather_hw_nhwc(ctx, x_idx, y_idx);
    if (is_zeros) {
        auto ok = inside_mask_indexed(ctx, x_unclamped, y_unclamped, w_in_f, h_in_f);
        auto mask = std::make_shared<v0::Convert>(ok, ctx.calc_type);
        auto maske = std::make_shared<v0::Unsqueeze>(mask, ctx.i32_neg1);
        if (ctx.element_type != ctx.calc_type) {
            out = std::make_shared<v0::Convert>(out, ctx.calc_type);
        }
        out = std::make_shared<v1::Multiply>(out, maske);
        if (ctx.element_type != ctx.calc_type) {
            out = std::make_shared<v0::Convert>(out, ctx.element_type);
        }
    }
    return out;
}

// inside mask for zeros (same as in transformation)
static std::shared_ptr<Node> inside_mask_indexed(const Ctx& ctx,
                                                 const std::shared_ptr<Node>& xi,
                                                 const std::shared_ptr<Node>& yi,
                                                 const std::shared_ptr<Node>& w_in_f,
                                                 const std::shared_ptr<Node>& h_in_f) {
    auto x_ge_0 = std::make_shared<v1::GreaterEqual>(xi, ctx.c0);
    auto y_ge_0 = std::make_shared<v1::GreaterEqual>(yi, ctx.c0);
    auto x_lt_w = std::make_shared<v1::Less>(xi, w_in_f);
    auto y_lt_h = std::make_shared<v1::Less>(yi, h_in_f);
    return std::make_shared<v1::LogicalAnd>(std::make_shared<v1::LogicalAnd>(x_ge_0, x_lt_w),
                                            std::make_shared<v1::LogicalAnd>(y_ge_0, y_lt_h));
}

std::shared_ptr<Node> build_bilinear_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_border = attrs.padding_mode == v9::GridSample::PaddingMode::BORDER;
    const bool is_zeros = attrs.padding_mode == v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == v9::GridSample::PaddingMode::REFLECTION;

    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);

    std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
    if (is_reflection && attrs.align_corners) {
        x_pad = reflect_coord(ctx, ctx.x, w_in_f, true);
        y_pad = reflect_coord(ctx, ctx.y, h_in_f, true);
    }

    std::shared_ptr<Node> x0 = std::make_shared<v0::Floor>(x_pad);
    std::shared_ptr<Node> y0 = std::make_shared<v0::Floor>(y_pad);
    std::shared_ptr<Node> x1 = std::make_shared<v1::Add>(x0, ctx.c1);
    std::shared_ptr<Node> y1 = std::make_shared<v1::Add>(y0, ctx.c1);

    std::shared_ptr<Node> dx = std::make_shared<v1::Subtract>(x_pad, x0);
    std::shared_ptr<Node> dy = std::make_shared<v1::Subtract>(y_pad, y0);
    if (!is_zeros) {
        auto eps = v0::Constant::create(ctx.calc_type, {}, {1e-7F});
        auto one_m_eps = std::make_shared<v1::Subtract>(ctx.c1, eps);
        dx = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(dx, ctx.c0), one_m_eps);
        dy = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(dy, ctx.c0), one_m_eps);
    }
    auto omdx = std::make_shared<v1::Subtract>(ctx.c1, dx);
    auto omdy = std::make_shared<v1::Subtract>(ctx.c1, dy);

    std::vector<std::shared_ptr<Node>> weights{std::make_shared<v1::Multiply>(omdx, omdy),
                                               std::make_shared<v1::Multiply>(dx, omdy),
                                               std::make_shared<v1::Multiply>(omdx, dy),
                                               std::make_shared<v1::Multiply>(dx, dy)};

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
            v = std::make_shared<v0::Convert>(v, ctx.calc_type);
        }
        if (is_zeros) {
            auto ok = inside_mask_indexed(ctx, xs[i], ys[i], w_in_f, h_in_f);
            auto mask = std::make_shared<v0::Convert>(ok, ctx.calc_type);
            v = std::make_shared<v1::Multiply>(v, std::make_shared<v0::Unsqueeze>(mask, ctx.i32_neg1));
        }
        auto wexp = std::make_shared<v0::Unsqueeze>(weights[i], ctx.i32_neg1);
        auto tap = std::make_shared<v1::Multiply>(v, wexp);
        res = res ? std::shared_ptr<Node>(std::make_shared<v1::Add>(res, tap)) : tap;
    }
    if (ctx.element_type != ctx.calc_type) {
        res = std::make_shared<v0::Convert>(res, ctx.element_type);
    }
    return res;
}

// Bicubic: mirror the transformation structure (weights + 4x4 taps accumulation)
std::shared_ptr<Node> build_bicubic_nhwc(const Ctx& ctx, const ov::op::v9::GridSample::Attributes& attrs) {
    const bool is_border = attrs.padding_mode == v9::GridSample::PaddingMode::BORDER;
    const bool is_zeros = attrs.padding_mode == v9::GridSample::PaddingMode::ZEROS;
    const bool is_reflection = attrs.padding_mode == v9::GridSample::PaddingMode::REFLECTION;
    auto w_in_f = to_f32(ctx.w_in);
    auto h_in_f = to_f32(ctx.h_in);
    std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
    if (is_reflection && attrs.align_corners) {
        x_pad = reflect_coord(ctx, ctx.x, w_in_f, true);
        y_pad = reflect_coord(ctx, ctx.y, h_in_f, true);
    }
    auto x0 = std::make_shared<v0::Floor>(x_pad);
    auto y0 = std::make_shared<v0::Floor>(y_pad);
    auto dx_raw = std::make_shared<v1::Subtract>(x_pad, x0);
    auto dy_raw = std::make_shared<v1::Subtract>(y_pad, y0);
    std::shared_ptr<Node> dx = dx_raw;
    std::shared_ptr<Node> dy = dy_raw;
    if (!is_zeros) {
        auto eps = v0::Constant::create(ctx.calc_type, {}, {1e-7F});
        auto one_m_eps = std::make_shared<v1::Subtract>(ctx.c1, eps);
        dx = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(dx_raw, ctx.c0), one_m_eps);
        dy = std::make_shared<v1::Minimum>(std::make_shared<v1::Maximum>(dy_raw, ctx.c0), one_m_eps);
    }
    auto w_is_one = std::make_shared<v1::Equal>(w_in_f, ctx.c1);
    auto h_is_one = std::make_shared<v1::Equal>(h_in_f, ctx.c1);
    dx = std::make_shared<v1::Select>(w_is_one, ctx.c0, dx);
    dy = std::make_shared<v1::Select>(h_is_one, ctx.c0, dy);
    auto a = v0::Constant::create(ctx.calc_type, {}, {-0.75F});
    auto c3 = v0::Constant::create(ctx.calc_type, {}, {3.0F});
    auto c4 = v0::Constant::create(ctx.calc_type, {}, {4.0F});
    auto c5 = v0::Constant::create(ctx.calc_type, {}, {5.0F});
    auto c8 = v0::Constant::create(ctx.calc_type, {}, {8.0F});
    auto dx_p1 = std::make_shared<v1::Add>(dx, ctx.c1);
    auto one_m_dx = std::make_shared<v1::Subtract>(ctx.c1, dx);
    auto two_m_dx = std::make_shared<v1::Subtract>(ctx.c2, dx);
    auto dx2 = std::make_shared<v1::Multiply>(dx, dx);
    auto dx_p1_2 = std::make_shared<v1::Multiply>(dx_p1, dx_p1);
    auto one_m_dx_2 = std::make_shared<v1::Multiply>(one_m_dx, one_m_dx);
    auto two_m_dx_2 = std::make_shared<v1::Multiply>(two_m_dx, two_m_dx);
    auto a_dx_p1 = std::make_shared<v1::Multiply>(a, dx_p1);
    auto a5 = std::make_shared<v1::Multiply>(c5, a);
    auto term_v0_1 = std::make_shared<v1::Subtract>(a_dx_p1, a5);
    auto term_v0_2 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(term_v0_1, dx_p1),
                                               std::make_shared<v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_x0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(term_v0_2, dx_p1),
                                                                std::make_shared<v1::Multiply>(c4, a));
    auto ap2 = std::make_shared<v1::Add>(a, ctx.c2);
    auto ap3 = std::make_shared<v1::Add>(a, c3);
    std::shared_ptr<Node> w_x1 = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(ap2, dx), ap3),
                                       dx2),
        ctx.c1);
    std::shared_ptr<Node> w_x2 = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(
            std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(ap2, one_m_dx), ap3),
            one_m_dx_2),
        ctx.c1);
    auto a_two_m_dx = std::make_shared<v1::Multiply>(a, two_m_dx);
    auto term_v3_1 = std::make_shared<v1::Subtract>(a_two_m_dx, a5);
    auto term_v3_2 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(term_v3_1, two_m_dx),
                                               std::make_shared<v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_x3 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(term_v3_2, two_m_dx),
                                                                std::make_shared<v1::Multiply>(c4, a));
    w_x0 = std::make_shared<v1::Select>(w_is_one, ctx.c0, w_x0);
    w_x1 = std::make_shared<v1::Select>(w_is_one, ctx.c1, w_x1);
    w_x2 = std::make_shared<v1::Select>(w_is_one, ctx.c0, w_x2);
    w_x3 = std::make_shared<v1::Select>(w_is_one, ctx.c0, w_x3);
    auto dy_p1 = std::make_shared<v1::Add>(dy, ctx.c1);
    auto one_m_dy = std::make_shared<v1::Subtract>(ctx.c1, dy);
    auto two_m_dy = std::make_shared<v1::Subtract>(ctx.c2, dy);
    auto dy2 = std::make_shared<v1::Multiply>(dy, dy);
    auto dy_p1_2 = std::make_shared<v1::Multiply>(dy_p1, dy_p1);
    auto one_m_dy_2 = std::make_shared<v1::Multiply>(one_m_dy, one_m_dy);
    auto two_m_dy_2 = std::make_shared<v1::Multiply>(two_m_dy, two_m_dy);
    auto a_dy_p1 = std::make_shared<v1::Multiply>(a, dy_p1);
    auto term_wy0_1 = std::make_shared<v1::Subtract>(a_dy_p1, a5);
    auto term_wy0_2 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(term_wy0_1, dy_p1),
                                                std::make_shared<v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_y0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(term_wy0_2, dy_p1),
                                                                std::make_shared<v1::Multiply>(c4, a));
    std::shared_ptr<Node> w_y1 = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(ap2, dy), ap3),
                                       dy2),
        ctx.c1);
    std::shared_ptr<Node> w_y2 = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(
            std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(ap2, one_m_dy), ap3),
            one_m_dy_2),
        ctx.c1);
    auto a_two_m_dy = std::make_shared<v1::Multiply>(a, two_m_dy);
    auto term_wy3_1 = std::make_shared<v1::Subtract>(a_two_m_dy, a5);
    auto term_wy3_2 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(term_wy3_1, two_m_dy),
                                                std::make_shared<v1::Multiply>(c8, a));
    std::shared_ptr<Node> w_y3 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(term_wy3_2, two_m_dy),
                                                                std::make_shared<v1::Multiply>(c4, a));
    w_y0 = std::make_shared<v1::Select>(h_is_one, ctx.c0, w_y0);
    w_y1 = std::make_shared<v1::Select>(h_is_one, ctx.c1, w_y1);
    w_y2 = std::make_shared<v1::Select>(h_is_one, ctx.c0, w_y2);
    w_y3 = std::make_shared<v1::Select>(h_is_one, ctx.c0, w_y3);
    std::vector<std::shared_ptr<Node>> x_idx{
        std::make_shared<v1::Subtract>(x0, ctx.c1),
        x0,
        std::make_shared<v1::Add>(x0, ctx.c1),
        std::make_shared<v1::Add>(x0, ctx.c2),
    };
    std::vector<std::shared_ptr<Node>> y_idx{
        std::make_shared<v1::Subtract>(y0, ctx.c1),
        y0,
        std::make_shared<v1::Add>(y0, ctx.c1),
        std::make_shared<v1::Add>(y0, ctx.c2),
    };
    std::vector<std::shared_ptr<Node>> wx{w_x0, w_x1, w_x2, w_x3};
    std::vector<std::shared_ptr<Node>> wy{w_y0, w_y1, w_y2, w_y3};
    std::vector<std::shared_ptr<Node>> x_ok(4), y_ok(4);
    if (is_zeros) {
        for (int i = 0; i < 4; ++i) {
            x_ok[i] = std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(x_idx[i], ctx.c0),
                                                       std::make_shared<v1::Less>(x_idx[i], w_in_f));
            y_ok[i] = std::make_shared<v1::LogicalAnd>(std::make_shared<v1::GreaterEqual>(y_idx[i], ctx.c0),
                                                       std::make_shared<v1::Less>(y_idx[i], h_in_f));
        }
    }
    if (is_reflection) {
        for (int i = 0; i < 4; ++i) {
            x_idx[i] = reflect_index(ctx, x_idx[i], w_in_f, attrs.align_corners);
            y_idx[i] = reflect_index(ctx, y_idx[i], h_in_f, attrs.align_corners);
        }
    } else {
        auto w_is_one_chk = std::make_shared<v1::Equal>(w_in_f, ctx.c1);
        auto h_is_one_chk = std::make_shared<v1::Equal>(h_in_f, ctx.c1);
        for (int i = 0; i < 4; ++i) {
            x_idx[i] = std::make_shared<v1::Select>(w_is_one_chk, ctx.c0, x_idx[i]);
            y_idx[i] = std::make_shared<v1::Select>(h_is_one_chk, ctx.c0, y_idx[i]);
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
                val = std::make_shared<v0::Convert>(val, ctx.calc_type);
            }
            auto wxy = std::make_shared<v1::Multiply>(wx[ix], wy[iy]);
            if (is_zeros) {
                auto ok = std::make_shared<v1::LogicalAnd>(x_ok[ix], y_ok[iy]);
                auto mask = std::make_shared<v0::Convert>(ok, ctx.calc_type);
                val = std::make_shared<v1::Multiply>(val, std::make_shared<v0::Unsqueeze>(mask, ctx.i32_neg1));
            }
            auto wexp = std::make_shared<v0::Unsqueeze>(wxy, ctx.i32_neg1);
            auto tap = std::make_shared<v1::Multiply>(val, wexp);
            row = row ? std::shared_ptr<Node>(std::make_shared<v1::Add>(row, tap)) : tap;
        }
        sum = sum ? std::shared_ptr<Node>(std::make_shared<v1::Add>(sum, row)) : row;
    }
    if (ctx.element_type != ctx.calc_type) {
        sum = std::make_shared<v0::Convert>(sum, ctx.element_type);
    }
    return sum;
}
}  // namespace

static std::shared_ptr<ov::Model> create_expected_decomposed_pattern(const ov::PartialShape& data_shape,
                                                                     const ov::PartialShape& grid_shape,
                                                                     const ov::element::Type& data_type,
                                                                     const ov::element::Type& grid_type,
                                                                     const ov::op::v9::GridSample::Attributes& attrs) {
    auto data = std::make_shared<ov::op::v0::Parameter>(data_type, data_shape);
    auto grid = std::make_shared<ov::op::v0::Parameter>(grid_type, grid_shape);

    // Replicate transformation decision points: for some NEAREST combos keep original
    const bool is_f32_data = data_type == ov::element::f32;
    const bool is_f32_grid = grid_type == ov::element::f32;
    const bool nearest_reflection_border_bad = attrs.mode == ov::op::v9::GridSample::InterpolationMode::NEAREST &&
                                               (attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::REFLECTION ||
                                                attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::BORDER) &&
                                               !attrs.align_corners;
    const bool nearest_zeros_bad = attrs.mode == ov::op::v9::GridSample::InterpolationMode::NEAREST &&
                                   attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::ZEROS &&
                                   !attrs.align_corners;
    const bool bicubic_zeros_bad = attrs.mode == ov::op::v9::GridSample::InterpolationMode::BICUBIC &&
                                   attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::ZEROS &&
                                   !attrs.align_corners;

    std::shared_ptr<ov::Node> out_node;
    if (is_f32_data && is_f32_grid && (nearest_reflection_border_bad || nearest_zeros_bad || bicubic_zeros_bad)) {
        // Keep original GridSample
        out_node = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
    } else if ((!is_f32_data || !is_f32_grid) && nearest_reflection_border_bad) {
        // Fallback: convert to f32, use GridSample, convert back
        auto data_f32 = std::make_shared<ov::op::v0::Convert>(data, ov::element::f32);
        auto grid_f32 = std::make_shared<ov::op::v0::Convert>(grid, ov::element::f32);
        auto gs = std::make_shared<ov::op::v9::GridSample>(data_f32, grid_f32, attrs);
        out_node = std::make_shared<ov::op::v0::Convert>(gs, data_type);
    } else {
        // Manual decomposition path (generic dynamic)
        Ctx ctx{};
        if (!build_ctx(data, grid, ctx)) {
            // Fallback to original if shapes are not 4D
            out_node = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
        } else {
            auto w_in_f = to_f32(ctx.w_in);
            auto h_in_f = to_f32(ctx.h_in);
            normalize_grid_to_pixels(ctx, ctx.x_norm, ctx.y_norm, w_in_f, h_in_f, attrs.align_corners, ctx.x, ctx.y);
            std::shared_ptr<ov::Node> result_nhwc;
            switch (attrs.mode) {
            case ov::op::v9::GridSample::InterpolationMode::NEAREST:
                result_nhwc = build_nearest_nhwc(ctx, attrs);
                break;
            case ov::op::v9::GridSample::InterpolationMode::BILINEAR:
                result_nhwc = build_bilinear_nhwc(ctx, attrs);
                break;
            case ov::op::v9::GridSample::InterpolationMode::BICUBIC:
                result_nhwc = build_bicubic_nhwc(ctx, attrs);
                break;
            }
            auto to_nchw = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
            out_node = std::make_shared<ov::op::v1::Transpose>(result_nhwc, to_nchw);
        }
    }

    out_node->set_friendly_name("GridSample_ref");
    auto result = std::make_shared<ov::op::v0::Result>(out_node);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, grid});
}

// ========== Test parameters structure ==========
struct GridSampleTestParams {
    ov::PartialShape data_shape;
    ov::PartialShape grid_shape;
    ov::element::Type data_type;
    ov::element::Type grid_type;
    bool align_corners;
    ov::op::v9::GridSample::InterpolationMode interp_mode;
    ov::op::v9::GridSample::PaddingMode padding_mode;
};

// ========== Base test classes ==========
class GridSampleDecompositionTest : public TransformationTestsF, public WithParamInterface<GridSampleTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GridSampleTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "data_shape=" << p.data_shape << "_grid_shape=" << p.grid_shape << "_data_type=" << p.data_type
               << "_grid_type=" << p.grid_type << "_align=" << p.align_corners
               << "_interp=" << ov::as_string(p.interp_mode) << "_padding=" << ov::as_string(p.padding_mode);
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        // Use default comparator; structural equality is not strictly enforced here

        const auto& p = GetParam();
        auto data = std::make_shared<ov::op::v0::Parameter>(p.data_type, p.data_shape);
        auto grid = std::make_shared<ov::op::v0::Parameter>(p.grid_type, p.grid_shape);

        ov::op::v9::GridSample::Attributes attrs{p.align_corners, p.interp_mode, p.padding_mode};
        auto grid_sample = std::make_shared<ov::op::v9::GridSample>(data, grid, attrs);
        auto result = std::make_shared<ov::op::v0::Result>(grid_sample);
        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data, grid});

        model_ref = create_expected_decomposed_pattern(p.data_shape, p.grid_shape, p.data_type, p.grid_type, attrs);
        manager.register_pass<ov::intel_cpu::GridSampleDecomposition>();
    }
};

TEST_P(GridSampleDecompositionTest, CompareGraphs) {}

// ========== Test parameters ==========
const std::vector<GridSampleTestParams> testStaticShapes = {
    // BILINEAR + BORDER - static shapes
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    {{2, 3, 8, 8},
     {2, 5, 5, 2},
     ov::element::f32,
     ov::element::f32,
     true,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // NEAREST mode
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::NEAREST,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // BICUBIC mode
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BICUBIC,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // ZEROS padding
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::ZEROS},

    // REFLECTION padding
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::REFLECTION},

    // Different data types
    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::f16,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    {{1, 2, 4, 4},
     {1, 3, 3, 2},
     ov::element::i32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // Large dimensions
    {{4, 16, 64, 64},
     {4, 32, 32, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // Edge cases
    {{1, 1, 1, 1},
     {1, 1, 1, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // Typical CV size
    {{10, 3, 224, 224},
     {10, 112, 112, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // All combinations for comprehensive coverage
    {{2, 3, 5, 7},
     {2, 4, 6, 2},
     ov::element::f32,
     ov::element::f32,
     true,
     ov::op::v9::GridSample::InterpolationMode::NEAREST,
     ov::op::v9::GridSample::PaddingMode::ZEROS},

    {{1, 1, 8, 8},
     {1, 4, 4, 2},
     ov::element::f16,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::NEAREST,
     ov::op::v9::GridSample::PaddingMode::REFLECTION},

    {{3, 2, 6, 10},
     {3, 5, 8, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BICUBIC,
     ov::op::v9::GridSample::PaddingMode::ZEROS},

    {{1, 4, 12, 16},
     {1, 10, 14, 2},
     ov::element::f32,
     ov::element::f32,
     true,
     ov::op::v9::GridSample::InterpolationMode::BICUBIC,
     ov::op::v9::GridSample::PaddingMode::REFLECTION},
};

const std::vector<GridSampleTestParams> testDynamicShapes = {
    // Dynamic batch dimension
    {{-1, 3, 8, 8},
     {-1, 4, 4, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // Dynamic spatial dimensions
    {{2, 3, -1, -1},
     {2, -1, -1, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::BILINEAR,
     ov::op::v9::GridSample::PaddingMode::BORDER},

    // Fully dynamic
    {{-1, -1, -1, -1},
     {-1, -1, -1, 2},
     ov::element::f32,
     ov::element::f32,
     false,
     ov::op::v9::GridSample::InterpolationMode::NEAREST,
     ov::op::v9::GridSample::PaddingMode::ZEROS},

    // Dynamic with different modes
    {{-1, 3, -1, -1},
     {-1, -1, -1, 2},
     ov::element::f32,
     ov::element::f32,
     true,
     ov::op::v9::GridSample::InterpolationMode::BICUBIC,
     ov::op::v9::GridSample::PaddingMode::REFLECTION},
};

INSTANTIATE_TEST_SUITE_P(StaticShapes,
                         GridSampleDecompositionTest,
                         ::testing::ValuesIn(testStaticShapes),
                         GridSampleDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DynamicShapes,
                         GridSampleDecompositionTest,
                         ::testing::ValuesIn(testDynamicShapes),
                         GridSampleDecompositionTest::getTestCaseName);
