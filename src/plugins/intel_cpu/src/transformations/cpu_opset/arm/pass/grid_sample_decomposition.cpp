// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"

#include <deque>
#include <functional>
#include <unordered_set>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/pass/tokenization.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/utils/utils.hpp"


namespace ov {
namespace intel_cpu {

namespace {  // ======= Common utilities and decomposition core (NOT mode-specific) =======

struct Ctx {
    // Shapes
    std::shared_ptr<Node> n_dim, c_dim, h_in, w_in, h_out, w_out;
    // Types
    element::Type element_type;
    element::Type calc_type;
    // Consts
    std::shared_ptr<Node> c0, c1, c2, c0_5;
    // Coordinates (normalized -> pixel coords, without padding/reflection)
    std::shared_ptr<Node> x, y;
    // Original normalized coordinates from grid [-1, 1]
    std::shared_ptr<Node> x_norm, y_norm;
    // Data NHWC for GatherND
    std::shared_ptr<Node> data_nhwc;
};

// Helpers that depend on context
static std::shared_ptr<Node> create_batch_indices(const Ctx& ctx) {
    auto range = std::make_shared<op::v4::Range>(
        op::v0::Constant::create(element::i32, {}, {0}),
        ctx.n_dim,
        op::v0::Constant::create(element::i32, {}, {1}),
        element::i32);

    auto n_dim_1d_bs = std::make_shared<op::v0::Unsqueeze>(ctx.n_dim, op::v0::Constant::create(element::i32, {}, {0}));
    auto batch_shape = std::make_shared<op::v0::Concat>(
        OutputVector{n_dim_1d_bs,
                     op::v0::Constant::create(element::i32, {1}, {1}),
                     op::v0::Constant::create(element::i32, {1}, {1})},
        0);

    auto batch_indices = std::make_shared<op::v1::Reshape>(range, batch_shape, false);

    auto n_dim_1d = std::make_shared<op::v0::Unsqueeze>(ctx.n_dim, op::v0::Constant::create(element::i32, {}, {0}));
    auto h_out_1d = std::make_shared<op::v0::Unsqueeze>(ctx.h_out, op::v0::Constant::create(element::i32, {}, {0}));
    auto w_out_1d = std::make_shared<op::v0::Unsqueeze>(ctx.w_out, op::v0::Constant::create(element::i32, {}, {0}));
    auto bshape   = std::make_shared<op::v0::Concat>(OutputVector{n_dim_1d, h_out_1d, w_out_1d}, 0);
    auto bcast    = std::make_shared<op::v3::Broadcast>(batch_indices, bshape);
    return std::make_shared<op::v0::Convert>(bcast, element::i32);
}

static std::shared_ptr<Node> gather_hw(const Ctx& ctx,
                                       const std::shared_ptr<Node>& x_coord,
                                       const std::shared_ptr<Node>& y_coord) {
    auto x_i32 = std::make_shared<op::v0::Convert>(x_coord, element::i32);
    auto y_i32 = std::make_shared<op::v0::Convert>(y_coord, element::i32);
    auto bidx  = create_batch_indices(ctx);

    auto b_exp = std::make_shared<op::v0::Unsqueeze>(bidx, op::v0::Constant::create(element::i32, {}, {-1}));
    auto y_exp = std::make_shared<op::v0::Unsqueeze>(y_i32, op::v0::Constant::create(element::i32, {}, {-1}));
    auto x_exp = std::make_shared<op::v0::Unsqueeze>(x_i32, op::v0::Constant::create(element::i32, {}, {-1}));

    auto indices = std::make_shared<op::v0::Concat>(OutputVector{b_exp, y_exp, x_exp}, -1);
    return std::make_shared<op::v8::GatherND>(ctx.data_nhwc, indices, 0);
}

// Reflection for continuous coordinates
static std::shared_ptr<Node> reflect_coord(const Ctx& ctx,
                                           const std::shared_ptr<Node>& coord,
                                           const std::shared_ptr<Node>& size_f,
                                           bool align_corners) {
    auto size_is_one = std::make_shared<op::v1::Equal>(size_f, ctx.c1);
    if (align_corners) {
        auto size_m1   = std::make_shared<op::v1::Subtract>(size_f, ctx.c1);
        auto period    = std::make_shared<op::v1::Multiply>(ctx.c2, size_m1);
        auto period_ok = std::make_shared<op::v1::Maximum>(period, ctx.c1);
        auto r   = std::make_shared<op::v1::FloorMod>(coord, period_ok);
        auto d   = std::make_shared<op::v1::Subtract>(r, size_m1);
        auto ad  = std::make_shared<op::v0::Abs>(d);
        auto ref = std::make_shared<op::v1::Subtract>(size_m1, ad);
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, ref);
    } else {
        // For align_corners=false: handle negative coords by transforming them
        // PyTorch formula: negative coords are transformed as -coord - 1
        auto is_negative = std::make_shared<op::v1::Less>(coord, ctx.c0);
        auto neg_coord = std::make_shared<op::v0::Negative>(coord);
        auto neg_minus_one = std::make_shared<op::v1::Subtract>(neg_coord, ctx.c1);
        auto coord_positive = std::make_shared<op::v1::Select>(is_negative, neg_minus_one, coord);
        
        auto two_size = std::make_shared<op::v1::Multiply>(ctx.c2, size_f);
        // Safe period to avoid division by zero
        auto two_size_safe = std::make_shared<op::v1::Maximum>(two_size, ctx.c1);
        
        // Use positive modulo on the transformed coordinate
        auto mod = std::make_shared<op::v1::FloorMod>(coord_positive, two_size_safe);
        
        auto need_ref = std::make_shared<op::v1::GreaterEqual>(mod, size_f);
        // Reflect: 2*size - 1 - mod
        auto two_size_m1 = std::make_shared<op::v1::Subtract>(two_size_safe, ctx.c1);
        auto ref_val = std::make_shared<op::v1::Subtract>(two_size_m1, mod);
        auto result = std::make_shared<op::v1::Select>(need_ref, ref_val, mod);
        
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, result);
    }
}

// Reflection for discrete indices (float tensors, will be converted to int32 where needed)
static std::shared_ptr<Node> reflect_index(const Ctx& ctx,
                                           const std::shared_ptr<Node>& idx,
                                           const std::shared_ptr<Node>& size_f,
                                           bool align_corners) {
    auto size_is_one = std::make_shared<op::v1::Equal>(size_f, ctx.c1);
    if (align_corners) {
        auto size_m1 = std::make_shared<op::v1::Subtract>(size_f, ctx.c1);
        auto two_sm1 = std::make_shared<op::v1::Multiply>(ctx.c2, size_m1);
        // Guard against division by zero when size==1
        auto period_safe = std::make_shared<op::v1::Maximum>(two_sm1, ctx.c1);
        // Use positive modulo: ((idx % period) + period) % period
        auto mod1    = std::make_shared<op::v1::FloorMod>(idx, period_safe);
        auto mod2    = std::make_shared<op::v1::FloorMod>(
                           std::make_shared<op::v1::Add>(mod1, period_safe), period_safe);
        auto flipped = std::make_shared<op::v1::Subtract>(two_sm1, mod2);
        auto cond    = std::make_shared<op::v1::GreaterEqual>(mod2, size_f);
        auto ref     = std::make_shared<op::v1::Select>(cond, flipped, mod2);
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, ref);
    } else {
        auto two_s   = std::make_shared<op::v1::Multiply>(ctx.c2, size_f);
        // Safe period to avoid division by zero
        auto two_s_safe = std::make_shared<op::v1::Maximum>(two_s, ctx.c1);
        auto per_m1  = std::make_shared<op::v1::Subtract>(two_s_safe, ctx.c1);
        auto mod1    = std::make_shared<op::v1::FloorMod>(idx, two_s_safe);
        auto mod2    = std::make_shared<op::v1::FloorMod>(std::make_shared<op::v1::Add>(mod1, two_s_safe), two_s_safe);
        auto need_rf = std::make_shared<op::v1::GreaterEqual>(mod2, size_f);
        auto ref_val = std::make_shared<op::v1::Subtract>(per_m1, mod2);
        auto res     = std::make_shared<op::v1::Select>(need_rf, ref_val, mod2);
        return std::make_shared<op::v1::Select>(size_is_one, ctx.c0, res);
    }
}

// Common core: prepares context, calls mode builder, does post-processing
static bool decompose_impl(
    const std::shared_ptr<ov::op::v9::GridSample>& gs,
    const std::function<std::shared_ptr<Node>(const Ctx&, const ov::op::v9::GridSample::Attributes&)>& build_mode_result_nhwc) {

    const auto& data = gs->input_value(0);
    const auto& grid = gs->input_value(1);

    // Only support 4D tensors
    if (data.get_partial_shape().rank().get_length() != 4 ||
        grid.get_partial_shape().rank().get_length() != 4) {
        return false;
    }

    Ctx ctx{};
    ctx.element_type = data.get_element_type();
    ctx.calc_type    = element::f32;  // Always use f32 for coordinate calculations

    // ShapeOf components
    auto dshape = std::make_shared<op::v3::ShapeOf>(data, element::i32);
    ctx.n_dim   = std::make_shared<op::v8::Gather>(dshape, op::v0::Constant::create(element::i32, {}, {0}),
                                               op::v0::Constant::create(element::i32, {}, {0}));
    ctx.c_dim   = std::make_shared<op::v8::Gather>(dshape, op::v0::Constant::create(element::i32, {}, {1}),
                                               op::v0::Constant::create(element::i32, {}, {0}));
    ctx.h_in    = std::make_shared<op::v8::Gather>(dshape, op::v0::Constant::create(element::i32, {}, {2}),
                                               op::v0::Constant::create(element::i32, {}, {0}));
    ctx.w_in    = std::make_shared<op::v8::Gather>(dshape, op::v0::Constant::create(element::i32, {}, {3}),
                                               op::v0::Constant::create(element::i32, {}, {0}));

    auto gshape = std::make_shared<op::v3::ShapeOf>(grid, element::i32);
    ctx.h_out   = std::make_shared<op::v8::Gather>(gshape, op::v0::Constant::create(element::i32, {}, {1}),
                                               op::v0::Constant::create(element::i32, {}, {0}));
    ctx.w_out   = std::make_shared<op::v8::Gather>(gshape, op::v0::Constant::create(element::i32, {}, {2}),
                                               op::v0::Constant::create(element::i32, {}, {0}));

    // Constants
    ctx.c0   = op::v0::Constant::create(ctx.calc_type, {}, {0.0f});
    ctx.c1   = op::v0::Constant::create(ctx.calc_type, {}, {1.0f});
    ctx.c2   = op::v0::Constant::create(ctx.calc_type, {}, {2.0f});
    ctx.c0_5 = op::v0::Constant::create(ctx.calc_type, {}, {0.5f});

    // Convert dimensions to float
    auto h_in_f = std::make_shared<op::v0::Convert>(ctx.h_in, ctx.calc_type);
    auto w_in_f = std::make_shared<op::v0::Convert>(ctx.w_in, ctx.calc_type);

    // Split grid into x/y
    auto grid_conv = grid;
    if (grid.get_element_type() != ctx.calc_type)
        grid_conv = std::make_shared<op::v0::Convert>(grid, ctx.calc_type);
    auto axis3 = op::v0::Constant::create(element::i32, {}, {3});
    auto x_grid = std::make_shared<op::v8::Gather>(grid_conv, op::v0::Constant::create(element::i32, {}, {0}), axis3);
    auto y_grid = std::make_shared<op::v8::Gather>(grid_conv, op::v0::Constant::create(element::i32, {}, {1}), axis3);

    // Save original normalized coordinates for ZEROS mask
    ctx.x_norm = x_grid;
    ctx.y_norm = y_grid;

    // Normalization [-1,1] -> pixels (without padding/reflection - that's the mode builder's job)
    if (gs->get_attributes().align_corners) {
        auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
        auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
        ctx.x = std::make_shared<op::v1::Multiply>(
                    std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(x_grid, ctx.c1), ctx.c2),
                    w_m1);
        ctx.y = std::make_shared<op::v1::Multiply>(
                    std::make_shared<op::v1::Divide>(std::make_shared<op::v1::Add>(y_grid, ctx.c1), ctx.c2),
                    h_m1);
    } else {
        // PyTorch formula for align_corners=false: ((x + 1) * W - 1) / 2
        auto x_p1_w = std::make_shared<op::v1::Multiply>(
                        std::make_shared<op::v1::Add>(x_grid, ctx.c1), w_in_f);
        auto y_p1_h = std::make_shared<op::v1::Multiply>(
                        std::make_shared<op::v1::Add>(y_grid, ctx.c1), h_in_f);
        ctx.x = std::make_shared<op::v1::Divide>(
                    std::make_shared<op::v1::Subtract>(x_p1_w, ctx.c1),
                    ctx.c2);
        ctx.y = std::make_shared<op::v1::Divide>(
                    std::make_shared<op::v1::Subtract>(y_p1_h, ctx.c1),
                    ctx.c2);
    }

    // NCHW -> NHWC
    auto to_nhwc = op::v0::Constant::create(element::i32, {4}, {0, 2, 3, 1});
    ctx.data_nhwc = std::make_shared<op::v1::Transpose>(data, to_nhwc);

    // Build NHWC result according to mode specifics
    auto result_nhwc = build_mode_result_nhwc(ctx, gs->get_attributes());
    if (!result_nhwc)
        return false;

    // NHWC -> NCHW
    auto to_nchw = op::v0::Constant::create(element::i32, {4}, {0, 3, 1, 2});
    auto result  = std::make_shared<op::v1::Transpose>(result_nhwc, to_nchw);

    // Finalization
    result->set_friendly_name(gs->get_friendly_name());
    ov::copy_runtime_info(gs, result);
    ov::replace_node_update_name(gs, result);
    return true;
}

} // unnamed namespace

// ========================= Composite pass =========================
GridSampleDecomposition::GridSampleDecomposition() {
    add_matcher<GridSampleDecompositionNearest>();
    add_matcher<GridSampleDecompositionBilinear>();
    add_matcher<GridSampleDecompositionBicubic>();
}

// ========================= NEAREST =========================
GridSampleDecompositionNearest::GridSampleDecompositionNearest() {
    auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>();
    matcher_pass_callback cb = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
        if (!gs || transformation_callback(gs))
            return false;
        if (gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::NEAREST)
            return false;

        return decompose_impl(gs, [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) -> std::shared_ptr<Node> {
            const bool is_zeros      = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
            const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

            auto h_in_f = std::make_shared<op::v0::Convert>(ctx.h_in, ctx.calc_type);
            auto w_in_f = std::make_shared<op::v0::Convert>(ctx.w_in, ctx.calc_type);

            // 1) Round to the nearest integer pixel using Round (banker's rounding like lrint)
            // This matches the reference implementation which uses std::lrint
            std::shared_ptr<Node> x_idx = std::make_shared<op::v5::Round>(ctx.x, op::v5::Round::RoundMode::HALF_TO_EVEN);
            std::shared_ptr<Node> y_idx = std::make_shared<op::v5::Round>(ctx.y, op::v5::Round::RoundMode::HALF_TO_EVEN);
            
            // 2) For REFLECTION: apply reflection to integer indices
            // The reference implementation applies reflection AFTER rounding to integers
            if (is_reflection) {
                x_idx = reflect_index(ctx, x_idx, w_in_f, attrs.align_corners);
                y_idx = reflect_index(ctx, y_idx, h_in_f, attrs.align_corners);
            }
            
            // 3) ALWAYS clamp indices to valid range [0, size-1] for safety
            // This is necessary because reflection can produce indices outside bounds 
            // (especially for align_corners=true with small sizes)
            auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
            auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
            
            // Store unclamped indices for ZEROS mask check (before reflection)
            auto x_idx_unclamped = std::make_shared<op::v5::Round>(ctx.x, op::v5::Round::RoundMode::HALF_TO_EVEN);
            auto y_idx_unclamped = std::make_shared<op::v5::Round>(ctx.y, op::v5::Round::RoundMode::HALF_TO_EVEN);
            
            // Always clamp for safe memory access
            x_idx = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(x_idx, ctx.c0), w_m1);
            y_idx = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(y_idx, ctx.c0), h_m1);

            std::shared_ptr<Node> result;
            
            if (is_zeros) {
                // Gather with clamped indices (safe memory access)
                result = gather_hw(ctx, x_idx, y_idx);
                
                // Build mask based on *unclamped rounded indices* for NEAREST mode (PyTorch behavior)
                // For NEAREST, a sample is inside if: 0 <= floor(x+0.5) <= W-1
                auto x_ge_0  = std::make_shared<op::v1::GreaterEqual>(x_idx_unclamped, ctx.c0);
                auto x_le_w1 = std::make_shared<op::v1::LessEqual>(x_idx_unclamped, w_m1);
                auto y_ge_0  = std::make_shared<op::v1::GreaterEqual>(y_idx_unclamped, ctx.c0);
                auto y_le_h1 = std::make_shared<op::v1::LessEqual>(y_idx_unclamped, h_m1);
                
                auto x_in_idx = std::make_shared<op::v1::LogicalAnd>(x_ge_0, x_le_w1);
                auto y_in_idx = std::make_shared<op::v1::LogicalAnd>(y_ge_0, y_le_h1);
                
                // No special handling for degenerate axes in NEAREST mode
                // The mask is based purely on whether rounded indices are in bounds
                auto ok = std::make_shared<op::v1::LogicalAnd>(x_in_idx, y_in_idx);

                auto mask   = std::make_shared<op::v0::Convert>(ok, ctx.element_type);
                auto mask_e = std::make_shared<op::v0::Unsqueeze>(mask, op::v0::Constant::create(element::i32, {}, {-1}));
                result = std::make_shared<op::v1::Multiply>(result, mask_e);
            } else {
                // For BORDER and REFLECTION, indices are already clamped
                result = gather_hw(ctx, x_idx, y_idx);
            }

            return result; // NHWC
        });
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionNearest"), cb);
}

// ========================= BILINEAR =========================
GridSampleDecompositionBilinear::GridSampleDecompositionBilinear() {
    auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>();
    matcher_pass_callback cb = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
        if (!gs || transformation_callback(gs))
            return false;
        if (gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BILINEAR)
            return false;

        return decompose_impl(gs, [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) -> std::shared_ptr<Node> {
            const bool is_border     = attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER;
            const bool is_zeros      = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
            const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

            auto h_in_f = std::make_shared<op::v0::Convert>(ctx.h_in, ctx.calc_type);
            auto w_in_f = std::make_shared<op::v0::Convert>(ctx.w_in, ctx.calc_type);

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
            auto one_m_dx = std::make_shared<op::v1::Subtract>(ctx.c1, dx);
            auto one_m_dy = std::make_shared<op::v1::Subtract>(ctx.c1, dy);

            std::vector<std::shared_ptr<Node>> weights{
                std::make_shared<op::v1::Multiply>(one_m_dx, one_m_dy), // w00
                std::make_shared<op::v1::Multiply>(dx,       one_m_dy), // w01
                std::make_shared<op::v1::Multiply>(one_m_dx, dy),       // w10
                std::make_shared<op::v1::Multiply>(dx,       dy)        // w11
            };

            if (is_reflection) {
                x0 = reflect_index(ctx, x0, w_in_f, attrs.align_corners);
                x1 = reflect_index(ctx, x1, w_in_f, attrs.align_corners);
                y0 = reflect_index(ctx, y0, h_in_f, attrs.align_corners);
                y1 = reflect_index(ctx, y1, h_in_f, attrs.align_corners);
            }

            std::vector<std::shared_ptr<Node>> x_coords{x0, x1, x0, x1};
            std::vector<std::shared_ptr<Node>> y_coords{y0, y0, y1, y1};
            std::vector<std::shared_ptr<Node>> values(4);

            for (size_t i = 0; i < 4; ++i) {
                auto xi = x_coords[i];
                auto yi = y_coords[i];

                if (is_zeros || is_border) {
                    auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
                    auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
                    xi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(xi, ctx.c0), w_m1);
                    yi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(yi, ctx.c0), h_m1);
                }

                auto val = gather_hw(ctx, xi, yi);

                if (is_zeros) {
                    auto x_ok = std::make_shared<op::v1::LogicalAnd>(
                        std::make_shared<op::v1::GreaterEqual>(x_coords[i], ctx.c0),
                        std::make_shared<op::v1::Less>(x_coords[i], w_in_f));
                    auto y_ok = std::make_shared<op::v1::LogicalAnd>(
                        std::make_shared<op::v1::GreaterEqual>(y_coords[i], ctx.c0),
                        std::make_shared<op::v1::Less>(y_coords[i], h_in_f));
                    auto ok   = std::make_shared<op::v1::LogicalAnd>(x_ok, y_ok);
                    auto mask = std::make_shared<op::v0::Convert>(ok, ctx.element_type);
                    auto mexp = std::make_shared<op::v0::Unsqueeze>(mask, op::v0::Constant::create(element::i32, {}, {-1}));
                    val = std::make_shared<op::v1::Multiply>(val, mexp);
                }

                values[i] = val;
            }

            std::shared_ptr<Node> res = std::make_shared<op::v1::Multiply>(
                values[0], std::make_shared<op::v0::Unsqueeze>(weights[0], op::v0::Constant::create(element::i32, {}, {-1})));
            for (size_t i = 1; i < 4; ++i) {
                auto wexp = std::make_shared<op::v0::Unsqueeze>(weights[i], op::v0::Constant::create(element::i32, {}, {-1}));
                res = std::make_shared<op::v1::Add>(res, std::make_shared<op::v1::Multiply>(values[i], wexp));
            }
            return res; // NHWC
        });
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBilinear"), cb);
}

// ========================= BICUBIC =========================
GridSampleDecompositionBicubic::GridSampleDecompositionBicubic() {
    auto pat = ov::pass::pattern::wrap_type<op::v9::GridSample>();
    matcher_pass_callback cb = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto gs = std::dynamic_pointer_cast<op::v9::GridSample>(m.get_match_root());
        if (!gs || transformation_callback(gs))
            return false;
        if (gs->get_attributes().mode != op::v9::GridSample::InterpolationMode::BICUBIC)
            return false;

        return decompose_impl(gs, [&](const Ctx& ctx, const op::v9::GridSample::Attributes& attrs) -> std::shared_ptr<Node> {
            const bool is_border     = attrs.padding_mode == op::v9::GridSample::PaddingMode::BORDER;
            const bool is_zeros      = attrs.padding_mode == op::v9::GridSample::PaddingMode::ZEROS;
            const bool is_reflection = attrs.padding_mode == op::v9::GridSample::PaddingMode::REFLECTION;

            auto h_in_f = std::make_shared<op::v0::Convert>(ctx.h_in, ctx.calc_type);
            auto w_in_f = std::make_shared<op::v0::Convert>(ctx.w_in, ctx.calc_type);

            // 1) padding/reflection on continuous coordinates (as in original)
            std::shared_ptr<Node> x_pad = ctx.x, y_pad = ctx.y;
            if (is_reflection) {
                if (attrs.align_corners) {
                    x_pad = reflect_coord(ctx, ctx.x, w_in_f, true);
                    y_pad = reflect_coord(ctx, ctx.y, h_in_f, true);
                }
            }
            // BORDER/ZEROS - don't touch continuous coordinates

            // 2) floor and fractional parts
            auto x0     = std::make_shared<op::v0::Floor>(x_pad);
            auto y0     = std::make_shared<op::v0::Floor>(y_pad);
            auto dx_raw = std::make_shared<op::v1::Subtract>(x_pad, x0);
            auto dy_raw = std::make_shared<op::v1::Subtract>(y_pad, y0);

            std::shared_ptr<Node> dx = dx_raw;
            std::shared_ptr<Node> dy = dy_raw;
            if (!is_zeros) {
                auto eps = op::v0::Constant::create(ctx.calc_type, {}, {1e-7f});
                auto one_m_eps = std::make_shared<op::v1::Subtract>(ctx.c1, eps);
                dx = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(dx_raw, ctx.c0), one_m_eps);
                dy = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(dy_raw, ctx.c0), one_m_eps);
            }

            // Account for axes of size 1
            auto w_is_one = std::make_shared<op::v1::Equal>(w_in_f, ctx.c1);
            auto h_is_one = std::make_shared<op::v1::Equal>(h_in_f, ctx.c1);
            dx = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, dx);
            dy = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, dy);

            // cubic weights (a = -0.75)
            auto a    = op::v0::Constant::create(ctx.calc_type, {}, {-0.75f});
            auto c3   = op::v0::Constant::create(ctx.calc_type, {}, {3.0f});
            auto c4   = op::v0::Constant::create(ctx.calc_type, {}, {4.0f});
            auto c5   = op::v0::Constant::create(ctx.calc_type, {}, {5.0f});
            auto c8   = op::v0::Constant::create(ctx.calc_type, {}, {8.0f});

            auto cubic_weight = [&](const std::shared_ptr<Node>& t) -> std::shared_ptr<Node> {
                auto t0  = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(t, ctx.c0), ctx.c2);
                auto t2  = std::make_shared<op::v1::Multiply>(t0, t0);
                auto t3  = std::make_shared<op::v1::Multiply>(t2, t0);
                auto ap2 = std::make_shared<op::v1::Add>(a, ctx.c2);
                auto ap3 = std::make_shared<op::v1::Add>(a, c3);

                auto f01_in = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(ap2, t0), ap3);
                auto f01    = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(f01_in, t2), ctx.c1);

                auto five_a  = std::make_shared<op::v1::Multiply>(c5, a);
                auto eight_a = std::make_shared<op::v1::Multiply>(c8, a);
                auto four_a  = std::make_shared<op::v1::Multiply>(c4, a);
                auto f12_in1 = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(a, t0), five_a);
                auto f12_in2 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(f12_in1, t0), eight_a);
                auto f12     = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(f12_in2, t0), four_a);

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

            // weight correction when W/H == 1
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

            // indices
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
                    x_ok[i] = std::make_shared<op::v1::LogicalAnd>(
                        std::make_shared<op::v1::GreaterEqual>(x_idx[i], ctx.c0),
                        std::make_shared<op::v1::Less>(x_idx[i], w_in_f));
                    y_ok[i] = std::make_shared<op::v1::LogicalAnd>(
                        std::make_shared<op::v1::GreaterEqual>(y_idx[i], ctx.c0),
                        std::make_shared<op::v1::Less>(y_idx[i], h_in_f));
                }
            }

            if (is_reflection) {
                for (int i = 0; i < 4; ++i) {
                    x_idx[i] = reflect_index(ctx, x_idx[i], w_in_f, attrs.align_corners);
                    y_idx[i] = reflect_index(ctx, y_idx[i], h_in_f, attrs.align_corners);
                }
            } else {
                // when size==1 - to 0
                for (int i = 0; i < 4; ++i) {
                    x_idx[i] = std::make_shared<op::v1::Select>(w_is_one, ctx.c0, x_idx[i]);
                    y_idx[i] = std::make_shared<op::v1::Select>(h_is_one, ctx.c0, y_idx[i]);
                }
            }

            std::shared_ptr<Node> sum = nullptr;
            for (int iy = 0; iy < 4; ++iy) {
                std::shared_ptr<Node> row = nullptr;
                for (int ix = 0; ix < 4; ++ix) {
                    auto xi = x_idx[ix];
                    auto yi = y_idx[iy];

                    if (is_zeros || is_border) {
                        auto w_m1 = std::make_shared<op::v1::Subtract>(w_in_f, ctx.c1);
                        auto h_m1 = std::make_shared<op::v1::Subtract>(h_in_f, ctx.c1);
                        xi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(xi, ctx.c0), w_m1);
                        yi = std::make_shared<op::v1::Minimum>(std::make_shared<op::v1::Maximum>(yi, ctx.c0), h_m1);
                    }

                    auto val = gather_hw(ctx, xi, yi);
                    auto wxy = std::make_shared<op::v1::Multiply>(wx[ix], wy[iy]);

                    if (is_zeros) {
                        auto ok = std::make_shared<op::v1::LogicalAnd>(x_ok[ix], y_ok[iy]);
                        auto m  = std::make_shared<op::v0::Convert>(ok, ctx.element_type);
                        wxy = std::make_shared<op::v1::Multiply>(wxy, m);
                    }

                    auto wexp = std::make_shared<op::v0::Unsqueeze>(wxy, op::v0::Constant::create(element::i32, {}, {-1}));
                    auto tap  = std::make_shared<op::v1::Multiply>(val, wexp);

                    row = row ? std::shared_ptr<Node>(std::make_shared<op::v1::Add>(row, tap)) : std::shared_ptr<Node>(tap);
                }
                sum = sum ? std::shared_ptr<Node>(std::make_shared<op::v1::Add>(sum, row)) : row;
            }
            return sum; // NHWC
        });
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pat, "GridSampleDecompositionBicubic"), cb);
}

} // namespace intel_cpu
} // namespace ov
