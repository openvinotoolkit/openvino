// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/gated_residual_fusion.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/primitives/gated_residual_caps.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"

using namespace testing;
using namespace ov::intel_gpu;
namespace caps = cldnn::gated_residual_caps;

namespace {

constexpr int64_t C = 4;
constexpr int64_t D = 64;
constexpr int64_t L = 16;
constexpr int64_t T = 8;
constexpr float EPS = 1e-6f;

std::shared_ptr<ov::op::v1::Reshape> reshape_to(const ov::Output<ov::Node>& src, const std::vector<int64_t>& shape) {
    auto target = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);
    return std::make_shared<ov::op::v1::Reshape>(src, target, false);
}

std::shared_ptr<ov::op::v1::Multiply> multiply(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b, bool swap_operands) {
    return swap_operands ? std::make_shared<ov::op::v1::Multiply>(b, a) : std::make_shared<ov::op::v1::Multiply>(a, b);
}

/// Builds the canonical GR READ sub-graph and returns the final `mixed` node.
std::shared_ptr<ov::Node> build_gr_read(const std::shared_ptr<ov::op::v0::Parameter>& residual,
                                        float low_scale,
                                        float mix_scale,
                                        int64_t branches,
                                        const ov::element::Type& et = ov::element::f16,
                                        bool with_gain = true,
                                        bool swap_operands = false,
                                        bool scalar_first_low = false,
                                        bool scalar_first_mix = false) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    // The gain is per (branch, channel): shape [C, D], i.e. C independent groups of width D.
    auto gamma = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(branches), static_cast<size_t>(D)}, std::vector<float>(branches * D, 1.0f));
    std::shared_ptr<ov::Node> rms = with_gain ? std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS) : std::make_shared<ov::op::internal::RMS>(r3, EPS);
    auto rbar = reshape_to(rms, {-1, branches * D});

    auto w_down = ov::op::v0::Constant::create(et, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = multiply(low, ov::op::v0::Constant::create(et, ov::Shape{}, {low_scale}), scalar_first_low);
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);
    auto weighted = multiply(gate, rbar, swap_operands);

    auto weighted3 = reshape_to(weighted, {-1, branches, D});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted3, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    return multiply(reduced, ov::op::v0::Constant::create(et, ov::Shape{}, {mix_scale}), scalar_first_mix);
}

/// Builds the GR READ sub-graph in the *decomposed* form emitted by checkpoints that
/// were exported before RMS-fusion ran: Power -> ReduceMean -> Add(eps) -> Sqrt ->
/// Divide -> Multiply for the norm, an explicit `gain + 1` fold, and a ReduceMean
/// (rather than ReduceSum + Multiply) for the branch mix.
std::shared_ptr<ov::Node> build_gr_read_decomposed(const std::shared_ptr<ov::op::v0::Parameter>& residual,
                                                   float low_scale,
                                                   int64_t branches,
                                                   bool with_gain = true,
                                                   bool swap_operands = false) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    auto sq = std::make_shared<ov::op::v1::Power>(r3, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {2.0f}));
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    auto biased = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {EPS}));
    auto rms = std::make_shared<ov::op::v0::Sqrt>(biased);
    auto inv_rms = std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f}), rms);
    auto normalized = multiply(r3, inv_rms, swap_operands);
    auto normalized_flat = reshape_to(normalized, {-1, branches * D});
    auto gain_raw = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(branches * D)}, std::vector<float>(branches * D, 0.0f));
    auto gain_plus_one = std::make_shared<ov::op::v1::Add>(gain_raw, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f}));
    auto rbar = with_gain ? std::static_pointer_cast<ov::Node>(multiply(normalized_flat, gain_plus_one, swap_operands))
                          : std::static_pointer_cast<ov::Node>(normalized_flat);

    auto w_down = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {low_scale}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);

    auto gate_split = reshape_to(gate, {-1, branches, D});
    auto rbar_split = reshape_to(rbar, {-1, branches, D});
    auto weighted = multiply(gate_split, rbar_split, swap_operands);
    return std::make_shared<ov::op::v1::ReduceMean>(weighted, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
}

size_t count_gated_residual(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::as_type_ptr<op::GatedResidual>(node)) {
            count++;
        }
    }
    return count;
}

using split_fn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&, const std::vector<int64_t>&)>;

/// [B, L, X] -> [B, L, tail...] with a target built from ShapeOf, as dynamic exporters emit it.
ov::Output<ov::Node> shape_of_split(const ov::Output<ov::Node>& src, const std::vector<int64_t>& tail) {
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64);
    auto lead = std::make_shared<ov::op::v8::Gather>(shape,
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1}),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto tail_c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{tail.size()}, tail);
    auto target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lead, tail_c}, 0);
    return std::make_shared<ov::op::v1::Reshape>(src, target, false);
}

/// [B, L, X] -> [B, L, tail...] via a special_zero Constant target.
ov::Output<ov::Node> zero_split(const ov::Output<ov::Node>& src, const std::vector<int64_t>& tail) {
    std::vector<int64_t> target = {0, 0};
    target.insert(target.end(), tail.begin(), tail.end());
    auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target.size()}, target);
    return std::make_shared<ov::op::v1::Reshape>(src, c, true);
}

/// GR READ over a rank-3 [B, L, C*D] residual; `split` produces the rank-4 branch views.
std::shared_ptr<ov::Node> build_gr_read_rank3(const std::shared_ptr<ov::op::v0::Parameter>& residual, const split_fn& split) {
    auto r4 = split(residual, {C, D});
    auto gamma = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{C, D}, std::vector<float>(C * D, 1.0f));
    auto rms = std::make_shared<ov::op::internal::RMS>(r4, gamma, EPS);
    auto rbar = zero_split(rms, {C * D});

    auto w_down = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{L, C * D}, std::vector<float>(L * C * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f / C}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);
    auto w_up = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{C * D, L}, std::vector<float>(C * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(gate, rbar);

    auto weighted4 = split(weighted, {C, D});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted4, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-2}), false);
    return std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f / C}));
}

constexpr int64_t LQ = 32;  // compressed weights need L % 32 == 0

/// NNCF-style decompression: u4 with `group` < K is Constant[N,G,group] -> Convert -> [Subtract(Convert(zp[N,G,1]))] ->
/// Multiply(scale[N,G,1]) -> Reshape[N,K]; per-channel (group == K) has no Reshape and [N,1] scale / zp. `with_zp` false is the
/// symmetric form without the Subtract; `scalar_zp` >= 0 makes the zero point a scalar of that value.
ov::Output<ov::Node>
make_decompressed(const ov::element::Type& wt, size_t N, size_t K, size_t group, bool scalar_scale = false, bool with_zp = true, int scalar_zp = -1) {
    const size_t G = K / group;
    const bool grouped = group != K;
    const ov::Shape wshape = grouped ? ov::Shape{N, G, group} : ov::Shape{N, K};
    const ov::Shape sshape = grouped ? ov::Shape{N, G, 1} : ov::Shape{N, 1};
    auto w = ov::op::v0::Constant::create(wt, wshape, std::vector<int>(ov::shape_size(wshape), 3));
    std::shared_ptr<ov::Node> dq = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    if (with_zp) {
        const ov::Shape zshape = scalar_zp >= 0 ? ov::Shape{} : sshape;
        auto zp = ov::op::v0::Constant::create(wt, zshape, std::vector<int>(ov::shape_size(zshape), scalar_zp >= 0 ? scalar_zp : 1));
        auto zpf = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
        dq = std::make_shared<ov::op::v1::Subtract>(dq, zpf);
    }
    const ov::Shape scshape = scalar_scale ? ov::Shape{} : sshape;
    auto sc = ov::op::v0::Constant::create(ov::element::f16, scshape, std::vector<float>(ov::shape_size(scshape), 0.01f));
    auto mul = std::make_shared<ov::op::v1::Multiply>(dq, sc);
    if (!grouped) {
        return mul;
    }
    return reshape_to(mul, {static_cast<int64_t>(N), static_cast<int64_t>(K)});
}

/// The canonical GR READ sub-graph with caller-provided W_down [LQ, C*D] / W_up [C*D, LQ] producers. `rbar_out` receives the
/// normalized residual so that tests can hang a write-gate branch off it.
std::shared_ptr<ov::Node> build_gr_read_with_weights(const std::shared_ptr<ov::op::v0::Parameter>& residual,
                                                     const ov::Output<ov::Node>& w_down,
                                                     const ov::Output<ov::Node>& w_up,
                                                     ov::Output<ov::Node>* rbar_out = nullptr) {
    auto r3 = reshape_to(residual, {-1, C, D});
    auto gamma = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(C), static_cast<size_t>(D)}, std::vector<float>(C * D, 1.0f));
    auto rms = std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS);
    auto rbar = reshape_to(rms, {-1, C * D});
    if (rbar_out != nullptr) {
        *rbar_out = rbar->output(0);
    }
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f / C}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(gate, rbar);
    auto weighted3 = reshape_to(weighted, {-1, C, D});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted3, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    return std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f / C}));
}

std::shared_ptr<op::GatedResidual> run_read_fusion(const ov::Output<ov::Node>& w_down, const ov::Output<ov::Node>& w_up, bool systolic) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read_with_weights(residual, w_down, w_up);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});
    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>(systolic);
    manager.run_passes(model);
    std::shared_ptr<op::GatedResidual> found;
    for (const auto& node : model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            found = gr;
        }
    }
    return found;
}

void expect_compressed(const std::shared_ptr<op::GatedResidual>& gr, const ov::element::Type& wt, size_t groups_down, size_t groups_up, bool has_zp = true) {
    ASSERT_NE(gr, nullptr);
    ASSERT_EQ(gr->get_input_size(), has_zp ? 8u : 6u);
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::W_DOWN), wt);
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::W_UP), wt);
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::W_DOWN), (ov::Shape{static_cast<size_t>(LQ), static_cast<size_t>(C * D)}));
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::W_UP), (ov::Shape{static_cast<size_t>(C * D), static_cast<size_t>(LQ)}));
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::SCALE_DOWN), ov::element::f16);
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::SCALE_DOWN), (ov::Shape{groups_down, static_cast<size_t>(LQ)}));
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::SCALE_UP), (ov::Shape{groups_up, static_cast<size_t>(C * D)}));
    if (has_zp) {
        EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::ZP_DOWN), wt);
        EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::ZP_UP), wt);
        EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::ZP_UP), (ov::Shape{groups_up, static_cast<size_t>(C * D)}));
    }
}

}  // namespace

TEST(gated_residual_fusion, gated_residual_read_is_fused) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        auto gr = ov::as_type_ptr<op::GatedResidual>(node);
        if (!gr) {
            continue;
        }
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::FINAL_MIX);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_lowrank(), static_cast<size_t>(L));
        EXPECT_FALSE(gr->get_produce_write_gate());
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    }
}

TEST(gated_residual_fusion, gated_residual_read_requires_both_inv_c_scalings) {
    // The 1/C inside SiLU(W_down * Rbar / C) is model semantics, not a foldable constant.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u) << "the generic graph must be preserved when 1/C is missing";
}

TEST(gated_residual_fusion, scale_behind_minimum_is_not_a_constant) {
    // min(1/C, x) equals 1/C only for some runtime x: the generic graph must be kept.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto limit = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C);
    mixed->input(1).replace_source_output(std::make_shared<ov::op::v1::Minimum>(mixed->input_value(1), limit));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual, limit});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_read_rejects_unsupported_branch_count) {
    constexpr int64_t branches = 2;
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, branches * D});
    auto mixed = build_gr_read(residual, 1.0f / branches, 1.0f / branches, branches);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u) << "only C == 4 is supported by the first version";
}

TEST(gated_residual_fusion, gated_residual_read_decomposed_form_is_fused) {
    // Mirrors the graph shape actually emitted by the model export:
    // decomposed RMSNorm (no fused ov::op::internal::RMS) and a ReduceMean-based
    // branch mix (no trailing 1/C Multiply).
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read_decomposed(residual, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        auto gr = ov::as_type_ptr<op::GatedResidual>(node);
        if (!gr) {
            continue;
        }
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::FINAL_MIX);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_lowrank(), static_cast<size_t>(L));
        EXPECT_FALSE(gr->get_produce_write_gate());
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    }
}

// The commutative Multiply / Add nodes of the patterns are matched in either operand order.
TEST(gated_residual_fusion, gated_residual_read_operand_order_is_irrelevant) {
    for (const bool decomposed : {false, true}) {
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
        auto mixed = decomposed ? build_gr_read_decomposed(residual, 1.0f / C, C, true, true)
                                : build_gr_read(residual, 1.0f / C, 1.0f / C, C, ov::element::f16, true, true);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

        ov::pass::Manager manager;
        manager.register_pass<GatedResidualReadFusion>();
        manager.run_passes(model);

        ASSERT_EQ(count_gated_residual(model), 1u) << (decomposed ? "decomposed" : "fused_rms");
    }
}

// The 1/C scalars are read by operand, not by position.
TEST(gated_residual_fusion, gated_residual_read_scalar_as_first_operand_of_the_down_scale) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C, ov::element::f16, true, false, true, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
}

TEST(gated_residual_fusion, gated_residual_read_scalar_as_first_operand_of_the_mix_scale) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C, ov::element::f16, true, false, false, true);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
}

class gated_residual_fusion_folded_gain : public ::testing::TestWithParam<bool> {};

// An exporter folds a gain of ones away, leaving the bare (fused or decomposed) normalization in front of the GEMM.
TEST_P(gated_residual_fusion_folded_gain, gated_residual_read_gets_a_unit_gain) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = GetParam() ? build_gr_read_decomposed(residual, 1.0f / C, C, false) : build_gr_read(residual, 1.0f / C, 1.0f / C, C, ov::element::f16, false);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        auto gr = ov::as_type_ptr<op::GatedResidual>(node);
        if (!gr) {
            continue;
        }
        auto gain = ov::as_type_ptr<ov::op::v0::Constant>(gr->get_input_node_shared_ptr(op::GatedResidual::ReadInputIdx::NORM_WEIGHT));
        ASSERT_NE(gain, nullptr);
        EXPECT_EQ(gain->get_shape(), (ov::Shape{static_cast<size_t>(C * D)}));
        EXPECT_EQ(gain->cast_vector<float>(), std::vector<float>(C * D, 1.0f));
        EXPECT_FALSE(gr->get_norm_gain_plus_one());
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, gated_residual_fusion_folded_gain, ::testing::Values(false, true), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? std::string("decomposed") : std::string("fused_rms");
});

TEST(gated_residual_fusion, gated_residual_write_is_fused) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C});

    auto s3 = reshape_to(s, {-1, C, 1});
    auto y3 = reshape_to(y, {-1, 1, D});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s3, y3);
    auto r3 = reshape_to(residual, {-1, C, D});
    auto added = std::make_shared<ov::op::v1::Add>(r3, scaled);
    auto out = reshape_to(added, {-1, C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        auto gr = ov::as_type_ptr<op::GatedResidual>(node);
        if (!gr) {
            continue;
        }
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::WRITE);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, C * D}));
    }
}

TEST(gated_residual_fusion, gated_residual_write_rejects_wrong_broadcast_axis) {
    // Broadcasting the gate along the channel axis instead of the branch axis is a
    // different operator and must not be fused. The graph itself stays valid so that
    // the matcher - not shape inference - is what rejects it.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});

    auto s3 = reshape_to(s, {-1, 1, D});  // channel-wise gate, not [T, C, 1]
    auto y3 = reshape_to(y, {-1, 1, D});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s3, y3);
    auto r3 = reshape_to(residual, {-1, C, D});
    auto added = std::make_shared<ov::op::v1::Add>(r3, scaled);
    auto out = reshape_to(added, {-1, C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_write_rejects_mismatched_rows) {
    // Y broadcasts over the rows ([1, D] against [T, C*D]), which the row-wise kernel does not model.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C});

    auto s3 = reshape_to(s, {-1, C, 1});
    auto y3 = reshape_to(y, {-1, 1, D});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s3, y3);
    auto r3 = reshape_to(residual, {-1, C, D});
    auto added = std::make_shared<ov::op::v1::Add>(r3, scaled);
    auto out = reshape_to(added, {-1, C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_write_op_rejects_mismatched_rows) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T + 1, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C});
    EXPECT_THROW(static_cast<void>(std::make_shared<op::GatedResidual>(ov::OutputVector{residual, y, s},
                                                                       op::GatedResidual::Mode::WRITE,
                                                                       static_cast<size_t>(C),
                                                                       static_cast<size_t>(D),
                                                                       0,
                                                                       EPS,
                                                                       false,
                                                                       false)),
                 ov::NodeValidationFailure);
}

TEST(gated_residual_fusion, gated_residual_write_rejects_static_one_against_dynamic_rows) {
    // [1, 1, D] against a dynamic [?, ?, C*D] residual is a broadcast the row-wise kernel would read out of bounds.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C});

    auto s4 = std::make_shared<ov::op::v0::Unsqueeze>(s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
    auto y4 = std::make_shared<ov::op::v0::Unsqueeze>(y, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s4, y4);
    auto added = std::make_shared<ov::op::v1::Add>(zero_split(residual, {C, D}), scaled);
    auto out = zero_split(added, {C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_write_op_rejects_static_one_against_dynamic_rows) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{1, 1, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C});
    EXPECT_THROW(static_cast<void>(std::make_shared<op::GatedResidual>(ov::OutputVector{residual, y, s},
                                                                       op::GatedResidual::Mode::WRITE,
                                                                       static_cast<size_t>(C),
                                                                       static_cast<size_t>(D),
                                                                       0,
                                                                       EPS,
                                                                       false,
                                                                       false)),
                 ov::NodeValidationFailure);
}

TEST(gated_residual_fusion, gated_residual_write_decomposed_form_is_fused) {
    // Mirrors the real export: the gate/y are broadcast to 3D via Unsqueeze (not
    // Reshape), and the Add happens in the flat [*, C*D] domain against an
    // already-flat residual, with the split-form reshape appearing *after* the Add.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C});

    auto gate_3d = std::make_shared<ov::op::v0::Unsqueeze>(s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto y_3d = std::make_shared<ov::op::v0::Unsqueeze>(y, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto scaled = std::make_shared<ov::op::v1::Multiply>(y_3d, gate_3d);
    auto scaled_flat = reshape_to(scaled, {-1, C * D});
    auto updated = std::make_shared<ov::op::v1::Add>(residual, scaled_flat);
    // A leftover reshape back to split form, left untouched by the fusion.
    auto out = reshape_to(updated, {-1, C, D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        auto gr = ov::as_type_ptr<op::GatedResidual>(node);
        if (!gr) {
            continue;
        }
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::WRITE);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, C * D}));
    }
}

TEST(gated_residual_fusion, gated_residual_write_operand_order_is_irrelevant) {
    for (const bool flat_add : {false, true}) {
        auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
        auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});
        auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C});

        auto scaled = std::make_shared<ov::op::v1::Multiply>(reshape_to(y, {-1, 1, D}), reshape_to(s, {-1, C, 1}));
        // Case A adds in the split domain, case B against an already-flat residual; the scaled term comes first in both.
        auto out = flat_add ? ov::Output<ov::Node>(std::make_shared<ov::op::v1::Add>(reshape_to(scaled, {-1, C * D}), residual))
                            : ov::Output<ov::Node>(reshape_to(std::make_shared<ov::op::v1::Add>(scaled, reshape_to(residual, {-1, C, D})), {-1, C * D}));
        auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

        ov::pass::Manager manager;
        manager.register_pass<GatedResidualWriteFusion>();
        manager.run_passes(model);

        ASSERT_EQ(count_gated_residual(model), 1u) << (flat_add ? "flat add" : "split add");
    }
}

class gated_residual_fusion_dynamic_split : public ::testing::TestWithParam<bool> {};

TEST_P(gated_residual_fusion_dynamic_split, gated_residual_read_rank3_dynamic_is_fused) {
    const split_fn split = GetParam() ? split_fn(shape_of_split) : split_fn(zero_split);
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C * D});
    auto mixed = build_gr_read_rank3(residual, split);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
            EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
            EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{-1, -1, D}));
        }
    }
}

TEST_P(gated_residual_fusion_dynamic_split, gated_residual_write_rank3_dynamic_is_fused) {
    const split_fn split = GetParam() ? split_fn(shape_of_split) : split_fn(zero_split);
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, C});

    auto s4 = std::make_shared<ov::op::v0::Unsqueeze>(s, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
    auto y4 = std::make_shared<ov::op::v0::Unsqueeze>(y, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s4, y4);
    auto r4 = split(residual, {C, D});
    auto added = std::make_shared<ov::op::v1::Add>(r4, scaled);
    auto out = zero_split(added, {C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 1u);
    for (const auto& node : model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::WRITE);
            EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{-1, -1, C * D}));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, gated_residual_fusion_dynamic_split, ::testing::Values(false, true), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? std::string("shape_of_target") : std::string("constant_target");
});

TEST(gated_residual_fusion, gated_residual_read_rejects_rank_changing_split) {
    // [B, L, C*D] -> [-1, C, D] flattens B and L; the fused op would keep them.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_read_rejects_f32) {
    // The kernels are f16-only; an f32 pattern must stay decomposed rather than be
    // fused into a node with no available GPU implementation.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C, ov::element::f32);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, gated_residual_write_rejects_f32) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C});

    auto s3 = reshape_to(s, {-1, C, 1});
    auto y3 = reshape_to(y, {-1, 1, D});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(s3, y3);
    auto r3 = reshape_to(residual, {-1, C, D});
    auto added = std::make_shared<ov::op::v1::Add>(r3, scaled);
    auto out = reshape_to(added, {-1, C * D});

    auto model = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{residual, y, s});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualWriteFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(gated_residual_fusion, compressed_u4_group32_weights_are_fused) {
    const size_t cd = C * D;
    auto gr = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), true);
    expect_compressed(gr, ov::element::u4, cd / 32, LQ / 32);
}

TEST(gated_residual_fusion, compressed_u8_per_channel_weights_are_fused) {
    const size_t cd = C * D;
    auto gr = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    expect_compressed(gr, ov::element::u8, 1, 1);
}

TEST(gated_residual_fusion, compressed_signed_weights_are_fused) {
    const size_t cd = C * D;
    expect_compressed(run_read_fusion(make_decompressed(ov::element::i4, LQ, cd, 32), make_decompressed(ov::element::i4, cd, LQ, 32), true),
                      ov::element::i4,
                      cd / 32,
                      LQ / 32);
    expect_compressed(run_read_fusion(make_decompressed(ov::element::i8, LQ, cd, cd), make_decompressed(ov::element::i8, cd, LQ, LQ), true),
                      ov::element::i8,
                      1,
                      1);
}

TEST(gated_residual_fusion, compressed_weights_without_zero_points_are_fused_with_six_inputs) {
    const size_t cd = C * D;
    for (const auto& wt : {ov::element::u4, ov::element::i4}) {
        expect_compressed(run_read_fusion(make_decompressed(wt, LQ, cd, 32, false, false), make_decompressed(wt, cd, LQ, 32, false, false), true),
                          wt,
                          cd / 32,
                          LQ / 32,
                          false);
    }
    for (const auto& wt : {ov::element::u8, ov::element::i8}) {
        expect_compressed(run_read_fusion(make_decompressed(wt, LQ, cd, cd, false, false), make_decompressed(wt, cd, LQ, LQ, false, false), true),
                          wt,
                          1,
                          1,
                          false);
    }
}

TEST(gated_residual_fusion, compressed_weights_must_agree_on_type_and_zero_points) {
    const size_t cd = C * D;
    // W_down has zero points, W_up has none
    auto mixed_zp = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32, false, false), true);
    ASSERT_NE(mixed_zp, nullptr);
    EXPECT_EQ(mixed_zp->get_input_size(), 4u);
    // signed and unsigned 4-bit weights are different element types
    auto mixed_sign = run_read_fusion(make_decompressed(ov::element::i4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), true);
    ASSERT_NE(mixed_sign, nullptr);
    EXPECT_EQ(mixed_sign->get_input_size(), 4u);
}

TEST(gated_residual_fusion, scalar_symmetric_zero_point_is_the_signed_form_without_zero_points) {
    const size_t cd = C * D;
    // unsigned weights with the scalar zero point 2^(bits - 1) (older NNCF symmetric) become i4 / i8 without zero points, q - zp unchanged
    auto u4 =
        run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32, false, true, 8), make_decompressed(ov::element::u4, cd, LQ, 32, false, true, 8), true);
    expect_compressed(u4, ov::element::i4, cd / 32, LQ / 32, false);
    auto u4_w = ov::as_type_ptr<ov::op::v0::Constant>(u4->get_input_node_shared_ptr(op::GatedResidual::ReadInputIdx::W_DOWN));
    ASSERT_NE(u4_w, nullptr);
    for (const auto v : u4_w->cast_vector<int32_t>()) {
        ASSERT_EQ(v, 3 - 8);
    }
    auto u8 = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd, false, true, 128),
                              make_decompressed(ov::element::u8, cd, LQ, LQ, false, true, 128),
                              true);
    expect_compressed(u8, ov::element::i8, 1, 1, false);
    auto u8_w = ov::as_type_ptr<ov::op::v0::Constant>(u8->get_input_node_shared_ptr(op::GatedResidual::ReadInputIdx::W_UP));
    ASSERT_NE(u8_w, nullptr);
    for (const auto v : u8_w->cast_vector<int32_t>()) {
        ASSERT_EQ(v, 3 - 128);
    }
    // a scalar zero point of 0 is the symmetric form of any weight type and keeps it
    expect_compressed(
        run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32, false, true, 0), make_decompressed(ov::element::u4, cd, LQ, 32, false, true, 0), true),
        ov::element::u4,
        cd / 32,
        LQ / 32,
        false);
    // any other scalar zero point is not supported
    auto other =
        run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32, false, true, 5), make_decompressed(ov::element::u4, cd, LQ, 32, false, true, 5), true);
    ASSERT_NE(other, nullptr);
    EXPECT_EQ(other->get_input_size(), 4u);
    // a scalar 2^(bits - 1) on signed weights is a real zero point, not the symmetric form
    auto signed_zp =
        run_read_fusion(make_decompressed(ov::element::i4, LQ, cd, 32, false, true, 4), make_decompressed(ov::element::i4, cd, LQ, 32, false, true, 4), true);
    ASSERT_NE(signed_zp, nullptr);
    EXPECT_EQ(signed_zp->get_input_size(), 4u);
}

TEST(gated_residual_fusion, compressed_scale_is_transposed_to_group_major) {
    const size_t cd = C * D;
    const size_t g = cd / 32;
    // scale[n, g] = n * 10 + g, so [G, N] holds element (g, n) at g * N + n
    std::vector<float> scale_ng(LQ * g);
    for (size_t n = 0; n < static_cast<size_t>(LQ); n++) {
        for (size_t i = 0; i < g; i++) {
            scale_ng[n * g + i] = static_cast<float>(n * 10 + i);
        }
    }
    auto w = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{static_cast<size_t>(LQ), g, 32}, std::vector<int>(LQ * cd, 3));
    auto wf = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    auto zp = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{static_cast<size_t>(LQ), g, 1}, std::vector<int>(LQ * g, 1));
    auto sub = std::make_shared<ov::op::v1::Subtract>(wf, std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16));
    auto sc = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(LQ), g, 1}, scale_ng);
    auto w_down = reshape_to(std::make_shared<ov::op::v1::Multiply>(sub, sc), {LQ, static_cast<int64_t>(cd)});
    auto gr = run_read_fusion(w_down, make_decompressed(ov::element::u4, cd, LQ, 32), true);
    expect_compressed(gr, ov::element::u4, g, LQ / 32);
    auto scale = ov::as_type_ptr<ov::op::v0::Constant>(gr->get_input_node_shared_ptr(op::GatedResidual::ReadInputIdx::SCALE_DOWN));
    ASSERT_NE(scale, nullptr);
    const auto v = scale->cast_vector<float>();
    for (size_t i = 0; i < g; i++) {
        for (size_t n = 0; n < static_cast<size_t>(LQ); n++) {
            ASSERT_EQ(v[i * LQ + n], scale_ng[n * g + i]);
        }
    }
}

TEST(gated_residual_geometry, systolic_kernels_lift_the_scalar_limits) {
    using Mode = op::GatedResidual::Mode;
    // D = 5120 (C*D = 20480), L = 640
    EXPECT_TRUE(caps::is_geometry_supported(Mode::READ, 4, 5120, 640, true));
    EXPECT_FALSE(caps::is_geometry_supported(Mode::READ, 4, 5120, 640, false));
    EXPECT_FALSE(caps::is_geometry_supported(Mode::READ, 4, 2560, 640, false)) << "L is kept in registers by the scalar kernel";
    EXPECT_FALSE(caps::is_geometry_supported(Mode::READ, 4, 5120, 512, false)) << "C*D is kept in SLM by the scalar kernel";
    EXPECT_TRUE(caps::is_geometry_supported(Mode::READ, 4, 64, 16, true)) << "the scalar kernel is still there";
    EXPECT_FALSE(caps::is_geometry_supported(Mode::READ, 4, 64, 8, true));
    EXPECT_TRUE(caps::is_geometry_supported(Mode::WRITE, 4, 5120, 0, false)) << "no SLM in WRITE";
    EXPECT_FALSE(caps::is_geometry_supported(Mode::WRITE, 3, 64, 0, false));
}

TEST(gated_residual_fusion, compressed_weights_are_fused_without_systolic_arrays) {
    const size_t cd = C * D;
    // the scalar gr_read.cl dequantizes the compressed weights too
    expect_compressed(run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), false),
                      ov::element::u4,
                      cd / 32,
                      LQ / 32);
    expect_compressed(run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd), make_decompressed(ov::element::u8, cd, LQ, LQ), false),
                      ov::element::u8,
                      1,
                      1);
}

TEST(gated_residual_fusion, compressed_rejects_unsupported_quantization) {
    const size_t cd = C * D;
    // W_down and W_up must be compressed the same way
    auto mixed = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    ASSERT_NE(mixed, nullptr);
    EXPECT_EQ(mixed->get_input_size(), 4u);
    // scalar scale is not a per-(group, channel) tensor
    auto scalar = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd, true), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    ASSERT_NE(scalar, nullptr);
    EXPECT_EQ(scalar->get_input_size(), 4u);
}

TEST(gated_residual_fusion, compressed_group_sizes_are_multiples_of_32) {
    const size_t cd = C * D;
    // W_down [LQ, cd] takes 32 / 64 / 128 / per-channel (K = 256); W_up [cd, LQ] has K = 32, i.e. one group of 32
    for (const auto& wt : {ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8}) {
        for (const size_t group : {size_t{32}, size_t{64}, size_t{128}, cd}) {
            auto gr = run_read_fusion(make_decompressed(wt, LQ, cd, group), make_decompressed(wt, cd, LQ, 32), true);
            expect_compressed(gr, wt, cd / group, 1);
        }
    }
}

TEST(gated_residual_fusion, compressed_group_size_is_derived_from_the_shapes) {
    const size_t cd = C * D;
    // group size 16 is not a multiple of 32
    auto g16 = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 16), make_decompressed(ov::element::u4, cd, LQ, 16), true);
    ASSERT_NE(g16, nullptr);
    EXPECT_EQ(g16->get_input_size(), 4u);
    auto u8_g16 = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, 16), make_decompressed(ov::element::u8, cd, LQ, 16), true);
    ASSERT_NE(u8_g16, nullptr);
    EXPECT_EQ(u8_g16->get_input_size(), 4u);
}

namespace {

/// A compressed READ op whose scale / zp group counts are chosen by the caller (the group size is K / G); 8 inputs, 6 without
/// zero points.
std::shared_ptr<op::GatedResidual> make_compressed_op(const ov::element::Type& wt, size_t g_down, size_t g_up, bool has_zp = true) {
    const size_t cd = C * D;
    const auto lq = static_cast<size_t>(LQ);
    auto param = [](const ov::element::Type& et, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(et, shape);
    };
    ov::OutputVector inputs{param(ov::element::f16, {static_cast<size_t>(T), cd}),
                            param(ov::element::f16, {cd}),
                            param(wt, {lq, cd}),
                            param(wt, {cd, lq}),
                            param(ov::element::f16, {g_down, lq}),
                            param(ov::element::f16, {g_up, cd})};
    if (has_zp) {
        inputs.push_back(param(wt, {g_down, lq}));
        inputs.push_back(param(wt, {g_up, cd}));
    }
    return std::make_shared<op::GatedResidual>(inputs, op::GatedResidual::Mode::READ, static_cast<size_t>(C), static_cast<size_t>(D), lq, EPS, false, false);
}

}  // namespace

TEST(gated_residual_read_layout, input_forms) {
    using Op = op::GatedResidual;
    // without a write gate: 4 floating-point, 6 compressed, 8 compressed with zero points
    EXPECT_FALSE(Op::read_layout(4, false)->compressed);
    EXPECT_TRUE(Op::read_layout(6, false)->compressed);
    EXPECT_FALSE(Op::read_layout(6, false)->has_zp);
    EXPECT_TRUE(Op::read_layout(8, false)->has_zp);
    EXPECT_FALSE(Op::read_layout(5, false).has_value());
    EXPECT_FALSE(Op::read_layout(7, false).has_value());
    // with a write gate: 5 floating-point, 8 compressed, 11 compressed with zero points; W_inject is always last
    const auto f16 = Op::read_layout(5, true);
    ASSERT_TRUE(f16.has_value());
    EXPECT_FALSE(f16->compressed);
    EXPECT_EQ(f16->w_inject, 4u);
    const auto sym = Op::read_layout(8, true);
    ASSERT_TRUE(sym.has_value());
    EXPECT_TRUE(sym->compressed);
    EXPECT_FALSE(sym->has_zp);
    EXPECT_EQ(sym->w_inject, 6u);
    EXPECT_EQ(sym->scale_inject, 7u);
    const auto asym = Op::read_layout(11, true);
    ASSERT_TRUE(asym.has_value());
    EXPECT_TRUE(asym->has_zp);
    EXPECT_EQ(asym->w_inject, 8u);
    EXPECT_EQ(asym->scale_inject, 9u);
    EXPECT_EQ(asym->zp_inject, 10u);
    EXPECT_FALSE(Op::read_layout(6, true).has_value());
    EXPECT_FALSE(Op::read_layout(4, true).has_value());
}

TEST(gated_residual_group_size, op_validates_the_zero_point_type) {
    const size_t cd = C * D;
    const size_t lq = static_cast<size_t>(LQ);
    auto param = [](const ov::element::Type& et, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(et, shape);
    };
    ov::OutputVector inputs{param(ov::element::f16, {static_cast<size_t>(T), cd}),
                            param(ov::element::f16, {cd}),
                            param(ov::element::i4, {lq, cd}),
                            param(ov::element::i4, {cd, lq}),
                            param(ov::element::f16, {cd / 32, lq}),
                            param(ov::element::f16, {lq / 32, cd}),
                            param(ov::element::u4, {cd / 32, lq}),
                            param(ov::element::u4, {lq / 32, cd})};
    EXPECT_THROW(static_cast<void>(std::make_shared<op::GatedResidual>(inputs, op::GatedResidual::Mode::READ, C, D, LQ, EPS, false, false)),
                 ov::NodeValidationFailure)
        << "the zero point must have the weight element type";
}

TEST(gated_residual_group_size, derive_group_size) {
    EXPECT_EQ(caps::derive_group_size(256, 8), 32u);
    EXPECT_EQ(caps::derive_group_size(256, 1), 256u);
    EXPECT_EQ(caps::derive_group_size(256, 0), 0u) << "no scale rows";
    EXPECT_EQ(caps::derive_group_size(256, 7), 0u) << "K is not divisible by G";
}

TEST(gated_residual_group_size, supported_group_sizes) {
    EXPECT_TRUE(caps::is_group_size_supported(256, 32));
    EXPECT_TRUE(caps::is_group_size_supported(256, 64));
    EXPECT_TRUE(caps::is_group_size_supported(256, 128));
    EXPECT_TRUE(caps::is_group_size_supported(256, 256)) << "per-channel";
    EXPECT_TRUE(caps::is_group_size_supported(320, 160)) << "a multiple of 32 that divides K";
    EXPECT_FALSE(caps::is_group_size_supported(256, 16));
    EXPECT_FALSE(caps::is_group_size_supported(256, 48)) << "not a multiple of 32";
    EXPECT_FALSE(caps::is_group_size_supported(256, 96)) << "does not divide K";
    EXPECT_FALSE(caps::is_group_size_supported(320, 128)) << "does not divide K";
    EXPECT_FALSE(caps::is_group_size_supported(256, 512)) << "larger than K";
    EXPECT_FALSE(caps::is_group_size_supported(256, 0));
}

TEST(gated_residual_group_size, matrix_compression_support_uses_k_over_g) {
    // W_down [32, 256], W_up [256, 32]
    for (const size_t bits : {size_t{4}, size_t{8}}) {
        for (const size_t g_down : {size_t{8}, size_t{4}, size_t{2}, size_t{1}}) {
            EXPECT_TRUE(caps::is_matrix_compression_supported(bits, 32, 256, g_down)) << bits << " bits, " << g_down << " groups";
        }
        EXPECT_TRUE(caps::is_matrix_compression_supported(bits, 256, 32, 1));
        EXPECT_FALSE(caps::is_matrix_compression_supported(bits, 32, 256, 16)) << "W_down group size 16";
        EXPECT_FALSE(caps::is_matrix_compression_supported(bits, 256, 32, 2)) << "W_up group size 16";
        EXPECT_FALSE(caps::is_matrix_compression_supported(bits, 32, 256, 7)) << "K not divisible by G";
        EXPECT_FALSE(caps::is_matrix_compression_supported(bits, 32, 256, 0));
        EXPECT_FALSE(caps::is_matrix_compression_supported(bits, 4, 256, 1)) << "N % 16";
    }
}

TEST(gated_residual_group_size, row_compression_support_has_no_row_count_rule) {
    // W_inject [C = 4, K = 256]: N is not a multiple of 16, the group rules are those of W_down / W_up
    for (const size_t bits : {size_t{4}, size_t{8}}) {
        for (const size_t g : {size_t{8}, size_t{4}, size_t{2}, size_t{1}}) {
            EXPECT_TRUE(caps::is_row_compression_supported(bits, 256, g)) << bits << " bits, " << g << " groups";
        }
        EXPECT_FALSE(caps::is_row_compression_supported(bits, 256, 16)) << "group size 16";
        EXPECT_FALSE(caps::is_row_compression_supported(bits, 256, 7)) << "K not divisible by G";
        EXPECT_FALSE(caps::is_row_compression_supported(bits, 48, 1)) << "K % 32";
    }
    EXPECT_FALSE(caps::is_row_compression_supported(2, 256, 8)) << "bits";
}

TEST(gated_residual_group_size, op_accepts_supported_group_sizes) {
    const size_t cd = C * D;
    const size_t lq = static_cast<size_t>(LQ);
    for (const bool has_zp : {true, false}) {
        for (const auto& wt : {ov::element::u4, ov::element::i4, ov::element::u8, ov::element::i8}) {
            for (const size_t group : {size_t{32}, size_t{64}, size_t{128}, cd}) {
                // W_up keeps its K32 groups for the smallest W_down group size, per-channel otherwise
                const size_t g_up = group == 32 ? lq / 32 : 1;
                std::shared_ptr<op::GatedResidual> gr;
                ASSERT_NO_THROW(gr = make_compressed_op(wt, cd / group, g_up, has_zp)) << wt << " group " << group;
                EXPECT_EQ(gr->get_input_size(), has_zp ? 8u : 6u);
            }
        }
    }
}

TEST(gated_residual_group_size, op_rejects_only_indivisible_group_counts) {
    const size_t cd = C * D;
    const size_t lq = static_cast<size_t>(LQ);
    for (const auto& wt : {ov::element::u4, ov::element::u8}) {
        EXPECT_THROW(make_compressed_op(wt, 7, 1), ov::NodeValidationFailure) << "K not divisible by G";
        EXPECT_THROW(make_compressed_op(wt, cd / 32, 3), ov::NodeValidationFailure) << "K not divisible by G";
        // what the kernels support is the impl's and the fusion's business, not the op's
        EXPECT_NO_THROW(make_compressed_op(wt, cd / 16, 1)) << "W_down group size 16";
        EXPECT_NO_THROW(make_compressed_op(wt, cd / 32, lq / 16)) << "W_up group size 16";
    }
}

namespace {

/// A compressed READ op with a compressed W_inject (11 inputs, 8 without zero points); g_inject is the group count of W_inject's
/// scale / zp.
std::shared_ptr<op::GatedResidual> make_compressed_gate_op(const ov::element::Type& wt,
                                                           size_t g_down,
                                                           size_t g_up,
                                                           size_t g_inject,
                                                           const ov::element::Type& inject_wt,
                                                           const ov::Shape& scale_inject_tail,
                                                           bool has_zp = true) {
    const size_t cd = C * D;
    const auto lq = static_cast<size_t>(LQ);
    const auto c = static_cast<size_t>(C);
    auto param = [](const ov::element::Type& et, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(et, shape);
    };
    ov::Shape scale_inject{g_inject};
    scale_inject.insert(scale_inject.end(), scale_inject_tail.begin(), scale_inject_tail.end());
    ov::OutputVector inputs{param(ov::element::f16, {static_cast<size_t>(T), cd}),
                            param(ov::element::f16, {cd}),
                            param(wt, {lq, cd}),
                            param(wt, {cd, lq}),
                            param(ov::element::f16, {g_down, lq}),
                            param(ov::element::f16, {g_up, cd})};
    if (has_zp) {
        inputs.push_back(param(wt, {g_down, lq}));
        inputs.push_back(param(wt, {g_up, cd}));
    }
    inputs.push_back(param(inject_wt, {c, cd}));
    inputs.push_back(param(ov::element::f16, scale_inject));
    if (has_zp) {
        inputs.push_back(param(inject_wt, scale_inject));
    }
    return std::make_shared<op::GatedResidual>(inputs, op::GatedResidual::Mode::READ, c, static_cast<size_t>(D), lq, EPS, false, true);
}

}  // namespace

TEST(gated_residual_write_gate_op, compressed_write_gate_has_eleven_inputs) {
    const size_t cd = C * D;
    const size_t lq = static_cast<size_t>(LQ);
    const auto c = static_cast<size_t>(C);
    std::shared_ptr<op::GatedResidual> u4;
    ASSERT_NO_THROW(u4 = make_compressed_gate_op(ov::element::u4, cd / 32, lq / 32, cd / 32, ov::element::u4, {c}));
    EXPECT_EQ(u4->get_input_size(), 11u);
    ASSERT_EQ(u4->get_output_size(), 2u);
    EXPECT_EQ(u4->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    EXPECT_EQ(u4->get_output_partial_shape(1), (ov::PartialShape{T, C}));
    EXPECT_EQ(u4->get_output_element_type(0), ov::element::f16);
    EXPECT_EQ(u4->get_output_element_type(1), ov::element::f16);
    EXPECT_NO_THROW(make_compressed_gate_op(ov::element::u8, 1, 1, 1, ov::element::u8, {c}));
    // signed weights, and no zero points at all (8 inputs: W_inject, scale_inject right after the scales)
    EXPECT_NO_THROW(make_compressed_gate_op(ov::element::i4, cd / 32, lq / 32, cd / 32, ov::element::i4, {c}));
    EXPECT_NO_THROW(make_compressed_gate_op(ov::element::i8, 1, 1, 1, ov::element::i8, {c}));
    auto sym = make_compressed_gate_op(ov::element::u4, cd / 32, lq / 32, cd / 32, ov::element::u4, {c}, false);
    EXPECT_EQ(sym->get_input_size(), 8u);
    ASSERT_EQ(sym->get_output_size(), 2u);
    EXPECT_EQ(sym->get_output_partial_shape(1), (ov::PartialShape{T, C}));
}

TEST(gated_residual_write_gate_op, compressed_write_gate_rejects_inconsistent_inputs) {
    const size_t cd = C * D;
    const size_t lq = static_cast<size_t>(LQ);
    const auto c = static_cast<size_t>(C);
    EXPECT_THROW(make_compressed_gate_op(ov::element::u4, cd / 32, lq / 32, cd / 32, ov::element::u8, {c}), ov::NodeValidationFailure)
        << "W_inject must have the element type of W_down / W_up";
    EXPECT_THROW(make_compressed_gate_op(ov::element::u4, cd / 32, lq / 32, cd / 32, ov::element::u4, {c + 1}), ov::NodeValidationFailure)
        << "scale_inject must have C columns";
    EXPECT_THROW(make_compressed_gate_op(ov::element::u4, cd / 32, lq / 32, 7, ov::element::u4, {c}), ov::NodeValidationFailure)
        << "W_inject K not divisible by G";
}

TEST(gated_residual_write_gate_op, f16_write_gate_keeps_five_inputs) {
    const size_t cd = C * D;
    auto param = [](const ov::element::Type& et, const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(et, shape);
    };
    ov::OutputVector inputs{param(ov::element::f16, {static_cast<size_t>(T), cd}),
                            param(ov::element::f16, {cd}),
                            param(ov::element::f16, {static_cast<size_t>(LQ), cd}),
                            param(ov::element::f16, {cd, static_cast<size_t>(LQ)}),
                            param(ov::element::f16, {static_cast<size_t>(C), cd})};
    auto gr = std::make_shared<op::GatedResidual>(inputs, op::GatedResidual::Mode::READ, C, D, LQ, EPS, false, true);
    EXPECT_EQ(gr->get_output_size(), 2u);
    EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    // a compressed-weight W_inject needs the full 11-input form
    inputs.back() = param(ov::element::u4, {static_cast<size_t>(C), cd});
    EXPECT_THROW(static_cast<void>(std::make_shared<op::GatedResidual>(inputs, op::GatedResidual::Mode::READ, C, D, LQ, EPS, false, true)),
                 ov::NodeValidationFailure);
}

namespace {

enum class GateScale { multiply, divide, folded };

/// S = 2 * sigmoid(MatMul(Rbar, W_inject, transpose_b) / C) with the 1/C written as a Multiply, a Divide or folded into W_inject.
/// `scale` is the constant as written (1/C for multiply / folded, C for divide).
ov::Output<ov::Node> build_write_gate(const ov::Output<ov::Node>& rbar,
                                      const ov::Output<ov::Node>& w_inject,
                                      GateScale form = GateScale::multiply,
                                      float scale = 1.0f / C,
                                      float times = 2.0f) {
    auto f16_const = [](float v) {
        return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {v});
    };
    std::shared_ptr<ov::Node> x;
    switch (form) {
    case GateScale::multiply:
        x = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::MatMul>(rbar, w_inject, false, true), f16_const(scale));
        break;
    case GateScale::divide:
        x = std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v0::MatMul>(rbar, w_inject, false, true), f16_const(scale));
        break;
    case GateScale::folded:
        x = std::make_shared<ov::op::v0::MatMul>(rbar, std::make_shared<ov::op::v1::Multiply>(w_inject, f16_const(scale)), false, true);
        break;
    }
    return std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Sigmoid>(x), f16_const(times))->output(0);
}

ov::Output<ov::Node> f16_inject_weights() {
    return ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(C), static_cast<size_t>(C * D)}, std::vector<float>(C * C * D, 0.02f));
}

using gate_fn = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>& residual, const ov::Output<ov::Node>& rbar)>;

struct GateFusion {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::GatedResidual> gr;
};

/// READ sub-graph plus the write-gate branch built by `make_gate`; the model outputs are {mixed, S, extra...}.
GateFusion run_gate_fusion(const ov::Output<ov::Node>& w_down,
                           const ov::Output<ov::Node>& w_up,
                           const gate_fn& make_gate,
                           bool allow_compressed,
                           const std::function<ov::OutputVector(const ov::Output<ov::Node>& rbar)>& extra_outputs = nullptr) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    ov::Output<ov::Node> rbar;
    auto mixed = build_gr_read_with_weights(residual, w_down, w_up, &rbar);
    ov::OutputVector outputs{mixed, make_gate(residual, rbar)};
    if (extra_outputs) {
        const auto extra = extra_outputs(rbar);
        outputs.insert(outputs.end(), extra.begin(), extra.end());
    }
    GateFusion result;
    result.model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{residual});
    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>(allow_compressed);
    manager.run_passes(result.model);
    for (const auto& node : result.model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            result.gr = gr;
        }
    }
    return result;
}

void expect_gate_fused(const GateFusion& r, size_t inputs) {
    ASSERT_NE(r.gr, nullptr);
    EXPECT_EQ(r.gr->get_input_size(), inputs);
    EXPECT_TRUE(r.gr->get_produce_write_gate());
    ASSERT_EQ(r.gr->get_output_size(), 2u);
    EXPECT_EQ(r.gr->get_output_partial_shape(1), (ov::PartialShape{T, C}));
    // both model outputs come from the fused op, and no MatMul / Sigmoid of the write-gate branch is left
    const auto mixed_src = r.model->get_results()[0]->input_value(0);
    const auto gate_src = r.model->get_results()[1]->input_value(0);
    EXPECT_EQ(mixed_src.get_node(), r.gr.get());
    EXPECT_EQ(mixed_src.get_index(), 0u);
    EXPECT_EQ(gate_src.get_node(), r.gr.get());
    EXPECT_EQ(gate_src.get_index(), 1u);
    for (const auto& node : r.model->get_ops()) {
        EXPECT_EQ(ov::as_type_ptr<ov::op::v0::MatMul>(node), nullptr);
        EXPECT_EQ(ov::as_type_ptr<ov::op::v0::Sigmoid>(node), nullptr);
    }
}

void expect_gate_not_fused(const GateFusion& r, size_t inputs) {
    ASSERT_NE(r.gr, nullptr) << "the READ itself must still be fused";
    EXPECT_EQ(r.gr->get_input_size(), inputs);
    EXPECT_FALSE(r.gr->get_produce_write_gate());
    EXPECT_EQ(r.gr->get_output_size(), 1u);
    EXPECT_NE(r.model->get_results()[1]->input_value(0).get_node(), r.gr.get());
}

gate_fn gate_on_rbar(GateScale form = GateScale::multiply, float scale = 1.0f / C, float times = 2.0f) {
    return [=](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        return build_write_gate(rbar, f16_inject_weights(), form, scale, times);
    };
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> f16_read_weights() {
    return {
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(LQ), static_cast<size_t>(C * D)}, std::vector<float>(LQ * C * D, 0.01f)),
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(C * D), static_cast<size_t>(LQ)}, std::vector<float>(LQ * C * D, 0.01f))};
}

}  // namespace

TEST(gated_residual_fusion_write_gate, multiply_scale_is_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    expect_gate_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(), false), 5u);
}

TEST(gated_residual_fusion_write_gate, divide_scale_is_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    expect_gate_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::divide, static_cast<float>(C)), false), 5u);
}

TEST(gated_residual_fusion_write_gate, scale_folded_into_w_inject_is_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    auto r = run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::folded), false);
    expect_gate_fused(r, 5u);
    // the fused op reads the unscaled W_inject: the 1/C is applied inside the kernel
    auto w_inject = ov::as_type_ptr<ov::op::v0::Constant>(r.gr->get_input_node_shared_ptr(r.gr->get_read_layout()->w_inject));
    EXPECT_NE(w_inject, nullptr);
}

TEST(gated_residual_fusion_write_gate, write_gate_feeds_the_write_fusion) {
    const auto [w_down, w_up] = f16_read_weights();
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    ov::Output<ov::Node> rbar;
    auto mixed = build_gr_read_with_weights(residual, w_down, w_up, &rbar);
    auto gate = build_write_gate(rbar, f16_inject_weights());
    // R' = R + S * Y, canonical case A
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, D});
    auto r3 = reshape_to(residual, {-1, C, D});
    auto s3 = reshape_to(gate, {-1, C, 1});
    auto y3 = reshape_to(y, {-1, 1, D});
    auto updated = reshape_to(std::make_shared<ov::op::v1::Add>(r3, std::make_shared<ov::op::v1::Multiply>(s3, y3)), {-1, C * D});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed, updated}, ov::ParameterVector{residual, y});
    ov::pass::Manager manager;
    manager.register_pass<GatedResidualFusion>(false);
    manager.run_passes(model);

    std::shared_ptr<op::GatedResidual> read;
    std::shared_ptr<op::GatedResidual> write;
    for (const auto& node : model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            (gr->get_mode() == op::GatedResidual::Mode::READ ? read : write) = gr;
        }
    }
    ASSERT_NE(read, nullptr);
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(read->get_produce_write_gate());
    EXPECT_EQ(write->input_value(op::GatedResidual::WriteInputIdx::WRITE_GATE).get_node(), read.get());
}

TEST(gated_residual_fusion_write_gate, wrong_branch_scale_leaves_the_gate_in_the_graph) {
    const auto [w_down, w_up] = f16_read_weights();
    expect_gate_not_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::multiply, 1.0f / (C + 1)), false), 4u);
    expect_gate_not_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::divide, static_cast<float>(C + 1)), false), 4u);
}

TEST(gated_residual_fusion_write_gate, wrong_gate_range_leaves_the_gate_in_the_graph) {
    const auto [w_down, w_up] = f16_read_weights();
    // a plain sigmoid or 2 * sigmoid scaled differently is not the write gate
    expect_gate_not_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::multiply, 1.0f / C, 1.0f), false), 4u);
    expect_gate_not_fused(run_gate_fusion(w_down, w_up, gate_on_rbar(GateScale::multiply, 1.0f / C, 3.0f), false), 4u);
}

TEST(gated_residual_fusion_write_gate, branch_on_another_normalization_is_not_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    // the inject MatMul reads its own RMS of the residual, not the Rbar of the READ
    gate_fn other_rbar = [](const ov::Output<ov::Node>& residual, const ov::Output<ov::Node>&) {
        auto r3 = reshape_to(residual, {-1, C, D});
        auto rms = std::make_shared<ov::op::internal::RMS>(r3, EPS);
        return build_write_gate(reshape_to(rms, {-1, C * D}), f16_inject_weights());
    };
    expect_gate_not_fused(run_gate_fusion(w_down, w_up, other_rbar, false), 4u);
}

TEST(gated_residual_fusion_write_gate, transposed_or_misshaped_w_inject_is_not_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    gate_fn misshaped = [](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        auto w = ov::op::v0::Constant::create(ov::element::f16,
                                              ov::Shape{static_cast<size_t>(C + 4), static_cast<size_t>(C * D)},
                                              std::vector<float>((C + 4) * C * D, 0.02f));
        return build_write_gate(rbar, w);
    };
    auto r = run_gate_fusion(w_down, w_up, misshaped, false);
    ASSERT_NE(r.gr, nullptr);
    EXPECT_FALSE(r.gr->get_produce_write_gate());
}

TEST(gated_residual_fusion_write_gate, gate_chain_with_other_consumers_is_not_fused) {
    const auto [w_down, w_up] = f16_read_weights();
    // the raw inject MatMul is also a model output, so it cannot be removed
    ov::Output<ov::Node> inject_mm;
    gate_fn make_gate = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        auto mm = std::make_shared<ov::op::v0::MatMul>(rbar, f16_inject_weights(), false, true);
        inject_mm = mm->output(0);
        auto scaled = std::make_shared<ov::op::v1::Multiply>(mm, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f / C}));
        auto sig = std::make_shared<ov::op::v0::Sigmoid>(scaled);
        return std::make_shared<ov::op::v1::Multiply>(sig, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {2.0f}))->output(0);
    };
    auto r = run_gate_fusion(w_down, w_up, make_gate, false, [&](const ov::Output<ov::Node>&) {
        return ov::OutputVector{inject_mm};
    });
    ASSERT_NE(r.gr, nullptr);
    EXPECT_FALSE(r.gr->get_produce_write_gate());
}

TEST(gated_residual_fusion_write_gate, compressed_weights_take_a_compressed_w_inject) {
    const size_t cd = C * D;
    const auto c = static_cast<size_t>(C);
    const auto lq = static_cast<size_t>(LQ);
    struct Case {
        ov::element::Type wt;
        size_t group_k_cd;  // group size along C*D (W_down, W_inject)
        size_t group_k_l;   // group size along L (W_up)
        bool has_zp;
    };
    for (const auto& tc : {Case{ov::element::u4, 32, 32, true},
                           Case{ov::element::u8, cd, lq, true},
                           Case{ov::element::i4, 32, 32, true},
                           Case{ov::element::i8, cd, lq, true},
                           Case{ov::element::u4, 32, 32, false},
                           Case{ov::element::i8, cd, lq, false}}) {
        gate_fn make_gate = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
            return build_write_gate(rbar, make_decompressed(tc.wt, c, cd, tc.group_k_cd, false, tc.has_zp));
        };
        auto r = run_gate_fusion(make_decompressed(tc.wt, lq, cd, tc.group_k_cd, false, tc.has_zp),
                                 make_decompressed(tc.wt, cd, lq, tc.group_k_l, false, tc.has_zp),
                                 make_gate,
                                 true);
        expect_gate_fused(r, tc.has_zp ? 11u : 8u);
        const auto layout = r.gr->get_read_layout();
        ASSERT_TRUE(layout.has_value());
        EXPECT_EQ(layout->has_zp, tc.has_zp);
        EXPECT_EQ(r.gr->get_input_element_type(layout->w_inject), tc.wt);
        EXPECT_EQ(r.gr->get_input_shape(layout->w_inject), (ov::Shape{c, cd}));
        EXPECT_EQ(r.gr->get_input_element_type(layout->scale_inject), ov::element::f16);
        EXPECT_EQ(r.gr->get_input_shape(layout->scale_inject), (ov::Shape{cd / tc.group_k_cd, c}));
        if (tc.has_zp) {
            EXPECT_EQ(r.gr->get_input_element_type(layout->zp_inject), tc.wt);
            EXPECT_EQ(r.gr->get_input_shape(layout->zp_inject), (ov::Shape{cd / tc.group_k_cd, c}));
        }
    }
}

TEST(gated_residual_fusion_write_gate, compressed_read_with_f16_or_mismatched_w_inject_fuses_only_the_read) {
    const size_t cd = C * D;
    const auto c = static_cast<size_t>(C);
    // W_inject stays an f16 tensor while W_down / W_up are compressed
    auto f16_inject = run_gate_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), gate_on_rbar(), true);
    expect_gate_not_fused(f16_inject, 8u);
    // W_inject uses another weight type than W_down / W_up
    gate_fn u8_inject = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        return build_write_gate(rbar, make_decompressed(ov::element::u8, c, cd, cd));
    };
    expect_gate_not_fused(run_gate_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), u8_inject, true), 8u);
    // W_inject with an unsupported group size
    gate_fn g16_inject = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        return build_write_gate(rbar, make_decompressed(ov::element::u4, c, cd, 16));
    };
    expect_gate_not_fused(run_gate_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), g16_inject, true),
                          8u);
    // W_inject may use another group size than W_down / W_up
    gate_fn g128_inject = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        return build_write_gate(rbar, make_decompressed(ov::element::u4, c, cd, 128));
    };
    expect_gate_fused(run_gate_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), g128_inject, true), 11u);
}

TEST(gated_residual_fusion_write_gate, compressed_w_inject_with_f16_read_weights_uses_the_decompressed_tensor) {
    const size_t cd = C * D;
    const auto c = static_cast<size_t>(C);
    const auto [w_down, w_up] = f16_read_weights();
    gate_fn compressed_inject = [&](const ov::Output<ov::Node>&, const ov::Output<ov::Node>& rbar) {
        return build_write_gate(rbar, make_decompressed(ov::element::u4, c, cd, 32));
    };
    expect_gate_fused(run_gate_fusion(w_down, w_up, compressed_inject, true), 5u);
}
