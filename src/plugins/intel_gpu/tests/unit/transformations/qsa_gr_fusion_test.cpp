// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/qsa_gr_fusion.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/gated_residual.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
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

/// Builds the canonical GR READ sub-graph and returns the final `mixed` node.
std::shared_ptr<ov::Node> build_gr_read(const std::shared_ptr<ov::op::v0::Parameter>& residual,
                                        float low_scale,
                                        float mix_scale,
                                        int64_t branches,
                                        const ov::element::Type& et = ov::element::f16) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    // The gain is per (branch, channel): shape [C, D], i.e. C independent groups of width D.
    auto gamma = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(branches), static_cast<size_t>(D)}, std::vector<float>(branches * D, 1.0f));
    auto rms = std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS);
    auto rbar = reshape_to(rms, {-1, branches * D});

    auto w_down = ov::op::v0::Constant::create(et, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(et, ov::Shape{}, {low_scale}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(et, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(gate, rbar);

    auto weighted3 = reshape_to(weighted, {-1, branches, D});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted3, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    return std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(et, ov::Shape{}, {mix_scale}));
}

/// Builds the GR READ sub-graph in the *decomposed* form emitted by checkpoints that
/// were exported before RMS-fusion ran: Power -> ReduceMean -> Add(eps) -> Sqrt ->
/// Divide -> Multiply for the norm, an explicit `gain + 1` fold, and a ReduceMean
/// (rather than ReduceSum + Multiply) for the branch mix.
std::shared_ptr<ov::Node> build_gr_read_decomposed(const std::shared_ptr<ov::op::v0::Parameter>& residual, float low_scale, int64_t branches) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    auto sq = std::make_shared<ov::op::v1::Power>(r3, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {2.0f}));
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    auto biased = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {EPS}));
    auto rms = std::make_shared<ov::op::v0::Sqrt>(biased);
    auto inv_rms = std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f}), rms);
    auto normalized = std::make_shared<ov::op::v1::Multiply>(r3, inv_rms);
    auto normalized_flat = reshape_to(normalized, {-1, branches * D});
    auto gain_raw = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(branches * D)}, std::vector<float>(branches * D, 0.0f));
    auto gain_plus_one = std::make_shared<ov::op::v1::Add>(gain_raw, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f}));
    auto rbar = std::make_shared<ov::op::v1::Multiply>(normalized_flat, gain_plus_one);

    auto w_down = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {low_scale}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);

    auto gate_split = reshape_to(gate, {-1, branches, D});
    auto rbar_split = reshape_to(rbar, {-1, branches, D});
    auto weighted = std::make_shared<ov::op::v1::Multiply>(gate_split, rbar_split);
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

/// NNCF-style asymmetric decompression: u4 with `group` < K is Constant[N,G,group] -> Convert -> Subtract(Convert(zp[N,G,1]))
/// -> Multiply(scale[N,G,1]) -> Reshape[N,K]; per-channel (group == K) has no Reshape and [N,1] scale / zp.
ov::Output<ov::Node> make_decompressed(const ov::element::Type& wt, size_t N, size_t K, size_t group, bool scalar_scale = false) {
    const size_t G = K / group;
    const bool grouped = group != K;
    const ov::Shape wshape = grouped ? ov::Shape{N, G, group} : ov::Shape{N, K};
    const ov::Shape sshape = grouped ? ov::Shape{N, G, 1} : ov::Shape{N, 1};
    auto w = ov::op::v0::Constant::create(wt, wshape, std::vector<int>(ov::shape_size(wshape), 3));
    auto wf = std::make_shared<ov::op::v0::Convert>(w, ov::element::f16);
    auto zp = ov::op::v0::Constant::create(wt, sshape, std::vector<int>(ov::shape_size(sshape), 1));
    auto zpf = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f16);
    auto sub = std::make_shared<ov::op::v1::Subtract>(wf, zpf);
    const ov::Shape scshape = scalar_scale ? ov::Shape{} : sshape;
    auto sc = ov::op::v0::Constant::create(ov::element::f16, scshape, std::vector<float>(ov::shape_size(scshape), 0.01f));
    auto mul = std::make_shared<ov::op::v1::Multiply>(sub, sc);
    if (!grouped) {
        return mul;
    }
    return reshape_to(mul, {static_cast<int64_t>(N), static_cast<int64_t>(K)});
}

/// The canonical GR READ sub-graph with caller-provided W_down [LQ, C*D] / W_up [C*D, LQ] producers.
std::shared_ptr<ov::Node> build_gr_read_with_weights(const std::shared_ptr<ov::op::v0::Parameter>& residual,
                                                     const ov::Output<ov::Node>& w_down,
                                                     const ov::Output<ov::Node>& w_up) {
    auto r3 = reshape_to(residual, {-1, C, D});
    auto gamma = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{static_cast<size_t>(C), static_cast<size_t>(D)}, std::vector<float>(C * D, 1.0f));
    auto rms = std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS);
    auto rbar = reshape_to(rms, {-1, C * D});
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

std::shared_ptr<op::GatedResidual> run_read_fusion(const ov::Output<ov::Node>& w_down, const ov::Output<ov::Node>& w_up, bool allow_compressed) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read_with_weights(residual, w_down, w_up);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});
    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>(allow_compressed);
    manager.run_passes(model);
    std::shared_ptr<op::GatedResidual> found;
    for (const auto& node : model->get_ops()) {
        if (auto gr = ov::as_type_ptr<op::GatedResidual>(node)) {
            found = gr;
        }
    }
    return found;
}

void expect_compressed(const std::shared_ptr<op::GatedResidual>& gr, const ov::element::Type& wt, size_t groups_down, size_t groups_up) {
    ASSERT_NE(gr, nullptr);
    ASSERT_EQ(gr->get_input_size(), 8u);
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::W_DOWN), wt);
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::W_UP), wt);
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::W_DOWN), (ov::Shape{static_cast<size_t>(LQ), static_cast<size_t>(C * D)}));
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::W_UP), (ov::Shape{static_cast<size_t>(C * D), static_cast<size_t>(LQ)}));
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::SCALE_DOWN), ov::element::f16);
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::SCALE_DOWN), (ov::Shape{groups_down, static_cast<size_t>(LQ)}));
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::SCALE_UP), (ov::Shape{groups_up, static_cast<size_t>(C * D)}));
    EXPECT_EQ(gr->get_input_element_type(op::GatedResidual::ReadInputIdx::ZP_DOWN), ov::element::u8);
    EXPECT_EQ(gr->get_input_shape(op::GatedResidual::ReadInputIdx::ZP_UP), (ov::Shape{groups_up, static_cast<size_t>(C * D)}));
}

}  // namespace

TEST(qsa_gr_fusion, gated_residual_read_is_fused) {
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
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::READ);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_lowrank(), static_cast<size_t>(L));
        EXPECT_FALSE(gr->get_produce_write_gate());
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    }
}

TEST(qsa_gr_fusion, gated_residual_read_requires_both_inv_c_scalings) {
    // The 1/C inside SiLU(W_down * Rbar / C) is model semantics, not a foldable constant.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u) << "the generic graph must be preserved when 1/C is missing";
}

TEST(qsa_gr_fusion, gated_residual_read_rejects_unsupported_branch_count) {
    constexpr int64_t branches = 2;
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{T, branches * D});
    auto mixed = build_gr_read(residual, 1.0f / branches, 1.0f / branches, branches);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u) << "only C == 4 is supported by the first version";
}

TEST(qsa_gr_fusion, gated_residual_read_decomposed_form_is_fused) {
    // Mirrors the graph shape actually emitted by the Qwen4-Exp checkpoint export:
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
        EXPECT_EQ(gr->get_mode(), op::GatedResidual::Mode::READ);
        EXPECT_EQ(gr->get_hc_count(), static_cast<size_t>(C));
        EXPECT_EQ(gr->get_hidden_size(), static_cast<size_t>(D));
        EXPECT_EQ(gr->get_lowrank(), static_cast<size_t>(L));
        EXPECT_FALSE(gr->get_produce_write_gate());
        EXPECT_EQ(gr->get_output_partial_shape(0), (ov::PartialShape{T, D}));
    }
}

TEST(qsa_gr_fusion, gated_residual_write_is_fused) {
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

TEST(qsa_gr_fusion, gated_residual_write_rejects_wrong_broadcast_axis) {
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

TEST(qsa_gr_fusion, gated_residual_write_decomposed_form_is_fused) {
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

class qsa_gr_fusion_dynamic_split : public ::testing::TestWithParam<bool> {};

TEST_P(qsa_gr_fusion_dynamic_split, gated_residual_read_rank3_dynamic_is_fused) {
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

TEST_P(qsa_gr_fusion_dynamic_split, gated_residual_write_rank3_dynamic_is_fused) {
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

INSTANTIATE_TEST_SUITE_P(smoke, qsa_gr_fusion_dynamic_split, ::testing::Values(false, true), [](const ::testing::TestParamInfo<bool>& info) {
    return info.param ? std::string("shape_of_target") : std::string("constant_target");
});

TEST(qsa_gr_fusion, gated_residual_read_rejects_rank_changing_split) {
    // [B, L, C*D] -> [-1, C, D] flattens B and L; the fused op would keep them.
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, T, C * D});
    auto mixed = build_gr_read(residual, 1.0f / C, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u);
}

TEST(qsa_gr_fusion, gated_residual_read_rejects_f32) {
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

TEST(qsa_gr_fusion, gated_residual_write_rejects_f32) {
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

TEST(qsa_gr_fusion, compressed_u4_group32_weights_are_fused) {
    const size_t cd = C * D;
    auto gr = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), true);
    expect_compressed(gr, ov::element::u4, cd / 32, LQ / 32);
}

TEST(qsa_gr_fusion, compressed_u8_per_channel_weights_are_fused) {
    const size_t cd = C * D;
    auto gr = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    expect_compressed(gr, ov::element::u8, 1, 1);
}

TEST(qsa_gr_fusion, compressed_scale_is_transposed_to_group_major) {
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

TEST(qsa_gr_fusion, compressed_weights_need_the_flag) {
    const size_t cd = C * D;
    auto gr = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u4, cd, LQ, 32), false);
    ASSERT_NE(gr, nullptr);
    EXPECT_EQ(gr->get_input_size(), 4u) << "the decompressed f16 tensors feed the op when compression is not allowed";
}

TEST(qsa_gr_fusion, compressed_rejects_unsupported_quantization) {
    const size_t cd = C * D;
    // group 64 is not a supported u4 group size
    auto g64 = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 64), make_decompressed(ov::element::u4, cd, LQ, 32), true);
    ASSERT_NE(g64, nullptr);
    EXPECT_EQ(g64->get_input_size(), 4u);
    // W_down and W_up must be compressed the same way
    auto mixed = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, 32), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    ASSERT_NE(mixed, nullptr);
    EXPECT_EQ(mixed->get_input_size(), 4u);
    // scalar scale is not a per-(group, channel) tensor
    auto scalar = run_read_fusion(make_decompressed(ov::element::u8, LQ, cd, cd, true), make_decompressed(ov::element::u8, cd, LQ, LQ), true);
    ASSERT_NE(scalar, nullptr);
    EXPECT_EQ(scalar->get_input_size(), 4u);
    // u4 per-channel (group == K) is not supported either
    auto u4_pc = run_read_fusion(make_decompressed(ov::element::u4, LQ, cd, cd), make_decompressed(ov::element::u4, cd, LQ, LQ), true);
    ASSERT_NE(u4_pc, nullptr);
    EXPECT_EQ(u4_pc->get_input_size(), 4u);
}
