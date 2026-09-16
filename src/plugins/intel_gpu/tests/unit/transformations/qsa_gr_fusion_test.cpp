// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/op/qsa_indexer_cache.hpp"
#include "intel_gpu/op/qsa_sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "plugin/transformations/qsa_gr_fusion.hpp"

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
std::shared_ptr<ov::Node> build_gr_read(const std::shared_ptr<ov::op::v0::Parameter>& residual, float low_scale, float mix_scale, int64_t branches) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    // The gain is per (branch, channel): shape [C, D], i.e. C independent groups of width D.
    auto gamma = ov::op::v0::Constant::create(ov::element::f32,
                                              ov::Shape{static_cast<size_t>(branches), static_cast<size_t>(D)},
                                              std::vector<float>(branches * D, 1.0f));
    auto rms = std::make_shared<ov::op::internal::RMS>(r3, gamma, EPS);
    auto rbar = reshape_to(rms, {-1, branches * D});

    auto w_down = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {low_scale}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up, false, true);
    auto gate = std::make_shared<ov::op::v0::Sigmoid>(up);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(gate, rbar);

    auto weighted3 = reshape_to(weighted, {-1, branches, D});
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted3, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    return std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {mix_scale}));
}

/// Builds the GR READ sub-graph in the *decomposed* form emitted by checkpoints that
/// were exported before RMS-fusion ran: Power -> ReduceMean -> Add(eps) -> Sqrt ->
/// Divide -> Multiply for the norm, an explicit `gain + 1` fold, and a ReduceMean
/// (rather than ReduceSum + Multiply) for the branch mix.
std::shared_ptr<ov::Node> build_gr_read_decomposed(const std::shared_ptr<ov::op::v0::Parameter>& residual, float low_scale, int64_t branches) {
    auto r3 = reshape_to(residual, {-1, branches, D});
    auto sq = std::make_shared<ov::op::v1::Power>(r3, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f}));
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    auto biased = std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {EPS}));
    auto rms = std::make_shared<ov::op::v0::Sqrt>(biased);
    auto inv_rms = std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}), rms);
    auto normalized = std::make_shared<ov::op::v1::Multiply>(r3, inv_rms);
    auto normalized_flat = reshape_to(normalized, {-1, branches * D});
    auto gain_raw = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(branches * D)}, std::vector<float>(branches * D, 0.0f));
    auto gain_plus_one = std::make_shared<ov::op::v1::Add>(gain_raw, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
    auto rbar = std::make_shared<ov::op::v1::Multiply>(normalized_flat, gain_plus_one);

    auto w_down = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{L, static_cast<size_t>(branches * D)}, std::vector<float>(L * branches * D, 0.01f));
    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down, false, true);
    auto low_scaled = std::make_shared<ov::op::v1::Multiply>(low, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {low_scale}));
    auto act = std::make_shared<ov::op::v4::Swish>(low_scaled);

    auto w_up = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(branches * D), L}, std::vector<float>(branches * D * L, 0.01f));
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

}  // namespace

TEST(qsa_gr_fusion, gated_residual_read_is_fused) {
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
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
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
    auto mixed = build_gr_read(residual, 1.0f, 1.0f / C, C);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual});

    ov::pass::Manager manager;
    manager.register_pass<GatedResidualReadFusion>();
    manager.run_passes(model);

    ASSERT_EQ(count_gated_residual(model), 0u) << "the generic graph must be preserved when 1/C is missing";
}

TEST(qsa_gr_fusion, gated_residual_read_rejects_unsupported_branch_count) {
    constexpr int64_t branches = 2;
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, branches * D});
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
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
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
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, D});

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
    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C * D});
    auto y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, D});
    auto s = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{T, C});

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

namespace {

/// Builds the QSA indexer scoring signature: ReLU per head -> head reduction -> scale -> TopK.
struct indexer_graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v11::TopK> topk;
};

indexer_graph build_indexer(int64_t heads, int64_t head_dim, int64_t blocks, int64_t topk, float scale) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, heads, head_dim});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, blocks, head_dim});
    auto logits = std::make_shared<ov::op::v0::MatMul>(q, k, false, true);
    auto relu = std::make_shared<ov::op::v0::Relu>(logits);
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(relu, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(reduced, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {scale}));
    auto topk_node = std::make_shared<ov::op::v11::TopK>(scaled,
                                                         ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                                         -1,
                                                         ov::op::v11::TopK::Mode::MAX,
                                                         ov::op::v11::TopK::SortType::SORT_VALUES);
    return {std::make_shared<ov::Model>(ov::OutputVector{topk_node->output(1)}, ov::ParameterVector{q, k}), topk_node};
}

/// Builds the indexer signature in the form actually emitted by the checkpoint export:
/// a Reshape between the QK score and the per-head ReLU, a Divide by sqrt(Di) instead
/// of a Multiply by 1/sqrt(Di), and a causal-mask Select right before TopK.
indexer_graph build_indexer_decomposed(int64_t heads, int64_t head_dim, int64_t blocks, int64_t topk) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, heads, head_dim});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, blocks, head_dim});
    auto logits = std::make_shared<ov::op::v0::MatMul>(q, k, false, true);
    auto logits_reshaped = std::make_shared<ov::op::v1::Reshape>(
        logits, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, heads, blocks}), false);
    auto relu = std::make_shared<ov::op::v0::Relu>(logits_reshaped);
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(relu, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}), false);
    auto scaled = std::make_shared<ov::op::v1::Divide>(
        reduced, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {std::sqrt(static_cast<float>(head_dim))}));
    auto cond = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1, static_cast<size_t>(blocks)}, std::vector<char>(static_cast<size_t>(blocks), 0));
    auto fill = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1e9f});
    auto masked = std::make_shared<ov::op::v1::Select>(cond, fill, scaled);
    auto topk_node = std::make_shared<ov::op::v11::TopK>(masked,
                                                         ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                                         -1,
                                                         ov::op::v11::TopK::Mode::MAX,
                                                         ov::op::v11::TopK::SortType::SORT_VALUES);
    return {std::make_shared<ov::Model>(ov::OutputVector{topk_node->output(1)}, ov::ParameterVector{q, k}), topk_node};
}

}  // namespace

TEST(qsa_gr_fusion, qsa_indexer_signature_is_detected) {
    constexpr int64_t head_dim = 128;
    constexpr int64_t topk = 32;
    auto g = build_indexer(4, head_dim, 64, topk, 1.0f / std::sqrt(static_cast<float>(head_dim)));

    ov::pass::Manager manager;
    manager.register_pass<QSAFusion>();
    manager.run_passes(g.model);

    const auto& rt = g.topk->get_rt_info();
    ASSERT_TRUE(rt.count("qsa_indexer_detected"));
    EXPECT_EQ(rt.at("qsa_indexer_head_dim").as<int64_t>(), head_dim);
    EXPECT_EQ(rt.at("qsa_block_topk").as<int64_t>(), topk);
}

TEST(qsa_gr_fusion, qsa_indexer_decomposed_form_is_detected) {
    // Mirrors the real export: Reshape before ReLU, Divide instead of Multiply for the
    // 1/sqrt(Di) scale, and a mask Select between the scale and TopK.
    constexpr int64_t head_dim = 128;
    constexpr int64_t topk = 32;
    auto g = build_indexer_decomposed(4, head_dim, 64, topk);

    ov::pass::Manager manager;
    manager.register_pass<QSAFusion>();
    manager.run_passes(g.model);

    const auto& rt = g.topk->get_rt_info();
    ASSERT_TRUE(rt.count("qsa_indexer_detected"));
    EXPECT_EQ(rt.at("qsa_indexer_head_dim").as<int64_t>(), head_dim);
    EXPECT_EQ(rt.at("qsa_block_topk").as<int64_t>(), topk);
}

TEST(qsa_gr_fusion, qsa_indexer_rejects_wrong_scale_position) {
    // 1/sqrt(Di) must be applied AFTER the ReLU sum; a different constant means the
    // scale was folded somewhere else and the operator is not QSA.
    constexpr int64_t head_dim = 128;
    auto g = build_indexer(4, head_dim, 64, 32, 0.5f);

    ov::pass::Manager manager;
    manager.register_pass<QSAFusion>();
    manager.run_passes(g.model);

    ASSERT_FALSE(g.topk->get_rt_info().count("qsa_indexer_detected"));
}

// ===========================================================================
// QSASDPAFusion
// ===========================================================================
namespace {

constexpr int64_t sdpa_Di = 16;    // indexer head dim
constexpr int64_t sdpa_Hidx = 2;   // indexer heads
constexpr int64_t sdpa_D = 32;     // hidden size
constexpr int64_t sdpa_Hq = 2;     // main attention heads
constexpr int64_t sdpa_Dh = 16;    // main K head size
constexpr int64_t sdpa_r = 4;      // compress ratio
constexpr int64_t sdpa_S = 8;      // main K/V cache capacity (static case)
constexpr int64_t sdpa_topk = 1;   // block_topk

template <typename T>
size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::as_type_ptr<T>(node)) {
            count++;
        }
    }
    return count;
}

/// Builds a single-token (T=1, decode-style), batch-1 SDPA-style QSA layer:
///   - a shared `index_qk_proj`-style MatMul producing indexer Q and the current
///     token's indexer K contribution;
///   - indexer Q/K RMSNorm gains ([Di]-sized Constants) and a shared, already-packed
///     rotary table (when `include_rotary_table` is true);
///   - a ReadValue-backed indexer raw-K history, block-mean-pooled with a static
///     group size `r` (mirrors the real checkpoint's from-scratch recomputation);
///   - the QSAFusion-recognized indexer score signature (ReLU -> ReduceSum ->
///     1/sqrt(Di) -> TopK);
///   - a dense `ScaledDotProductAttention`, reachable from the TopK output, fed by
///     ReadValue-backed main K/V (static shape when `static_kv_shape` is true,
///     dynamic - mirroring the real Qwen4-Exp export - otherwise).
struct qsasdpa_graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    std::shared_ptr<ov::op::v11::TopK> topk;
};

qsasdpa_graph build_qsasdpa_graph(bool static_kv_shape, bool include_rotary_table) {
    auto i64_const = [](std::vector<int64_t> v) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
    };

    // --- indexer Q/K projection: mixed -> shared MatMul -> per-leg Slice ----------
    auto mixed = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, sdpa_D});
    auto qk_weight =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>((sdpa_Hidx + 1) * sdpa_Di), static_cast<size_t>(sdpa_D)},
                                     std::vector<float>((sdpa_Hidx + 1) * sdpa_Di * sdpa_D, 0.01f));
    auto qk_proj = std::make_shared<ov::op::v0::MatMul>(mixed, qk_weight, false, true);  // [1,1,(Hidx+1)*Di]

    auto q_slice = std::make_shared<ov::op::v8::Slice>(qk_proj, i64_const({0, 0, 0}), i64_const({1, 1, sdpa_Hidx * sdpa_Di}), i64_const({1, 1, 1}));
    auto q_headsplit = std::make_shared<ov::op::v1::Reshape>(q_slice, i64_const({1, 1, sdpa_Hidx, sdpa_Di}), false);
    auto q_gain = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(sdpa_Di)}, std::vector<float>(sdpa_Di, 1.0f));
    auto q_normed = std::make_shared<ov::op::v1::Multiply>(q_headsplit, q_gain);

    auto k_slice =
        std::make_shared<ov::op::v8::Slice>(qk_proj, i64_const({0, 0, sdpa_Hidx * sdpa_Di}), i64_const({1, 1, (sdpa_Hidx + 1) * sdpa_Di}),
                                            i64_const({1, 1, 1}));
    auto k_headsplit = std::make_shared<ov::op::v1::Reshape>(k_slice, i64_const({1, 1, 1, sdpa_Di}), false);
    auto k_gain = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{static_cast<size_t>(sdpa_Di)}, std::vector<float>(sdpa_Di, 1.0f));
    auto k_normed = std::make_shared<ov::op::v1::Multiply>(k_headsplit, k_gain);

    ov::Output<ov::Node> idx_q = q_normed;
    ov::Output<ov::Node> new_k = k_normed;
    if (include_rotary_table) {
        // A single already-packed [positions, Di] table shared by the Q and K legs
        // (full-width rotary here, purely to keep this structural test's broadcast
        // shapes simple; the fusion pass itself does not require full-width rotary).
        auto rope_table = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{64, static_cast<size_t>(sdpa_Di)},
                                                        std::vector<float>(64 * sdpa_Di, 0.0f));
        auto rope_slice = std::make_shared<ov::op::v8::Slice>(rope_table, i64_const({0, 0}), i64_const({1, sdpa_Di}), i64_const({1, 1}));
        auto rope_reshaped = std::make_shared<ov::op::v1::Reshape>(rope_slice, i64_const({1, 1, 1, sdpa_Di}), false);
        idx_q = std::make_shared<ov::op::v1::Add>(q_normed, rope_reshaped);
        new_k = std::make_shared<ov::op::v1::Add>(k_normed, rope_reshaped);
    }

    // --- indexer raw-K history: ReadValue -> Concat(new_k) ------------------------
    auto raw_k_variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ov::PartialShape{1, ov::Dimension::dynamic(), sdpa_Di},
                                                                            ov::element::f32, "qsasdpa_raw_k"});
    auto raw_k_read = std::make_shared<ov::op::v6::ReadValue>(raw_k_variable);
    auto new_k_flat = std::make_shared<ov::op::v1::Reshape>(new_k, i64_const({1, 1, sdpa_Di}), false);
    auto raw_k_history = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{raw_k_read, new_k_flat}, 1);  // [1, ?, Di]

    // --- block-mean pooling with a STATIC group size r (n_blocks stays dynamic) ---
    auto pool_reshaped = std::make_shared<ov::op::v1::Reshape>(raw_k_history, i64_const({1, -1, sdpa_r, sdpa_Di}), false);  // [1,?,r,Di]
    auto pooled = std::make_shared<ov::op::v1::ReduceMean>(pool_reshaped, i64_const({2}), false);                          // [1,?,Di]
    auto idx_k = std::make_shared<ov::op::v0::Unsqueeze>(pooled, i64_const({1}));                                          // [1,1,?,Di]

    // --- QSAFusion-recognized indexer score signature -----------------------------
    auto logits = std::make_shared<ov::op::v0::MatMul>(idx_q, idx_k, false, true);  // [1,1,Hidx,?]
    auto relu = std::make_shared<ov::op::v0::Relu>(logits);
    auto reduced = std::make_shared<ov::op::v1::ReduceSum>(relu, i64_const({2}), false);  // [1,1,?]
    auto scaled = std::make_shared<ov::op::v1::Multiply>(
        reduced, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / std::sqrt(static_cast<float>(sdpa_Di))}));
    auto topk = std::make_shared<ov::op::v11::TopK>(scaled, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {sdpa_topk})->output(0), 2,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES);

    // --- main attention: ReadValue-backed K/V, reachable from the TopK output -----
    ov::PartialShape kv_shape{1, sdpa_Hq, static_cast<int64_t>(static_kv_shape ? sdpa_S : -1), sdpa_Dh};
    auto k_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, ov::element::f32, "qsasdpa_main_k"});
    auto v_variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{kv_shape, ov::element::f32, "qsasdpa_main_v"});
    auto k_read = std::make_shared<ov::op::v6::ReadValue>(k_variable);
    auto v_read = std::make_shared<ov::op::v6::ReadValue>(v_variable);

    // Thread the TopK output into the graph feeding SDPA without constraining
    // broadcast-compatibility (any-rank ShapeOf -> scalar reduce -> * 0.0 -> Add).
    auto topk_shape = std::make_shared<ov::op::v3::ShapeOf>(topk->output(0), ov::element::i64);
    auto topk_shape_f = std::make_shared<ov::op::v0::Convert>(topk_shape, ov::element::f32);
    auto topk_scalar = std::make_shared<ov::op::v1::ReduceSum>(topk_shape_f, i64_const({0}), false);
    auto zeroed = std::make_shared<ov::op::v1::Multiply>(topk_scalar, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
    auto k_biased = std::make_shared<ov::op::v1::Add>(k_read, zeroed);

    auto q_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, sdpa_Hq, 1, sdpa_Dh});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_param, k_biased, v_read, true);
    auto result = std::make_shared<ov::op::v0::Result>(sdpa);

    ov::ParameterVector params{mixed, q_param};
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, params);
    return {model, sdpa, topk};
}

}  // namespace

TEST(qsa_gr_fusion, qsa_sdpa_fusion_bails_out_on_dynamic_kv_shape) {
    // Mirrors the actual Qwen4-Exp export: the main K/V ReadValue Variable declares a
    // dynamic sequence length, which QSASDPA v1 cannot satisfy (`pa_block_size` is a
    // JIT compile-time constant) -- the pass must correctly leave the graph untouched.
    auto g = build_qsasdpa_graph(/*static_kv_shape=*/false, /*include_rotary_table=*/true);

    ov::pass::Manager manager;
    manager.register_pass<QSASDPAFusion>();
    manager.run_passes(g.model);

    EXPECT_EQ(count_ops<op::QSASDPA>(g.model), 0u);
    EXPECT_EQ(count_ops<op::QSAIndexerCache>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 1u);
}

TEST(qsa_gr_fusion, qsa_sdpa_fusion_bails_out_without_packed_rotary_table) {
    // A model that only exposes the indexer's RoPE via a decomposed rotate_half over
    // Parameter-fed cos/sin tables (the real HF-style export's actual convention) has
    // no single provable [positions, rotary_dim] tensor; v1 does not attempt to repack
    // it and must leave the graph untouched.
    auto g = build_qsasdpa_graph(/*static_kv_shape=*/true, /*include_rotary_table=*/false);

    ov::pass::Manager manager;
    manager.register_pass<QSASDPAFusion>();
    manager.run_passes(g.model);

    EXPECT_EQ(count_ops<op::QSASDPA>(g.model), 0u);
    EXPECT_EQ(count_ops<op::QSAIndexerCache>(g.model), 0u);
    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 1u);
}

TEST(qsa_gr_fusion, qsa_sdpa_fusion_replaces_with_qsasdpa_and_indexer_cache) {
    // All required tensors are provable and static: the pass must perform the actual
    // replacement (unlike QSAFusion, which only ever tags rt_info).
    auto g = build_qsasdpa_graph(/*static_kv_shape=*/true, /*include_rotary_table=*/true);

    ov::pass::Manager manager;
    manager.register_pass<QSASDPAFusion>();
    manager.run_passes(g.model);

    ASSERT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(g.model), 0u) << "the original SDPA must be replaced, not kept alongside";
    ASSERT_EQ(count_ops<op::QSASDPA>(g.model), 1u);
    ASSERT_EQ(count_ops<op::QSAIndexerCache>(g.model), 1u);

    for (const auto& node : g.model->get_ops()) {
        if (auto qsasdpa = ov::as_type_ptr<op::QSASDPA>(node)) {
            const auto& config = qsasdpa->get_config();
            EXPECT_EQ(config.heads_num, static_cast<size_t>(sdpa_Hq));
            EXPECT_EQ(config.kv_heads_num, static_cast<size_t>(sdpa_Hq));
            EXPECT_EQ(config.k_head_size, static_cast<size_t>(sdpa_Dh));
            EXPECT_EQ(config.v_head_size, static_cast<size_t>(sdpa_Dh));
            EXPECT_EQ(config.hidden_size, static_cast<size_t>(sdpa_D));
            EXPECT_EQ(config.indexer_heads, static_cast<size_t>(sdpa_Hidx));
            EXPECT_EQ(config.indexer_head_dim, static_cast<size_t>(sdpa_Di));
            EXPECT_EQ(config.indexer_rotary_dim, static_cast<size_t>(sdpa_Di));
            EXPECT_EQ(config.compress_ratio, static_cast<size_t>(sdpa_r));
            EXPECT_EQ(config.block_topk, static_cast<size_t>(sdpa_topk));
            EXPECT_EQ(config.budget, static_cast<size_t>(sdpa_topk * sdpa_r));
            EXPECT_EQ(config.pa_block_size, static_cast<size_t>(sdpa_S));
        }
        if (auto cache = ov::as_type_ptr<op::QSAIndexerCache>(node)) {
            EXPECT_EQ(cache->get_variable_id(), "qsasdpa_raw_k") << "the existing raw-K Variable must be reused verbatim";
            EXPECT_NE(cache->get_summary_variable_id(), cache->get_variable_id());
        }
    }
}
