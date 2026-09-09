// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/gated_residual.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/topk.hpp"
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
