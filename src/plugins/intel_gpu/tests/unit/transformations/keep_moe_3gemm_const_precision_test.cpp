// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_moe_3gemm_const_precision.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using namespace ov::intel_gpu;

namespace {

using ov::op::internal::MOECompressed;

constexpr size_t num_experts = 2;
constexpr size_t hidden_size = 64;
constexpr size_t inter_size = 32;
constexpr size_t group_size = 32;

struct MoEConstants {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> weights;
    std::vector<std::shared_ptr<ov::op::v0::Constant>> zero_points;
};

// GEMM3 MOECompressed with gate/up/down weights and zero points as Constants.
// An element::dynamic zero-point type builds the symmetric placeholder (has_zp=false).
std::shared_ptr<ov::Model> make_moe_model(ov::element::Type weight_type, ov::element::Type zp_type, MoEConstants& constants) {
    const bool has_zp = zp_type != ov::element::dynamic;
    auto hidden_states = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, hidden_size});
    auto routing_weights = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 1});
    auto topk_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4, 1});

    ov::OutputVector args{hidden_states, routing_weights, topk_indices};
    // {ofm, K} of gate, up and down.
    for (const auto& [ofm, k] : {std::pair{inter_size, hidden_size}, std::pair{inter_size, hidden_size}, std::pair{hidden_size, inter_size}}) {
        const size_t groups = k / group_size;
        auto weights = ov::op::v0::Constant::create(weight_type, ov::Shape{num_experts, ofm, groups, group_size}, {1});
        auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{num_experts, ofm, groups}, {0.01f});
        auto zp = has_zp ? ov::op::v0::Constant::create(zp_type, ov::Shape{num_experts, ofm, groups}, {1})
                         : ov::op::v0::Constant::create(ov::element::dynamic, ov::Shape{0}, {});
        constants.weights.push_back(weights);
        constants.zero_points.push_back(zp);
        args.insert(args.end(), {weights, scale, zp});
    }

    MOECompressed::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.hidden_size = hidden_size;
    config.inter_size = inter_size;
    config.num_expert = num_experts;
    config.group_size = group_size;
    config.top_k = 1;
    config.has_zp = has_zp;
    config.out_type = ov::element::f16;
    auto moe = std::make_shared<MOECompressed>(args, config);
    return std::make_shared<ov::Model>(moe->outputs(), ov::ParameterVector{hidden_states, routing_weights, topk_indices});
}

void run_pass(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<KeepMOE3GemmConstPrecision>();
    manager.run_passes(model);
}

}  // namespace

// GGUF Q4_1/Q4_K import: u4 weights with u8 zero points. The weights must stay u4; u8 is not unpacked anyway.
TEST(KeepMOE3GemmConstPrecisionTest, U4WeightsWithU8ZeroPoints) {
    MoEConstants constants;
    auto model = make_moe_model(ov::element::u4, ov::element::u8, constants);
    run_pass(model);

    for (const auto& weights : constants.weights) {
        EXPECT_TRUE(ov::is_keep_const_precision(weights));
    }
    for (const auto& zp : constants.zero_points) {
        EXPECT_FALSE(ov::is_keep_const_precision(zp));
    }
}

TEST(KeepMOE3GemmConstPrecisionTest, U4WeightsWithU4ZeroPoints) {
    MoEConstants constants;
    auto model = make_moe_model(ov::element::u4, ov::element::u4, constants);
    run_pass(model);

    for (const auto& weights : constants.weights) {
        EXPECT_TRUE(ov::is_keep_const_precision(weights));
    }
    for (const auto& zp : constants.zero_points) {
        EXPECT_TRUE(ov::is_keep_const_precision(zp));
    }
}

TEST(KeepMOE3GemmConstPrecisionTest, SymmetricI4Weights) {
    MoEConstants constants;
    auto model = make_moe_model(ov::element::i4, ov::element::dynamic, constants);
    run_pass(model);

    for (const auto& weights : constants.weights) {
        EXPECT_TRUE(ov::is_keep_const_precision(weights));
    }
    for (const auto& zp : constants.zero_points) {
        EXPECT_FALSE(ov::is_keep_const_precision(zp));
    }
}

// 8-bit weights are never unpacked, so the pass leaves them alone.
TEST(KeepMOE3GemmConstPrecisionTest, U8WeightsAreNotMarked) {
    MoEConstants constants;
    auto model = make_moe_model(ov::element::u8, ov::element::u8, constants);
    run_pass(model);

    for (const auto& weights : constants.weights) {
        EXPECT_FALSE(ov::is_keep_const_precision(weights));
    }
    for (const auto& zp : constants.zero_points) {
        EXPECT_FALSE(ov::is_keep_const_precision(zp));
    }
}
