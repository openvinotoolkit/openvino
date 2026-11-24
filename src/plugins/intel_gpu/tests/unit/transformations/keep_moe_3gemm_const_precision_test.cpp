// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_moe_3gemm_const_precision.hpp"

#include <gtest/gtest.h>

#include <array>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

using ov::op::internal::MOECompressed;

constexpr size_t num_experts = 2;
constexpr size_t hidden_size = 64;
constexpr size_t inter_size = 32;
constexpr size_t group_size = 32;

// MOECompressed GEMM3 inputs: gate/up/down weights at 3/6/9, their zero points at 5/8/11.
constexpr std::array<size_t, 3> weight_inputs{3, 6, 9};
constexpr std::array<size_t, 3> zp_inputs{5, 8, 11};

// An element::dynamic zero-point type builds the symmetric placeholder (has_zp=false).
std::shared_ptr<ov::Model> make_moe_model(ov::element::Type weight_type, ov::element::Type zp_type, bool mark_weights = false, bool mark_zero_points = false) {
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
        if (mark_weights) {
            ov::enable_keep_const_precision(weights);
        }
        if (mark_zero_points) {
            ov::enable_keep_const_precision(zp);
        }
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

std::shared_ptr<ov::Node> moe_input(const std::shared_ptr<ov::Model>& model, size_t index) {
    return model->get_results().at(0)->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(index);
}

// The fixture catches a missing mark but not an extra one, so Constants that must stay unmarked are checked explicitly.
class KeepMOE3GemmConstPrecisionTest : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<KeepMOE3GemmConstPrecision>();
    }
};

void run_pass_and_expect_unmarked(const std::shared_ptr<ov::Model>& model, const std::array<size_t, 3>& inputs) {
    ov::pass::Manager manager;
    manager.register_pass<KeepMOE3GemmConstPrecision>();
    manager.run_passes(model);

    for (const auto index : inputs) {
        EXPECT_FALSE(ov::is_keep_const_precision(moe_input(model, index))) << "MOECompressed input " << index;
    }
}

}  // namespace

// GGUF Q4_1/Q4_K import: u4 weights with u8 zero points.
TEST_F(KeepMOE3GemmConstPrecisionTest, U4WeightsWithU8ZeroPoints) {
    model = make_moe_model(ov::element::u4, ov::element::u8);
    model_ref = make_moe_model(ov::element::u4, ov::element::u8, /*mark_weights=*/true);
}

TEST_F(KeepMOE3GemmConstPrecisionTest, U4WeightsWithU4ZeroPoints) {
    model = make_moe_model(ov::element::u4, ov::element::u4);
    model_ref = make_moe_model(ov::element::u4, ov::element::u4, /*mark_weights=*/true, /*mark_zero_points=*/true);
}

TEST_F(KeepMOE3GemmConstPrecisionTest, SymmetricI4Weights) {
    model = make_moe_model(ov::element::i4, ov::element::dynamic);
    model_ref = make_moe_model(ov::element::i4, ov::element::dynamic, /*mark_weights=*/true);
}

TEST(KeepMOE3GemmConstPrecisionUnmarkedTest, U8ZeroPoints) {
    run_pass_and_expect_unmarked(make_moe_model(ov::element::u4, ov::element::u8), zp_inputs);
}

TEST(KeepMOE3GemmConstPrecisionUnmarkedTest, SymmetricZeroPointPlaceholders) {
    run_pass_and_expect_unmarked(make_moe_model(ov::element::i4, ov::element::dynamic), zp_inputs);
}

// 8-bit weights are never unpacked, so the pass leaves them alone.
TEST(KeepMOE3GemmConstPrecisionUnmarkedTest, U8Weights) {
    run_pass_and_expect_unmarked(make_moe_model(ov::element::u8, ov::element::u8), weight_inputs);
}
