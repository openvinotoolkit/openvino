// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/scale_down_moe_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

// Shapes used for all GEMM2_BIAS_SWIGLU_CLAMP tests.
// Matches the layout expected by CreateMOECompressedOp in src/plugin/ops/moe.cpp.
constexpr size_t kTokens = 4;
constexpr size_t kHiddenSize = 16;
constexpr size_t kInterSize = 8;  // ofm for up projection (2 * swiglu half)
constexpr size_t kNumExperts = 2;
constexpr size_t kTopK = 2;
constexpr size_t kGroupSize = 8;
constexpr size_t kNumGroups = kHiddenSize / kGroupSize;

// Build a GEMM2_BIAS_SWIGLU_CLAMP MOECompressed node with the given has_zp layout.
// Input order (see scale_down_moe_compressed.cpp):
//   0: hidden_states, 1: routing_weights, 2: topk_idx,
//   3: w_up, 4: scale_up, [5: zp_up,] <bias_up>,
//   <w_down>, <scale_down>, [<zp_down>,] <bias_down>
std::shared_ptr<ov::intel_gpu::op::MOECompressed> make_moe_compressed_gemm2(
    const ov::Output<ov::Node>& hidden_states,
    const ov::Output<ov::Node>& routing_weights,
    const ov::Output<ov::Node>& topk_idx,
    const ov::Output<ov::Node>& bias_up,
    const ov::Output<ov::Node>& bias_down,
    bool has_zp,
    float scale_factor = -1.0f) {
    auto w_up = ov::op::v0::Constant::create(
        element::u4, Shape{kNumExperts, kInterSize, kNumGroups, kGroupSize}, {1});
    auto scale_up = ov::op::v0::Constant::create(
        element::f16, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0.01f});
    auto w_down = ov::op::v0::Constant::create(
        element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, kGroupSize}, {1});
    auto scale_down = ov::op::v0::Constant::create(
        element::f16, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0.01f});

    ov::OutputVector args;
    args.push_back(hidden_states);
    args.push_back(routing_weights);
    args.push_back(topk_idx);
    args.push_back(w_up);
    args.push_back(scale_up);
    if (has_zp) {
        auto zp_up = ov::op::v0::Constant::create(
            element::u4, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0});
        args.push_back(zp_up);
    }
    args.push_back(bias_up);
    args.push_back(w_down);
    args.push_back(scale_down);
    if (has_zp) {
        auto zp_down = ov::op::v0::Constant::create(
            element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0});
        args.push_back(zp_down);
    }
    args.push_back(bias_down);

    ov::intel_gpu::op::MOECompressed::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.hidden_size = kHiddenSize;
    config.inter_size = kInterSize;
    config.num_expert = kNumExperts;
    config.top_k = kTopK;
    config.group_size = kGroupSize;
    config.has_zp = has_zp;
    config.out_type = ov::element::dynamic;
    config.scale_factor = scale_factor;
    return std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);
}

}  // namespace

// Verifies that ScaleDownMOECompressed:
//   - inserts Multiply(1/s) on hidden_states, bias_up, and bias_down
//   - inserts Multiply(s) on the MOECompressed output
//   - records s on the op via set_scale_factor
// for the has_zp=false (9-input) layout.
TEST_F(TransformationTestsF, ScaleDownMOECompressedTest_NoZp) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto moe = make_moe_compressed_gemm2(hidden_states, routing_weights, topk_idx,
                                             bias_up, bias_down, /*has_zp=*/false);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ScaleDownMOECompressed>(scale_factor, ov::element::f16);
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto scale_down_const = ov::op::v0::Constant::create(element::f16, Shape{}, {1.f / scale_factor});
        auto hidden_scaled = std::make_shared<ov::op::v1::Multiply>(hidden_states, scale_down_const);
        auto bias_up_scaled = std::make_shared<ov::op::v1::Multiply>(bias_up, scale_down_const);
        auto bias_down_scaled = std::make_shared<ov::op::v1::Multiply>(bias_down, scale_down_const);

        auto moe = make_moe_compressed_gemm2(hidden_scaled, routing_weights, topk_idx,
                                             bias_up_scaled, bias_down_scaled,
                                             /*has_zp=*/false, /*scale_factor=*/scale_factor);

        auto scale_up_const = ov::op::v0::Constant::create(element::f16, Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(moe, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// Same as above but with has_zp=true (11-input) layout — verifies that the
// bias indices shift by one to account for the zp inputs.
TEST_F(TransformationTestsF, ScaleDownMOECompressedTest_WithZp) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto moe = make_moe_compressed_gemm2(hidden_states, routing_weights, topk_idx,
                                             bias_up, bias_down, /*has_zp=*/true);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ScaleDownMOECompressed>(scale_factor, ov::element::f16);
    }
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        auto scale_down_const = ov::op::v0::Constant::create(element::f16, Shape{}, {1.f / scale_factor});
        auto hidden_scaled = std::make_shared<ov::op::v1::Multiply>(hidden_states, scale_down_const);
        auto bias_up_scaled = std::make_shared<ov::op::v1::Multiply>(bias_up, scale_down_const);
        auto bias_down_scaled = std::make_shared<ov::op::v1::Multiply>(bias_down, scale_down_const);

        auto moe = make_moe_compressed_gemm2(hidden_scaled, routing_weights, topk_idx,
                                             bias_up_scaled, bias_down_scaled,
                                             /*has_zp=*/true, /*scale_factor=*/scale_factor);

        auto scale_up_const = ov::op::v0::Constant::create(element::f16, Shape{}, {scale_factor});
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(moe, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result},
                                                ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// Idempotence: a MOECompressed whose scale_factor is already > 0 must be
// left untouched by the pass.
TEST_F(TransformationTestsF, ScaleDownMOECompressedTest_AlreadyScaled) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});
        auto bias_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kInterSize}, {0.5f});
        auto bias_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, 1, kHiddenSize}, {0.25f});

        // Pre-populate scale_factor to simulate an already-scaled op.
        auto moe = make_moe_compressed_gemm2(hidden_states, routing_weights, topk_idx,
                                             bias_up, bias_down, /*has_zp=*/false,
                                             /*scale_factor=*/scale_factor);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ScaleDownMOECompressed>(scale_factor, ov::element::f16);
    }
    // model_ref left unset -> TransformationTestsF compares against a clone of the input model.
}

// GEMM3_SWIGLU experts have no biases and must not be touched by this pass.
TEST_F(TransformationTestsF, ScaleDownMOECompressedTest_Gemm3SwigluSkipped) {
    const float scale_factor = 8.f;
    {
        auto hidden_states = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kHiddenSize});
        auto routing_weights = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{kTokens, kTopK});
        auto topk_idx = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{kTokens, kTopK});

        auto w_gate = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, kGroupSize}, {1});
        auto scale_gate = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0.01f});
        auto zp_gate = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0});
        auto w_up = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, kGroupSize}, {1});
        auto scale_up = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0.01f});
        auto zp_up = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kInterSize, kNumGroups, 1}, {0});
        auto w_down = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, kGroupSize}, {1});
        auto scale_down = ov::op::v0::Constant::create(element::f16, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0.01f});
        auto zp_down = ov::op::v0::Constant::create(element::u4, Shape{kNumExperts, kHiddenSize, kNumGroups, 1}, {0});

        ov::intel_gpu::op::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.hidden_size = kHiddenSize;
        config.inter_size = kInterSize;
        config.num_expert = kNumExperts;
        config.top_k = kTopK;
        config.group_size = kGroupSize;
        config.out_type = ov::element::dynamic;
        auto moe = std::make_shared<ov::intel_gpu::op::MOECompressed>(
            ov::OutputVector{hidden_states, routing_weights, topk_idx,
                             w_gate, scale_gate, zp_gate,
                             w_up, scale_up, zp_up,
                             w_down, scale_down, zp_down}, config);
        auto convert = std::make_shared<ov::op::v0::Convert>(moe, element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{hidden_states, routing_weights, topk_idx});
        manager.register_pass<ScaleDownMOECompressed>(scale_factor, ov::element::f16);
    }
    // model_ref left unset -> TransformationTestsF compares against a clone of the input model.
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
