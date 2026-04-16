// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// End-to-end tests for GPU MOE transformation pipeline.
//
// Full pipeline (steps 1+2+3+4): IR → MOE3GemmFusedCompressed
//   IR (Tile+MatMuls+ScatterElementsUpdate) → ConvertTiledMoeBlockTo3GatherMatmuls
//   → ConvertGatherMatmulToGatherMatmulCompressed
//   → Convert3GatherMatmulMoeBlockToMoeOp (Or-patterns produce MOECompressed)
//   → FuseMOE3GemmCompressed → MOE3GemmFusedCompressed + Convert(f32)

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "plugin/transformations/fuse_moe_3gemm_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"
#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// ============================================================================
// Shared constants
// ============================================================================

static constexpr size_t T2_HIDDEN = 128;
static constexpr size_t T2_INTER = 64;
static constexpr size_t T2_EXPERTS = 4;
static constexpr size_t T2_TOPK = 2;
static constexpr size_t T2_GROUP_SIZE = 32;
static constexpr size_t T2_GROUP_COUNT_GATE = T2_HIDDEN / T2_GROUP_SIZE;  // 4
static constexpr size_t T2_GROUP_COUNT_DOWN = T2_INTER / T2_GROUP_SIZE;   // 2

// ============================================================================
// Full IR → MOE3GemmFusedCompressed pipeline (parameterized by routing type)
// ============================================================================

class MOE_E2E_PipelineTest : public TransformationTestsF,
                             public ::testing::WithParamInterface<ov::test::MoERoutingType> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<ov::test::MoERoutingType>& info) {
        return info.param == ov::test::MoERoutingType::SOFTMAX ? "Softmax" : "SigmoidBias";
    }
};

TEST_P(MOE_E2E_PipelineTest, IRToMOE3GemmFusedCompressed) {
    const auto routing_type = GetParam();
    disable_rt_info_check();
    {
        // ---- Build input model using shared MOE builder ----
        ov::test::MoePatternParams moe_params;
        moe_params.data_shape = ov::PartialShape{-1, -1, static_cast<int64_t>(T2_HIDDEN)};
        moe_params.topk = T2_TOPK;
        moe_params.number_of_experts = T2_EXPERTS;
        moe_params.intermediate_size = T2_INTER;

        model = ov::test::initMoE3GeMMSubgraph(
            moe_params,
            ov::element::f32,                                      // data_precision
            ov::element::u4,                                       // weights_precision
            true,                                                  // use_weight_decompression
            ov::element::f16,                                      // decompression_precision
            ov::element::f16,                                      // scale_precision
            ov::test::utils::DecompressionType::full,              // decompression_multiply_type
            ov::test::utils::DecompressionType::full,              // decompression_subtract_type
            true,                                                  // reshape_on_decompression
            static_cast<int>(T2_GROUP_SIZE),                       // decompression_group_size
            routing_type);

        // Register pipeline passes
        manager.register_pass<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
        manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f16},
            std::vector<ov::element::Type>{ov::element::u4, ov::element::i4, ov::element::i8, ov::element::u8});
        manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(/*has_batch_dim=*/true);
        manager.register_pass<FuseMOE3GemmCompressed>();
    }
    {
        // ---- Reference model: MOE3GemmFusedCompressed → Convert(f32) ----
        // Graph topology comparison only (CONST_VALUES not enabled by default),
        // so constant values don't need to match — only shapes and types.
        auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, static_cast<int64_t>(T2_HIDDEN)});

        auto experts_reshape =
            std::make_shared<op::v1::Reshape>(param,
                                              op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(T2_HIDDEN)}),
                                              false);
        auto routers = op::v0::Constant::create(element::f32, Shape{T2_EXPERTS, T2_HIDDEN}, {0});
        auto routing_weights = std::make_shared<op::v0::MatMul>(experts_reshape, routers, false, true);

        // Weights after process_compressed_weights: groups combined to 3D [experts, N, K]
        auto wei_gate = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_HIDDEN}, {0});
        auto wei_up = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_HIDDEN}, {0});
        auto wei_down = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_HIDDEN, T2_INTER}, {0});

        // Scales: trailing 1 combined → 3D [experts, N, groups]
        auto scale_gate = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto scale_up = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto scale_down = op::v0::Constant::create(element::f16, Shape{T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN}, {0});

        // Zero-points: trailing 1 combined → 3D
        auto zp_gate = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto zp_up = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_INTER, T2_GROUP_COUNT_GATE}, {0});
        auto zp_down = op::v0::Constant::create(element::u4, Shape{T2_EXPERTS, T2_HIDDEN, T2_GROUP_COUNT_DOWN}, {0});

        ov::op::internal::MOECompressed::Config config;
        config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        config.hidden_size = T2_HIDDEN;
        config.inter_size = T2_INTER;
        config.num_expert = T2_EXPERTS;
        config.group_size = T2_GROUP_SIZE;
        config.top_k = T2_TOPK;
        config.out_type = element::f16;
        config.has_batch_dim = true;

        OutputVector args{experts_reshape, routing_weights, wei_gate, scale_gate, zp_gate, wei_up, scale_up, zp_up, wei_down, scale_down, zp_down};

        if (routing_type == ov::test::MoERoutingType::SIGMOID_BIAS) {
            config.routing_type = ov::op::internal::MOECompressed::RoutingType::SIGMOID_BIAS;
            auto routing_bias = op::v0::Constant::create(element::f32, Shape{1, T2_EXPERTS}, {0});
            args.push_back(routing_bias);
            auto routing_eps = op::v0::Constant::create(element::f32, Shape{1, 1}, {0});
            args.push_back(routing_eps);
        }

        std::shared_ptr<Node> moe_fused = std::make_shared<ov::intel_gpu::op::MOE3GemmFusedCompressed>(args, config);

        auto hidden_state_shape = std::make_shared<op::v3::ShapeOf>(param);
        moe_fused = std::make_shared<op::v1::Reshape>(moe_fused, hidden_state_shape, false);

        auto convert_back = std::make_shared<op::v0::Convert>(moe_fused, element::f32);

        model_ref = std::make_shared<Model>(convert_back, ParameterVector{param});
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         MOE_E2E_PipelineTest,
                         ::testing::Values(ov::test::MoERoutingType::SOFTMAX, ov::test::MoERoutingType::SIGMOID_BIAS),
                         MOE_E2E_PipelineTest::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
