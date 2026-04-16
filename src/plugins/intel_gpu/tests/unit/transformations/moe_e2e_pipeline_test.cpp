// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// End-to-end tests for GPU MOE transformation pipeline.
//
// Full pipeline: IR → MOE3GemmFusedCompressed
//   IR (Tile+MatMuls+ScatterElementsUpdate) → ConvertTiledMoeBlockTo3GatherMatmuls
//   → ConvertGatherMatmulToGatherMatmulCompressed
//   → Convert3GatherMatmulMoeBlockToMoeOp (Or-patterns produce MOECompressed)
//   → FuseMOE3GemmCompressed → MOE3GemmFusedCompressed

#include <gtest/gtest.h>

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "openvino/pass/manager.hpp"
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
    disable_result_friendly_names_check();

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
        // Step 1: IR → GatherMatmul
        manager.register_pass<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
        // Step 2: GatherMatmul → GatherMatmulCompressed
        manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8},
            std::vector<ov::element::Type>{ov::element::f16, ov::element::u4, ov::element::i4, ov::element::i8, ov::element::u8});
        // Step 3: GatherMatmul/GatherMatmulCompressed → MOE/MOECompressed
        manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(/*has_batch_dim=*/true);
        // Step 4: MOECompressed → MOE3GemmFusedCompressed
        manager.register_pass<FuseMOE3GemmCompressed>();
    }

    // Run the transformation pipeline (TransformationTestsF applies manager in TearDown).
    // Since the reference model depends on exact constant values produced by the helper's
    // decompression chain, we skip exact model comparison and instead verify the pipeline
    // produces the expected fused op.
    // model_ref is intentionally not set — TransformationTestsF will just run the passes
    // and verify no crash. We add explicit op-type checks below.
}

// After the parametrized test body, verify the output model topology
// by hooking into TearDown (TransformationTestsF applies the passes there).
// Since we can't easily hook TearDown, we use a second test that runs the passes manually.

class MOE_E2E_PipelineTopologyTest : public ::testing::TestWithParam<ov::test::MoERoutingType> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<ov::test::MoERoutingType>& info) {
        return info.param == ov::test::MoERoutingType::SOFTMAX ? "Softmax" : "SigmoidBias";
    }
};

TEST_P(MOE_E2E_PipelineTopologyTest, ProducesMOE3GemmFusedCompressed) {
    const auto routing_type = GetParam();

    ov::test::MoePatternParams moe_params;
    moe_params.data_shape = ov::PartialShape{-1, -1, static_cast<int64_t>(T2_HIDDEN)};
    moe_params.topk = T2_TOPK;
    moe_params.number_of_experts = T2_EXPERTS;
    moe_params.intermediate_size = T2_INTER;

    auto model = ov::test::initMoE3GeMMSubgraph(
        moe_params,
        ov::element::f32,
        ov::element::u4,
        true,
        ov::element::f16,
        ov::element::f16,
        ov::test::utils::DecompressionType::full,
        ov::test::utils::DecompressionType::full,
        true,
        static_cast<int>(T2_GROUP_SIZE),
        routing_type);

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
    manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f16, ov::element::i8, ov::element::u8},
        std::vector<ov::element::Type>{ov::element::f16, ov::element::u4, ov::element::i4, ov::element::i8, ov::element::u8});
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>(/*has_batch_dim=*/true);
    manager.register_pass<ov::intel_gpu::FuseMOE3GemmCompressed>();
    manager.run_passes(model);

    // Verify the pipeline produced the expected fused op
    bool found_fused_op = false;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::intel_gpu::op::MOE3GemmFusedCompressed>(op)) {
            found_fused_op = true;
            break;
        }
    }
    ASSERT_TRUE(found_fused_op)
        << "Pipeline should produce MOE3GemmFusedCompressed but no such op was found in the model";
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         MOE_E2E_PipelineTest,
                         ::testing::Values(ov::test::MoERoutingType::SOFTMAX, ov::test::MoERoutingType::SIGMOID_BIAS),
                         MOE_E2E_PipelineTest::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke,
                         MOE_E2E_PipelineTopologyTest,
                         ::testing::Values(ov::test::MoERoutingType::SOFTMAX, ov::test::MoERoutingType::SIGMOID_BIAS),
                         MOE_E2E_PipelineTopologyTest::get_test_case_name);

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
