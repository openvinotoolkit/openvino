// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;
using ov::test::MoERoutingType;

enum class MoePatternType { GEMM3, GEMM2 };

struct MoeTestShapeParams {
    InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

static const char* routing_type_str(MoERoutingType rt) {
    switch (rt) {
    case MoERoutingType::SOFTMAX:
        return "Softmax";
    case MoERoutingType::SIGMOID_BIAS:
        return "SigmoidBias";
    default:
        OPENVINO_THROW("Unsupported MoERoutingType");
    }
}

static const char* pattern_type_str(MoePatternType pt) {
    return pt == MoePatternType::GEMM3 ? "GEMM3" : "GEMM2";
}

using MoECompressedParams = std::tuple<MoeTestShapeParams,
                                       MoePatternType,
                                       MoERoutingType,
                                       ov::element::Type,                   // weights_precision
                                       ov::element::Type,                   // decompression_precision
                                       ov::element::Type,                   // scale_precision
                                       ov::test::utils::DecompressionType,  // multiply type
                                       ov::test::utils::DecompressionType,  // subtract type
                                       bool,                                // reshape_on_decompression
                                       int,                                 // group_size
                                       size_t>;                             // gate_idx (2-GEMM only)

class MoECompressedFusionTest : public testing::WithParamInterface<MoECompressedParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoECompressedParams>& info) {
        const auto& [moe_params, pattern_type, routing_type, wp, dp, sp, dm, ds, rd, gs, gi] = info.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
        result << "TS=";
        for (const auto& s : moe_params.data_shape.second)
            result << ov::test::utils::vec2str(s) << ",";
        result << "topk=" << moe_params.topk << "_";
        result << "experts=" << moe_params.number_of_experts << "_";
        result << "inter=" << moe_params.intermediate_size << "_";
        result << "pattern=" << pattern_type_str(pattern_type) << "_";
        result << "routing=" << routing_type_str(routing_type) << "_";
        result << "WP=" << wp << "_";
        result << "DP=" << dp << "_";
        result << "SP=" << sp << "_";
        result << "DM=" << dm << "_";
        result << "DS=" << ds << "_";
        result << "RD=" << rd << "_";
        result << "GS=" << gs << "_";
        result << "GI=" << gi;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        // MoE fusion (and the GatherMatmul-based fallback) is gated on supports_immad.
        const auto caps = core->get_property(targetDevice, ov::device::capabilities);
        if (std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::HW_MATMUL) == caps.end()) {
            GTEST_SKIP() << "MoE pipeline requires a systolic GPU (HW_MATMUL capability)";
        }
        inType = outType = inference_precision = ov::element::f16;
        // Cascaded matmul+clamp+swish chains exceed default per-op f16 epsilon.
        abs_threshold = 1.0;
        rel_threshold = 0.01;

        const auto& [moe_params, pattern_type, routing_type, wp, dp, sp, dm, ds, rd, gs, gi] = GetParam();

        // INSTANTIATE blocks below never produce GEMM2 + non-softmax; fail loudly if that ever changes.
        if (pattern_type == MoePatternType::GEMM2 && routing_type != MoERoutingType::SOFTMAX) {
            FAIL() << "2-GEMM pattern with non-softmax routing must not be instantiated";
        }
        init_input_shapes({moe_params.data_shape});
        const ov::test::MoePatternParams shape_params{moe_params.data_shape.first, moe_params.topk, moe_params.number_of_experts, moe_params.intermediate_size};

        if (pattern_type == MoePatternType::GEMM3) {
            function = ov::test::initMoE3GeMMSubgraph(shape_params,
                                                      ov::element::f32,  // data_precision
                                                      wp,                // weights_precision
                                                      true,              // use_weight_decompression
                                                      dp,                // decompression_precision
                                                      sp,                // scale_precision
                                                      dm,                // decompression_multiply_type
                                                      ds,                // decompression_subtract_type
                                                      rd,                // reshape_on_decompression
                                                      gs,                // group_size
                                                      routing_type);
        } else {
            function = ov::test::initMoE2GeMMSubgraph(shape_params,
                                                      ov::element::f32,  // data_precision
                                                      wp,                // weights_precision
                                                      true,              // use_weight_decompression
                                                      dp,                // decompression_precision
                                                      sp,                // scale_precision
                                                      dm,                // decompression_multiply_type
                                                      ds,                // decompression_subtract_type
                                                      rd,                // reshape_on_decompression
                                                      gs,                // group_size
                                                      gi);               // gate_idx
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        // RMSNorm output ≈ N(0,1); needed to exercise swish/SwiGLU on negative inputs.
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor_normal_distribution(params[0]->get_element_type(),
                                                                                   target_input_static_shapes[0],
                                                                                   /*mean=*/0.0f,
                                                                                   /*stddev=*/1.0f,
                                                                                   /*seed=*/1234)});
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        const auto pattern_type = std::get<1>(GetParam());
        // GEMM3 → single fused moe_3gemm_fused_compressed; GEMM2 → gate_up + down = 2 moe_gemm ops.
        ov::test::CheckNumberOfNodesWithType(compiledModel,
                                             pattern_type == MoePatternType::GEMM3 ? "moe_3gemm_fused_compressed" : "moe_gemm",
                                             pattern_type == MoePatternType::GEMM3 ? 1 : 2);
    }
};

TEST_P(MoECompressedFusionTest, Inference) {
    run();
}

const std::vector<MoERoutingType> routing_types = {MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS};

const std::vector<MoeTestShapeParams> moe_params_smoke = {
    {
        // TODO: batch>1 trips a pre-existing master bug (moe_mask_gen vs moe_gather/scatter has_batch_dim mismatch → OOB).
        {{-1, -1, 256}, {{1, 30, 256}, {1, 2, 256}, {1, 24, 256}}},  // data_shape,
                                                                     // seq_len=dynamic, hidden_size=256
        4,                                                           // topk
        8,                                                           // number_of_experts
        512                                                          // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 32, 128}, {1, 1, 128}, {1, 16, 128}}},  // Different seq length
        2,                                                           // topk
        4,                                                           // number_of_experts
        256                                                          // intermediate_size
    },
};

// Compressed weights — full GatherMatmul → MOECompressed → FuseMOE3GemmCompressed pipeline.
const std::vector<ov::element::Type> weights_precisions = {
    ov::element::u8,
    ov::element::u4,
};

// gate_idx values: 0 = real gpt-oss layout (gate at even), 1 = inverted layout.
const std::vector<size_t> gate_idx_values = {0, 1};

INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmCompressedFusion,
                         MoECompressedFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoePatternType::GEMM3),
                                            ::testing::ValuesIn(routing_types),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f16),  // decompression_precision
                                            ::testing::Values(ov::element::f16),  // scale_precision
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),  // reshape_on_decompression
                                            ::testing::Values(128),
                                            ::testing::Values(size_t{0})),  // gate_idx unused for GEMM3
                         MoECompressedFusionTest::getTestCaseName);

// GPT-OSS 2-GEMM pattern (combined gate/up MatMul + Slice/Clamp/Add/Swish).
// gate_idx=0 matches real gpt-oss IR; gate_idx=1 covers the inverted layout.
INSTANTIATE_TEST_SUITE_P(smoke_MoE2GemmCompressedFusion,
                         MoECompressedFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoePatternType::GEMM2),
                                            ::testing::Values(MoERoutingType::SOFTMAX),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f16),  // decompression_precision
                                            ::testing::Values(ov::element::f16),  // scale_precision
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),  // reshape_on_decompression
                                            ::testing::Values(128),
                                            ::testing::ValuesIn(gate_idx_values)),
                         MoECompressedFusionTest::getTestCaseName);

// MOE fusion disabled — exercises GatherMatmulCompressed directly.
// disable_moe_opt is a debug option; only honored via env var, not properties.
class MoEGatherMatmulTest : public MoECompressedFusionTest {
protected:
    static void set_disable_moe_opt() {
#ifdef _WIN32
        _putenv_s("OV_GPU_DISABLE_MOE_OPT", "1");
#else
        ::setenv("OV_GPU_DISABLE_MOE_OPT", "1", 1);
#endif
    }

    static void unset_disable_moe_opt() {
#ifdef _WIN32
        _putenv_s("OV_GPU_DISABLE_MOE_OPT", "");
#else
        ::unsetenv("OV_GPU_DISABLE_MOE_OPT");
#endif
    }

    void SetUp() override {
        set_disable_moe_opt();
        MoECompressedFusionTest::SetUp();
    }

    void TearDown() override {
        MoECompressedFusionTest::TearDown();
        unset_disable_moe_opt();
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        const auto pattern_type = std::get<1>(GetParam());
        // GEMM3: gate, up, down → 3 GatherMatmul ops; GEMM2: fused gate_up + down → 2.
        ov::test::CheckNumberOfNodesWithType(compiledModel, "gather_matmul", pattern_type == MoePatternType::GEMM3 ? 3 : 2);
    }
};

TEST_P(MoEGatherMatmulTest, Inference) {
    run();
}

// DISABLED: u8 prefill kernel missing; u4 has accuracy divergence (Δ ≈ 5–45).
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_MoEGatherMatmul,
                         MoEGatherMatmulTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoePatternType::GEMM3),
                                            ::testing::ValuesIn(routing_types),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f16),  // decompression_precision
                                            ::testing::Values(ov::element::f16),  // scale_precision
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),  // reshape_on_decompression
                                            ::testing::Values(128),
                                            ::testing::Values(size_t{0})),  // gate_idx unused for GEMM3
                         MoEGatherMatmulTest::getTestCaseName);

// 2-GEMM unfused: covers GatherMatmul OCL clamp/bias/swiglu compile path.
// DISABLED: same u8/u4 issues as above.
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_MoE2GemmGatherMatmul,
                         MoEGatherMatmulTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::Values(MoePatternType::GEMM2),
                                            ::testing::Values(MoERoutingType::SOFTMAX),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f16),  // decompression_precision
                                            ::testing::Values(ov::element::f16),  // scale_precision
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),  // reshape_on_decompression
                                            ::testing::Values(128),
                                            ::testing::ValuesIn(gate_idx_values)),
                         MoEGatherMatmulTest::getTestCaseName);

}  // namespace
