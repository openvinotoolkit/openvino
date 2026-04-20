// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/exec_model_info.hpp"
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
    case MoERoutingType::SOFTMAX:      return "Softmax";
    case MoERoutingType::SIGMOID_BIAS: return "SigmoidBias";
    default: OPENVINO_THROW("Unsupported MoERoutingType");
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
                                       int>;                                // group_size

class MoECompressedFusionTest : public testing::WithParamInterface<MoECompressedParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoECompressedParams>& info) {
        const auto& [moe_params, pattern_type, routing_type, wp, dp, sp, dm, ds, rd, gs] = info.param;
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
        result << "GS=" << gs;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = outType = inference_precision = ov::element::f16;

        const auto& [moe_params, pattern_type, routing_type, wp, dp, sp, dm, ds, rd, gs] = GetParam();
        // 2-GEMM builder only supports softmax routing.
        if (pattern_type == MoePatternType::GEMM2 && routing_type != MoERoutingType::SOFTMAX) {
            GTEST_SKIP() << "2-GEMM pattern only supports softmax routing";
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
                                                      gs);               // group_size
        }
    }

    void assert_runtime_model_has_op(const std::string& expected_layer_type) {
        auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_TRUE(runtime_model != nullptr) << "Runtime model should not be null";
        bool found = false;
        for (const auto& op : runtime_model->get_ordered_ops()) {
            auto layer_type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == expected_layer_type) {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << expected_layer_type << " op is not found in runtime model";
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor(params[0]->get_element_type(),
                                                               target_input_static_shapes[0],
                                                               ov::test::utils::InputGenerateData(0.125f, 2, 8, 1234))});
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        const auto pattern_type = std::get<1>(GetParam());
        // 3-GEMM lowers through FuseMOE3GemmCompressed; 2-GEMM stays as
        // MOECompressed and is expanded to moe_gather/moe_gemm/swiglu/moe_scatter_reduction.
        assert_runtime_model_has_op(pattern_type == MoePatternType::GEMM3 ? "moe_3gemm_fused_compressed"
                                                                          : "moe_gemm");
    }
};

TEST_P(MoECompressedFusionTest, Inference) {
    run();
}

const std::vector<MoERoutingType> routing_types = {MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS};

const std::vector<MoeTestShapeParams> moe_params_smoke = {
    {
        {{-1, -1, 256}, {{2, 15, 256}, {2, 1, 256}, {3, 8, 256}}},  // data_shape,
                                                                    // seq_len=dynamic, hidden_size=256
        4,                                                          // topk
        8,                                                          // number_of_experts
        512                                                         // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 32, 128}, {1, 1, 128}, {1, 16, 128}}},  // Different seq length
        2,                                                           // topk
        4,                                                           // number_of_experts
        256                                                          // intermediate_size
    },
};

// Compressed weights – covers the full GatherMatmul → MOECompressed →
// FuseMOE3GemmCompressed pipeline that runs in production.
const std::vector<ov::element::Type> weights_precisions = {
    ov::element::u8,
    ov::element::u4,
};

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
                                            ::testing::Values(128)),
                         MoECompressedFusionTest::getTestCaseName);

// GPT-OSS-style 2-GEMM pattern: combined gate/up MatMul with Slice/Clamp/Add/Swish.
// Exercises Convert2GatherMatmulMoeBlockToMoeOp → MOECompressed GEMM2_BIAS_SWIGLU_CLAMP path.
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
                                            ::testing::Values(128)),
                         MoECompressedFusionTest::getTestCaseName);

// Same MOE IR but with MOE fusion disabled (GPU_DISABLE_MOE_OPT=true).
// The model stays at GatherMatmulCompressed stage, validating GatherMatmul
// numerical correctness against CPU reference on the original untransformed model.
class MoEGatherMatmulTest : public MoECompressedFusionTest {
protected:
    void SetUp() override {
        MoECompressedFusionTest::SetUp();
        configuration.insert(ov::intel_gpu::disable_moe_opt(true));
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        assert_runtime_model_has_op("gather_matmul");
    }
};

TEST_P(MoEGatherMatmulTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MoEGatherMatmul,
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
                                            ::testing::Values(128)),
                         MoEGatherMatmulTest::getTestCaseName);

}  // namespace
