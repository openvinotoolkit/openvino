// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/moe_builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;
using ov::test::MoERoutingType;

struct MoeTestShapeParams {
    InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

using MoE3GemmParams = std::tuple<MoeTestShapeParams, MoERoutingType>;
using MoE3GemmCompressedParams = std::tuple<MoeTestShapeParams,
                                            MoERoutingType,
                                            ov::element::Type,                   // weights_precision
                                            ov::element::Type,                   // decompression_precision
                                            ov::element::Type,                   // scale_precision
                                            ov::test::utils::DecompressionType,  // multiply type
                                            ov::test::utils::DecompressionType,  // subtract type
                                            bool,                                // reshape_on_decompression
                                            int>;                                // group_size

class MoE3GemmFusionTest : public testing::WithParamInterface<MoE3GemmParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE3GemmParams>& info) {
        const auto& [moe_params, routing_type] = info.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
        result << "TS=";
        for (const auto& s : moe_params.data_shape.second)
            result << ov::test::utils::vec2str(s) << ",";
        result << "topk=" << moe_params.topk << "_";
        result << "experts=" << moe_params.number_of_experts << "_";
        result << "inter=" << moe_params.intermediate_size << "_";
        result << "routing=" << (routing_type == MoERoutingType::SIGMOID_BIAS ? "SigmoidBias" : "Softmax");
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        const auto& [moe_params, routing_type] = GetParam();
        init_input_shapes({moe_params.data_shape});
        const ov::test::MoePatternParams shape_params{moe_params.data_shape.first, moe_params.topk, moe_params.number_of_experts, moe_params.intermediate_size};

        function = ov::test::initMoE3GeMMSubgraph(shape_params,
                                                  ov::element::f32,  // data_precision – f16 computation matches GPU inference_precision
                                                  ov::element::f16,  // weights_precision – plain f32 weights (no decompression)
                                                  false,             // use_weight_decompression
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  routing_type);
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
};

class MoE3GemmCompressedFusionTest : public testing::WithParamInterface<MoE3GemmCompressedParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE3GemmCompressedParams>& info) {
        const auto& [moe_params, routing_type, wp, dp, sp, dm, ds, rd, gs] = info.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
        result << "TS=";
        for (const auto& s : moe_params.data_shape.second)
            result << ov::test::utils::vec2str(s) << ",";
        result << "topk=" << moe_params.topk << "_";
        result << "experts=" << moe_params.number_of_experts << "_";
        result << "inter=" << moe_params.intermediate_size << "_";
        result << "routing=" << (routing_type == MoERoutingType::SIGMOID_BIAS ? "SigmoidBias" : "Softmax") << "_";
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

        const auto& [moe_params, routing_type, wp, dp, sp, dm, ds, rd, gs] = GetParam();
        init_input_shapes({moe_params.data_shape});
        const ov::test::MoePatternParams shape_params{moe_params.data_shape.first, moe_params.topk, moe_params.number_of_experts, moe_params.intermediate_size};

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
};

TEST_P(MoE3GemmFusionTest, Inference) {
    run();
}

TEST_P(MoE3GemmCompressedFusionTest, Inference) {
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

INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmFusion,
                         MoE3GemmFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke), ::testing::ValuesIn(routing_types)),
                         MoE3GemmFusionTest::getTestCaseName);

// Compressed weights – covers the full FuseVectorizedMOE3GEMM + ConvertMOEToMOECompressed +
// FuseMOE3GemmCompressed pipeline that runs in production.
const std::vector<ov::element::Type> weights_precisions = {
    ov::element::u8,
    ov::element::i8,
    ov::element::u4,
    ov::element::i4,
};

INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmCompressedFusion,
                         MoE3GemmCompressedFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(routing_types),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::Values(ov::element::f16),  // decompression_precision
                                            ::testing::Values(ov::element::f16),  // scale_precision
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(ov::test::utils::DecompressionType::full),
                                            ::testing::Values(true),  // reshape_on_decompression
                                            ::testing::Values(128)),
                         MoE3GemmCompressedFusionTest::getTestCaseName);

}  // namespace
