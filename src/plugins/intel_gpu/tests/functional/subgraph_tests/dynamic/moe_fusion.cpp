// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/moe_builders.hpp"

namespace {
using ov::test::InputShape;
using ov::test::MoERoutingType;

// ─── Shared test data ─────────────────────────────────────────────────────────

// Default group size for weight decompression on GPU.
static constexpr int GROUP_SIZE = 128;

const std::vector<ov::test::MoePatternParams> moe_params_smoke = {
    {
        {ov::PartialShape{-1, -1, 256}, {{2, 15, 256}, {2, 1, 256}, {3, 8, 256}}},  // dynamic seq_len, hidden=256
        4,    // topk
        8,    // number_of_experts
        512,  // intermediate_size
    },
    {
        {ov::PartialShape{-1, -1, 128}, {{1, 32, 128}, {1, 1, 128}, {1, 16, 128}}},  // different seq length, hidden=128
        2,    // topk
        4,    // number_of_experts
        256,  // intermediate_size
    },
};

const std::vector<ov::element::Type> weights_precisions = {
    ov::element::u8,
    ov::element::i8,
    ov::element::u4,
    ov::element::i4,
};

// 3GeMM supports two routing variants; 2GeMM builder uses Softmax only.
const std::vector<MoERoutingType> routing_types = {MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS};

// ─── Shared name helper ───────────────────────────────────────────────────────

static std::string formatMoeBaseParams(const ov::test::MoePatternParams& p) {
    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({p.data_shape.first}) << "_";
    result << "TS=";
    for (const auto& s : p.data_shape.second)
        result << ov::test::utils::vec2str(s) << ",";
    result << "topk=" << p.topk << "_";
    result << "experts=" << p.number_of_experts << "_";
    result << "inter=" << p.intermediate_size;
    return result.str();
}

// ─── 2GeMM parameter types ────────────────────────────────────────────────────

// Non-compressed 2GeMM only varies over shape params.
using MoE2GemmParams = ov::test::MoePatternParams;

using MoE2GemmCompressedParams = std::tuple<ov::test::MoePatternParams,
                                             ov::element::Type,                          // weights_precision
                                             ov::element::Type,                          // decompression_precision
                                             ov::element::Type,                          // scale_precision
                                             ov::test::utils::DecompressionType,         // multiply type
                                             ov::test::utils::DecompressionType,         // subtract type
                                             bool,                                       // reshape_on_decompression
                                             int>;                                       // group_size

// ─── 3GeMM parameter types ────────────────────────────────────────────────────

using MoE3GemmParams = std::tuple<ov::test::MoePatternParams, MoERoutingType>;

using MoE3GemmCompressedParams = std::tuple<ov::test::MoePatternParams,
                                             MoERoutingType,
                                             ov::element::Type,                          // weights_precision
                                             ov::element::Type,                          // decompression_precision
                                             ov::element::Type,                          // scale_precision
                                             ov::test::utils::DecompressionType,         // multiply type
                                             ov::test::utils::DecompressionType,         // subtract type
                                             bool,                                       // reshape_on_decompression
                                             int>;                                       // group_size

// ─── 2GeMM: Non-compressed ───────────────────────────────────────────────────

class MoE2GemmFusionTest : public testing::WithParamInterface<MoE2GemmParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE2GemmParams>& info) {
        return formatMoeBaseParams(info.param);
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = outType = ov::element::f32;

        const auto& moe_params = GetParam();
        init_input_shapes({moe_params.data_shape});

        function = ov::test::initMoE2GeMMSubgraph(
            moe_params,
            ov::element::f32,  // data_precision
            ov::element::f32); // weights_precision – plain f32 weights
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor(
                           params[0]->get_element_type(),
                           target_input_static_shapes[0],
                           ov::test::utils::InputGenerateData(0.125f, 2, 8, 1234))});
    }
};

// ─── 2GeMM: Compressed-weights ───────────────────────────────────────────────

class MoE2GemmCompressedFusionTest : public testing::WithParamInterface<MoE2GemmCompressedParams>,
                                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE2GemmCompressedParams>& info) {
        const auto& [moe_params, wp, dp, sp, dm, ds, rd, gs] = info.param;
        std::ostringstream result;
        result << formatMoeBaseParams(moe_params) << "_";
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
        inType = outType = ov::element::f32;

        const auto& [moe_params, wp, dp, sp, dm, ds, rd, gs] = GetParam();
        init_input_shapes({moe_params.data_shape});

        function = ov::test::initMoE2GeMMSubgraph(
            moe_params,
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

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor(
                           params[0]->get_element_type(),
                           target_input_static_shapes[0],
                           ov::test::utils::InputGenerateData(0.125f, 2, 8, 1234))});
    }
};

// ─── 3GeMM: Non-compressed ───────────────────────────────────────────────────

class MoE3GemmFusionTest : public testing::WithParamInterface<MoE3GemmParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE3GemmParams>& info) {
        const auto& [moe_params, routing_type] = info.param;
        std::ostringstream result;
        result << formatMoeBaseParams(moe_params) << "_";
        result << "routing=" << (routing_type == MoERoutingType::SIGMOID_BIAS ? "SigmoidBias" : "Softmax");
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        // f16 parameter makes the GPU plugin select inference_precision=f16,
        // which is required by the MOE3GemmFusedCompressed kernel.
        inType = outType = ov::element::f16;

        const auto& [moe_params, routing_type] = GetParam();
        init_input_shapes({moe_params.data_shape});

        function = ov::test::initMoE3GeMMSubgraph(
            moe_params,
            ov::element::f32,  // data_precision – computation in f32
            ov::element::f32,  // weights_precision – plain f32 weights (no decompression)
            false,             // use_weight_decompression
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            routing_type,
            ov::element::f16); // input_precision – f16 Parameter
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor(
                           params[0]->get_element_type(),
                           target_input_static_shapes[0],
                           ov::test::utils::InputGenerateData(0.125f, 2, 8, 1234))});
    }
};

// ─── 3GeMM: Compressed-weights ───────────────────────────────────────────────

class MoE3GemmCompressedFusionTest : public testing::WithParamInterface<MoE3GemmCompressedParams>,
                                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoE3GemmCompressedParams>& info) {
        const auto& [moe_params, routing_type, wp, dp, sp, dm, ds, rd, gs] = info.param;
        std::ostringstream result;
        result << formatMoeBaseParams(moe_params) << "_";
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
        inType = outType = ov::element::f16;

        const auto& [moe_params, routing_type, wp, dp, sp, dm, ds, rd, gs] = GetParam();
        init_input_shapes({moe_params.data_shape});

        function = ov::test::initMoE3GeMMSubgraph(
            moe_params,
            ov::element::f32,  // data_precision
            wp,                // weights_precision
            true,              // use_weight_decompression
            dp,                // decompression_precision
            sp,                // scale_precision
            dm,                // decompression_multiply_type
            ds,                // decompression_subtract_type
            rd,                // reshape_on_decompression
            gs,                // group_size
            routing_type,
            ov::element::f16); // input_precision – f16 Parameter
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        inputs.insert({params[0],
                       ov::test::utils::create_and_fill_tensor(
                           params[0]->get_element_type(),
                           target_input_static_shapes[0],
                           ov::test::utils::InputGenerateData(0.125f, 2, 8, 1234))});
    }
};

// ─── Test bodies ──────────────────────────────────────────────────────────────

TEST_P(MoE2GemmFusionTest, Inference) {
    run();
}

TEST_P(MoE2GemmCompressedFusionTest, Inference) {
    run();
}

TEST_P(MoE3GemmFusionTest, Inference) {
    run();
}

TEST_P(MoE3GemmCompressedFusionTest, Inference) {
    run();
}

// ─── Instantiation ────────────────────────────────────────────────────────────

// 2GeMM – plain f32 weights
INSTANTIATE_TEST_SUITE_P(smoke_MoE2GemmFusion,
                         MoE2GemmFusionTest,
                         ::testing::ValuesIn(moe_params_smoke),
                         MoE2GemmFusionTest::getTestCaseName);

// 2GeMM – compressed weights
INSTANTIATE_TEST_SUITE_P(smoke_MoE2GemmCompressedFusion,
                         MoE2GemmCompressedFusionTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(moe_params_smoke),
                             ::testing::ValuesIn(weights_precisions),
                             ::testing::Values(ov::element::f16),  // decompression_precision
                             ::testing::Values(ov::element::f16),  // scale_precision
                             ::testing::Values(ov::test::utils::DecompressionType::full),
                             ::testing::Values(ov::test::utils::DecompressionType::full),
                             ::testing::Values(true),              // reshape_on_decompression
                             ::testing::Values(static_cast<int>(GROUP_SIZE))),
                         MoE2GemmCompressedFusionTest::getTestCaseName);

// 3GeMM – plain f32 weights, both routing types
INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmFusion,
                         MoE3GemmFusionTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(moe_params_smoke),
                             ::testing::ValuesIn(routing_types)),
                         MoE3GemmFusionTest::getTestCaseName);

// 3GeMM – compressed weights, both routing types
INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmCompressedFusion,
                         MoE3GemmCompressedFusionTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(moe_params_smoke),
                             ::testing::ValuesIn(routing_types),
                             ::testing::ValuesIn(weights_precisions),
                             ::testing::Values(ov::element::f16),  // decompression_precision
                             ::testing::Values(ov::element::f16),  // scale_precision
                             ::testing::Values(ov::test::utils::DecompressionType::full),
                             ::testing::Values(ov::test::utils::DecompressionType::full),
                             ::testing::Values(true),              // reshape_on_decompression
                             ::testing::Values(static_cast<int>(GROUP_SIZE))),
                         MoE3GemmCompressedFusionTest::getTestCaseName);

}  // namespace
