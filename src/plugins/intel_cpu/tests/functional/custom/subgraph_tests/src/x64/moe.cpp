// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/moe.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/scatter_elements_update.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {

struct MoeShapeParams {
    ov::test::InputShape data_shape;
    size_t hidden_size;
    size_t intermediate_size;
    size_t number_of_experts;
    size_t topk;
    size_t fusion_factor;
    // Weight decompression parameters
    ov::Shape gate_up_weights_shape;
    ov::Shape down_proj_weights_shape;
    int decompression_group_size;
};

typedef std::tuple<MoeShapeParams,
                   ov::test::ElementType,  // weights precision
                   ov::test::ElementType,  // decompression precision
                   ov::test::ElementType,  // scale precision
                   bool,                   // use weight decompression
                   DecompressionType,      // decompression multiply type
                   DecompressionType,      // decompression subtract type
                   bool,                   // reshape on decompression constants
                   ov::AnyMap,             // additional config
                   fusingSpecificParams>
    MoeTestParams;

class MoeSubgraphTest : public testing::WithParamInterface<MoeTestParams>,
                        virtual public SubgraphBaseTest,
                        public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj) {
        const auto& [shape_params,
                     weights_precision,
                     decompression_precision,
                     scale_precision,
                     use_weight_decompression,
                     decompression_multiply_type,
                     decompression_subtract_type,
                     reshape_on_decompression,
                     additional_config,
                     fusing_params] = obj.param;

        std::ostringstream result;
        result << "IS=" << shape_params.data_shape << "_";
        result << "HS=" << shape_params.hidden_size << "_";
        result << "IS=" << shape_params.intermediate_size << "_";
        result << "NE=" << shape_params.number_of_experts << "_";
        result << "TK=" << shape_params.topk << "_";
        result << "FF=" << shape_params.fusion_factor << "_";

        if (use_weight_decompression) {
            result << "WP=" << weights_precision << "_";
            result << "DP=" << decompression_precision << "_";
            result << "SP=" << scale_precision << "_";
            result << "DM=" << decompression_multiply_type << "_";
            result << "DS=" << decompression_subtract_type << "_";
            result << "RD=" << reshape_on_decompression << "_";
            result << "GS=" << shape_params.decompression_group_size << "_";
        }

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
        }
        result << ")";
        result << CpuTestWithFusing::getTestCaseName(fusing_params);

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> initMoeSubgraph(const MoeShapeParams& shape_params,
                                               const ov::element::Type data_precision,
                                               const ov::element::Type weights_precision,
                                               const ov::element::Type decompression_precision,
                                               const ov::element::Type scale_precision,
                                               const bool use_weight_decompression,
                                               const DecompressionType decompression_multiply_type,
                                               const DecompressionType decompression_subtract_type,
                                               const bool reshape_on_decompression) {
        // Use parameters from shape_params - static shapes only
        const size_t hidden_size = shape_params.hidden_size;
        const size_t intermediate_size = shape_params.intermediate_size;
        const size_t topk = shape_params.topk;
        const size_t number_of_experts = shape_params.number_of_experts;
        const size_t fusion_factor = shape_params.fusion_factor;
        const auto expert_alpha = 1.702f;
        const auto expert_beta = 7.0f;

        // Create input parameter with dynamic shape - batch and hidden_size are fixed, seq_len is dynamic
        const size_t batch = 2;                                      // Fixed batch size from reference
        const ov::Dimension seq_len_dim = ov::Dimension::dynamic();  // Dynamic sequence length
        auto input_shape = ov::PartialShape{batch, seq_len_dim, hidden_size};
        auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

        // Expert processing path - use -1 for dynamic reshape like in reference
        auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
            input,
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{2},
                                         std::vector<int64_t>{-1, hidden_size}),  // -1 flattens batch*seq_len
            false);

        auto tile = std::make_shared<ov::op::v0::Tile>(
            experts_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{number_of_experts, 1}));

        auto after_tile_reshape = std::make_shared<ov::op::v1::Reshape>(
            tile,
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{3},
                                         std::vector<int64_t>{number_of_experts, batch, hidden_size}),
            false);

        // Gate/Up projection
        auto gate_up_matmul = std::make_shared<ov::op::v0::MatMul>(
            after_tile_reshape,
            ov::test::utils::make_constant(
                ov::element::f32,
                ov::Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor}));

        auto gate_up_add = std::make_shared<ov::op::v1::Add>(
            gate_up_matmul,
            ov::test::utils::make_constant(ov::element::f32,
                                           ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor}));

        auto slice1 = std::make_shared<ov::op::v8::Slice>(
            gate_up_add,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, 0}),
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{3},
                                         std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 1, 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2}));
        auto clamp = std::make_shared<ov::op::v0::Clamp>(slice1, -expert_beta, expert_beta);
        auto add1 =
            std::make_shared<ov::op::v1::Add>(clamp, ov::test::utils::make_constant(ov::element::f32, ov::Shape{1}));

        auto slice2 = std::make_shared<ov::op::v8::Slice>(
            gate_up_add,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 0}),
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{3},
                                         std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 1, 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 1, 2}));
        auto minimum1 = std::make_shared<ov::op::v1::Minimum>(
            slice2,
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {10.0f}));
        auto swish_beta = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{expert_alpha});
        auto swish = std::make_shared<ov::op::v4::Swish>(minimum1, swish_beta);

        auto multiply2 = std::make_shared<ov::op::v1::Multiply>(add1, swish);

        // Down projection
        auto down_proj_matmul = std::make_shared<ov::op::v0::MatMul>(
            multiply2,
            ov::test::utils::make_constant(ov::element::f32,
                                           ov::Shape{number_of_experts, intermediate_size, hidden_size}));

        auto down_proj_add = std::make_shared<ov::op::v1::Add>(
            down_proj_matmul,
            ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, 1, hidden_size}));

        auto end_reshape = std::make_shared<ov::op::v1::Reshape>(
            down_proj_add,
            ov::op::v0::Constant::create(
                ov::element::i64,
                ov::Shape{4},
                std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),  // Use -1 for dynamic seq_len
            false);

        // Router subgraph - this is crucial for the MoE pattern recognition
        auto reshape_2nd_consumer_router_matmul = std::make_shared<ov::op::v0::MatMul>(
            experts_reshape,
            ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, hidden_size}),
            false,
            true);

        auto router_bias = std::make_shared<ov::op::v1::Add>(
            reshape_2nd_consumer_router_matmul,
            ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, number_of_experts}));

        auto router_topk_values_and_indices =
            std::make_shared<ov::op::v11::TopK>(router_bias,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                                -1,
                                                ov::op::v11::TopK::Mode::MAX,
                                                ov::op::v11::TopK::SortType::SORT_VALUES,
                                                ov::element::i64);

        auto router_topk_values = router_topk_values_and_indices->output(0);
        auto router_topk_indices = router_topk_values_and_indices->output(1);

        // ScatterElementsUpdate: Follow reference pattern exactly
        auto scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            router_topk_values,                                                           // data
            router_topk_indices,                                                          // indices
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{batch, topk}, {0}),  // updates - match reference
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));  // axis

        // Transpose: Dynamic shape transpose
        auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
            scatter_elements_update,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

        auto router_reshape = std::make_shared<ov::op::v1::Reshape>(
            router_transpose,
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{3},
                                         std::vector<int64_t>{number_of_experts, batch, -1}),
            true);

        auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
            router_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

        auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

        // ReduceSum - final node of the MOE pattern
        auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
            mul3,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
            false);

        ov::ParameterVector params = {input};
        return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(reduce_sum)},
                                           params,
                                           "MoeSubgraph");
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const auto& [shape_params,
                     weights_precision,
                     decompression_precision,
                     scale_precision,
                     use_weight_decompression,
                     decompression_multiply_type,
                     decompression_subtract_type,
                     reshape_on_decompression,
                     additional_config,
                     fusing_params] = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        std::tie(postOpMgrPtr, fusedOps) = fusing_params;

        init_input_shapes({shape_params.data_shape});

        inType = outType = ov::element::f32;
        function = initMoeSubgraph(shape_params,
                                   ov::element::f32,
                                   weights_precision,
                                   decompression_precision,
                                   ov::element::f32,
                                   use_weight_decompression,
                                   decompression_multiply_type,
                                   decompression_subtract_type,
                                   reshape_on_decompression);
    }
};

namespace {
// Test parameter generation
const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};

const std::vector<MoeShapeParams> input_shapes_basic = {
    {
        {{2, -1, 2048}, {{2, 1, 2048}, {2, 4, 2048}, {2, 8, 2048}}},  // data_shape - dynamic seq_len: batch=2,
                                                                      // seq_len=dynamic, hidden_size=2048
        2048,                                                         // hidden_size - match reference
        4096,                                                         // intermediate_size - match reference
        3,                                                            // number_of_experts - match reference
        2,                                                            // topk
        2,                                                            // fusion_factor
        {3, 2048, 8192},  // gate_up_weights_shape - match reference dimensions
        {3, 4096, 2048},  // down_proj_weights_shape - match reference dimensions
        64                // decompression_group_size
    },
    {
        {{1, -1, 1024}, {{1, 2, 1024}, {1, 6, 1024}, {1, 16, 1024}}},  // Different batch size and hidden size
        1024,                                                          // hidden_size
        2048,                                                          // intermediate_size
        4,                                                             // number_of_experts
        2,                                                             // topk
        2,                                                             // fusion_factor
        {4, 1024, 4096},                                               // gate_up_weights_shape
        {4, 2048, 1024},                                               // down_proj_weights_shape
        32                                                             // decompression_group_size
    }};
auto filter_additional_config_basic = []() {
    std::vector<std::map<std::string, ov::Any>> additional_config = {
        {{ov::hint::inference_precision.name(), ov::element::f32}}};
    return additional_config;
};

auto filter_additional_config_bf16 = []() {
    std::vector<std::map<std::string, ov::Any>> additional_config = {
        {{ov::hint::inference_precision.name(), ov::element::bf16}}};
    return additional_config;
};

const std::vector<CPUTestUtils::fusingSpecificParams> fusing_params = {emptyFusingSpec};

}  // namespace

// Basic FP32 tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_basic,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::AnyMap{
                                                {ov::hint::inference_precision.name(), ov::element::f32}}),
                                            ::testing::ValuesIn(fusing_params)),
                         MoeSubgraphTest::getTestCaseName);

// BF16 inference precision tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_bf16,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::AnyMap{
                                                {ov::hint::inference_precision.name(), ov::element::bf16}}),
                                            ::testing::ValuesIn(fusing_params)),
                         MoeSubgraphTest::getTestCaseName);

TEST_P(MoeSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

}  // namespace ov::test
