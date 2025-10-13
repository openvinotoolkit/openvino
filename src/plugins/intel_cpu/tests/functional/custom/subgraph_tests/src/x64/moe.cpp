// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/moe.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/scatter_elements_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
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

struct MoePatternParams {
    ov::test::InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

typedef std::tuple<MoePatternParams,
                   ov::test::ElementType,  // weights precision
                   ov::test::ElementType,  // decompression precision
                   ov::test::ElementType,  // scale precision
                   bool,                   // use weight decompression
                   DecompressionType,      // decompression multiply type
                   DecompressionType,      // decompression subtract type
                   bool,                   // reshape on decompression constants
                   int,                    // decompression_group_size
                   ov::AnyMap>             // additional config
    MoeTestParams;

class MoeSubgraphTest : public testing::WithParamInterface<MoeTestParams>,
                        virtual public SubgraphBaseTest,
                        public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj) {
        const auto& [moe_params,
                     weights_precision,
                     decompression_precision,
                     scale_precision,
                     use_weight_decompression,
                     decompression_multiply_type,
                     decompression_subtract_type,
                     reshape_on_decompression,
                     decompression_group_size,
                     additional_config] = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
        result << "TS=";
        for (const auto& static_shape : moe_params.data_shape.second) {
            result << ov::test::utils::vec2str(static_shape) << ",";
        }
        result << "top_k_experts=" << moe_params.topk << "_";
        result << "total_experts=" << moe_params.number_of_experts << "_";
        result << "intermediate_size=" << moe_params.intermediate_size << "_";

        if (use_weight_decompression) {
            result << "WP=" << weights_precision << "_";
            result << "DP=" << decompression_precision << "_";
            result << "SP=" << scale_precision << "_";
            result << "DM=" << decompression_multiply_type << "_";
            result << "DS=" << decompression_subtract_type << "_";
            result << "RD=" << reshape_on_decompression << "_";
            result << "GS=" << decompression_group_size << "_";
        }

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> initMoeSubgraph(const MoePatternParams& moe_params,
                                               const ov::element::Type data_precision,
                                               const ov::element::Type weights_precision,
                                               const ov::element::Type decompression_precision,
                                               const ov::element::Type scale_precision,
                                               const bool use_weight_decompression,
                                               const DecompressionType decompression_multiply_type,
                                               const DecompressionType decompression_subtract_type,
                                               const bool reshape_on_decompression,
                                               const int decompression_group_size) {
        // Use parameters from shape_params - static shapes only
        const auto& data_shape = moe_params.data_shape;
        const int64_t intermediate_size = moe_params.intermediate_size;
        const int64_t topk = moe_params.topk;
        const int64_t number_of_experts = moe_params.number_of_experts;

        const auto expert_alpha = 1.702f;
        const auto expert_beta = 7.0f;

        constexpr int64_t fusion_factor = 2;  // property of GPT-OSS

        const auto& input_shape = data_shape.first;
        // Create input parameter with dynamic shape - batch and hidden_size are fixed, seq_len is dynamic
        auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

        OPENVINO_ASSERT(input_shape[2].is_static());
        const auto hidden_size = input_shape[2].get_length();

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
                                         std::vector<int64_t>{number_of_experts, -1, hidden_size}),
            false);

        // Gate/Up projection
        auto gate_up_matmul = std::make_shared<ov::op::v0::MatMul>(
            after_tile_reshape,
            ov::test::utils::make_constant(
                ov::element::f32,
                ov::Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size}),
            false,
            true);

        gate_up_matmul->set_friendly_name("GateUpMatMul");

        auto gate_up_add = std::make_shared<ov::op::v1::Add>(
            gate_up_matmul,
            ov::test::utils::make_constant(ov::element::f32,
                                           ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor}));

        // Slice the last axis, every second element
        auto slice1 = std::make_shared<ov::op::v8::Slice>(
            gate_up_add,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{1},
                                         std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));
        auto clamp = std::make_shared<ov::op::v0::Clamp>(slice1, -expert_beta, expert_beta);
        auto add1 =
            std::make_shared<ov::op::v1::Add>(clamp, ov::test::utils::make_constant(ov::element::f32, ov::Shape{1}));

        auto slice2 = std::make_shared<ov::op::v8::Slice>(
            gate_up_add,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{1},
                                         std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));
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
                                           ov::Shape{number_of_experts, hidden_size, intermediate_size}),
            false,
            true);

        down_proj_matmul->set_friendly_name("DownProjMatMul");

        auto down_proj_add = std::make_shared<ov::op::v1::Add>(
            down_proj_matmul,
            ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, 1, hidden_size}));

        const auto fetch_input_shape = std::make_shared<ov::op::v0::ShapeOf>(input);
        // here we simulate shape calculation
        // concat a const with the total number of experts and fetch_input_shape
        const auto number_of_experts_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {number_of_experts});
        const auto end_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, fetch_input_shape}, 0);

        auto end_reshape = std::make_shared<ov::op::v1::Reshape>(down_proj_add, end_shape, false);

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
        auto router_topk_values_softmax = std::make_shared<ov::op::v1::Softmax>(router_topk_values, 1);
        auto router_topk_indices = router_topk_values_and_indices->output(1);

        // add ShapeOf after Add
        auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(router_bias);
        // broadcast zeroes to the tensor of shape [batch, topk]
        auto zero_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
        auto broadcast_zero = std::make_shared<ov::op::v3::Broadcast>(zero_const, shape_of);

        auto scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            broadcast_zero,
            router_topk_indices,
            router_topk_values_softmax,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));  // axis

        // Transpose: Dynamic shape transpose
        auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
            scatter_elements_update,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

        const auto batch_seq_len = std::make_shared<ov::op::v1::Gather>(
            fetch_input_shape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

        const auto router_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, batch_seq_len}, 0);

        auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);

        auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
            router_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));

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

        /*(-:**/
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
                     decompression_group_size,
                     additional_config] = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());

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
                                   reshape_on_decompression,
                                   decompression_group_size);
    }
};

namespace {
// Test parameter generation
const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};

const std::vector<MoePatternParams> moe_params = {
    {
        {{-1, -1, 2048}, {{2, 15, 2048}, {2, 1, 2048}, {3, 8, 2048}}},  // data_shape,
                                                                        // seq_len=dynamic, hidden_size=2048
        4,                                                              // topk
        32,                                                             // number_of_experts
        4096                                                            // intermediate_size
    },
    {
        {{-1, -1, 1024}, {{1, 32, 1024}, {1, 1, 1024}, {1, 16, 1024}}},  // Different seq length
        6,                                                               // topk
        64,                                                              // number_of_experts
        2048                                                             // intermediate_size
    }};

const ov::AnyMap additional_config_basic = {{ov::hint::inference_precision.name(), ov::element::f32}};
const ov::AnyMap additional_config_bf16 = {{ov::hint::inference_precision.name(), ov::element::bf16}};

}  // namespace

// Basic FP32 tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_basic,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(additional_config_basic)),
                         MoeSubgraphTest::getTestCaseName);

// BF16 inference precision tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_bf16,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(additional_config_bf16)),
                         MoeSubgraphTest::getTestCaseName);

TEST_P(MoeSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

}  // namespace ov::test
