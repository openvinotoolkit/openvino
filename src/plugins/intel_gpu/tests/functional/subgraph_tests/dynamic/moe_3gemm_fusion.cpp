// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/moe_builders.hpp"

namespace {
using ov::test::InputShape;
using ov::test::MoERoutingType;

//
// Builds a pre-fusion OV model that exercises the full MOE3Gemm transformation pipeline:
//   1. Expert computation: Reshape→Tile→Reshape→3×MatMul(decompressed weights, transpose_b)
//                          →Swish→Multiply→Reshape
//   2. Routing subgraph: MatMul→Softmax/Sigmoid→TopK→normalize→scatter→reshape→unsqueeze
//   3. Combination: Multiply(expert, routing)→ReduceSum
//
// The GPU plugin transformation pipeline converts this into a fused kernel:
//   FuseVectorizedMOE3GEMM       (folds the expert+combination subgraph into op::internal::MOE)
//   ConvertMOEToMOECompressed     (absorbs compressed weights into MOECompressed)
//   FuseMOE3GemmCompressed        (fuses routing subgraph for Softmax / Sigmoid+bias routing)
// which collapse the subgraph into a single MOE3GemmFusedCompressed kernel.
//
// Model parameters (kept small for fast execution):
//   tokens (seq) =   4  (SEQ_LEN)
//   hidden_size  = 128  (HIDDEN_SIZE)
//   inter_size   = 128  (INTER_SIZE)
//   num_experts  =   4  (NUM_EXPERTS)
//   top_k        =   2  (TOP_K)
//   group_size   = 128  (GROUP_SIZE)
//
// op::internal::MOE is GPU-only; validate() checks output shape and finiteness only.
//

class MoE3GemmFusionTest : public testing::WithParamInterface<MoERoutingType>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<MoERoutingType>& info) {
        return info.param == MoERoutingType::SIGMOID_BIAS ? "SigmoidBias" : "Softmax";
    }

protected:
    static constexpr size_t SEQ_LEN = 4;
    static constexpr size_t HIDDEN_SIZE = 128;
    static constexpr size_t INTER_SIZE = 128;
    static constexpr size_t NUM_EXPERTS = 4;
    static constexpr size_t TOP_K = 2;
    static constexpr size_t GROUP_SIZE = 128;

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        inType = outType = ov::element::f16;

        const InputShape hidden_shape{ov::PartialShape{-1, HIDDEN_SIZE}, {{SEQ_LEN, HIDDEN_SIZE}}};
        init_input_shapes({hidden_shape});

        // Build via the shared builder.
        // input_precision=f16 causes the builder to emit Parameter(f16)+Convert(f32), which is
        // required for the GPU plugin to set inference_precision=f16 (needed by the
        // MOE3GemmFusedCompressed kernel on XE-HPG/XE2+ hardware).
        const ov::test::MoePatternParams params{
            hidden_shape,
            TOP_K,
            NUM_EXPERTS,
            INTER_SIZE,
        };
        function = ov::test::initMoE3GeMMSubgraph(
            params,
            ov::element::f32,    // data_precision – computation in f32
            ov::element::u4,     // weights_precision
            true,                // use_weight_decompression
            ov::element::f16,    // decompression_precision (weights/zp cast to f16)
            ov::element::f16,    // scale_precision
            ov::test::utils::DecompressionType::full,
            ov::test::utils::DecompressionType::full,
            true,                // reshape_on_decompression
            static_cast<int>(GROUP_SIZE),
            GetParam(),          // MoERoutingType
            ov::element::f16);   // input_precision – f16 Parameter
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& itTargetShape = target_input_static_shapes.front();
        const auto& params = function->get_parameters();
        ASSERT_EQ(params.size(), 1);
        auto param = params.front();
        auto type = param->get_element_type();

        auto input_tensor = ov::test::utils::create_and_fill_tensor(type, itTargetShape, ov::test::utils::InputGenerateData(0.125, 2, 8, 1234));

        inputs.insert({param, input_tensor});
    }
};

TEST_P(MoE3GemmFusionTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MoE3GemmFusion,
                         MoE3GemmFusionTest,
                         ::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS),
                         MoE3GemmFusionTest::get_test_case_name);
}  // namespace
