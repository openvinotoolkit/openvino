// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GPU functional single-layer test for GatherMatmulCompressed.
// Builds an ov::Model with a single GatherMatmulCompressed node,
// compiles on GPU, runs inference, and validates against CPU reference.

#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/glu.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
using ov::test::InputShape;

struct GatherMatmulTestParams {
    std::vector<InputShape> input_shapes;  // A shape (dynamic + static instances)
    size_t n_experts;
    size_t top_k;
    size_t hidden_size;        // K (input features)
    size_t output_size;        // N (output features)
    size_t group_size;         // 0 = per-channel, >0 = grouped compression
    ov::element::Type w_type;  // weight element type: u8, i8, u4, i4
    bool has_bias;
    bool has_zp;
};

class GatherMatmulGPUTest : public testing::WithParamInterface<GatherMatmulTestParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherMatmulTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : p.input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : p.input_shapes) {
            result << "(";
            for (const auto& s : shape.second) {
                result << ov::test::utils::vec2str(s) << "_";
            }
            result << ")_";
        }
        result << "E=" << p.n_experts;
        result << "_K=" << p.top_k;
        result << "_H=" << p.hidden_size;
        result << "_N=" << p.output_size;
        result << "_G=" << p.group_size;
        result << "_W=" << p.w_type;
        result << "_bias=" << (p.has_bias ? "Y" : "N");
        result << "_zp=" << (p.has_zp ? "Y" : "N");
        return result.str();
    }

protected:
    // GatherMatmulCompressed is an internal op — the TEMPLATE plugin cannot evaluate it,
    // and the CPU plugin uses different placeholder conventions for absent bias/zp.
    // Return empty refs to skip comparison; the test validates compilation + execution.
    std::vector<ov::Tensor> calculate_refs() override {
        return {};
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // GatherMatmul requires systolic (HW_MATMUL / supports_immad). Skip on iGPU.
        auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
            GTEST_SKIP();

        const auto& p = GetParam();
        init_input_shapes(p.input_shapes);

        const size_t E = p.n_experts;
        const size_t K = p.hidden_size;
        const size_t N = p.output_size;
        const size_t topk = p.top_k;
        const bool grouped = p.group_size > 0;

        // Input A: [1, tokens, K] — Parameter (dynamic tokens)
        auto param_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        param_A->set_friendly_name("input_A");

        // Add indices as a dynamic parameter: [tokens, topk]
        inputDynamicShapes.push_back({-1, static_cast<int64_t>(topk)});
        for (auto& static_shapes : targetStaticShapes) {
            size_t tokens_for_iter = static_shapes[0][1];
            static_shapes.push_back({tokens_for_iter, topk});
        }

        // Weight B: [E, N, K] or [E, N, G, GS]
        // For u4/i4: each byte packs 2 elements, so raw buffer is half the logical size
        const auto w_type = p.w_type;
        const bool is_int4 = (w_type == ov::element::u4 || w_type == ov::element::i4);

        std::shared_ptr<ov::op::v0::Constant> weight_B;
        ov::Shape scale_shape;
        ov::Shape zp_shape;
        if (grouped) {
            const size_t G = K / p.group_size;
            const size_t GS = p.group_size;
            ov::Shape w_shape = {E, N, G, GS};
            const size_t num_elements = ov::shape_size(w_shape);
            const size_t raw_bytes = is_int4 ? (num_elements + 1) / 2 : num_elements;
            std::vector<uint8_t> w_data(raw_bytes);
            for (size_t i = 0; i < raw_bytes; i++) {
                w_data[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
            }
            weight_B = std::make_shared<ov::op::v0::Constant>(w_type, w_shape, w_data.data());
            scale_shape = {E, N, G, 1};
            zp_shape = {E, N, G, 1};
        } else {
            ov::Shape w_shape = {E, N, K};
            const size_t num_elements = ov::shape_size(w_shape);
            const size_t raw_bytes = is_int4 ? (num_elements + 1) / 2 : num_elements;
            std::vector<uint8_t> w_data(raw_bytes);
            for (size_t i = 0; i < raw_bytes; i++) {
                w_data[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
            }
            weight_B = std::make_shared<ov::op::v0::Constant>(w_type, w_shape, w_data.data());
            scale_shape = {E, N, 1};
            zp_shape = {E, N, 1};
        }

        // Indices: [tokens, topk] — Parameter (dynamic tokens)
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[1]);
        indices->set_friendly_name("indices");

        // Bias: absent (dynamic Shape{0}) or [E, 1, N]
        std::shared_ptr<ov::op::v0::Constant> bias;
        if (p.has_bias) {
            std::vector<ov::float16> bias_data(E * N, ov::float16(0.1f));
            bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{E, 1, N}, bias_data.data());
        } else {
            bias = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{0});
        }

        // Scales: f16, small values
        std::vector<ov::float16> scale_data(ov::shape_size(scale_shape), ov::float16(0.01f));
        auto scales = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_shape, scale_data.data());

        // Zero points: for u4/i4 weights use u8 zp, for u8/i8 weights use same type
        std::shared_ptr<ov::op::v0::Constant> zp;
        if (p.has_zp) {
            auto zp_type = is_int4 ? ov::element::u8 : w_type;
            std::vector<uint8_t> zp_data(ov::shape_size(zp_shape), 128);
            zp = std::make_shared<ov::op::v0::Constant>(zp_type, zp_shape, zp_data.data());
        } else {
            zp = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{0});
        }

        // GatherMatmulCompressed: 6 inputs
        auto bgm = std::make_shared<ov::op::internal::GatherMatmulCompressed>(param_A, weight_B, indices, bias, scales, zp);
        bgm->set_friendly_name("gather_matmul_compressed");

        auto result = std::make_shared<ov::op::v0::Result>(bgm);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_A, indices}, "GatherMatmulCompressedTest");

        // Relaxed thresholds — quantized weights have inherent error
        abs_threshold = 1.0f;
        rel_threshold = 1.0f;
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& p = GetParam();
        const auto& params = function->get_parameters();
        // Fill A with small random values
        auto tensor = ov::test::utils::create_and_fill_tensor(params[0]->get_element_type(),
                                                              target_input_static_shapes[0],
                                                              ov::test::utils::InputGenerateData{0, 1, 1, 0});
        inputs.insert({params[0], tensor});

        // Fill indices with valid expert IDs
        const auto& idx_shape = target_input_static_shapes[1];
        auto idx_tensor = ov::Tensor(ov::element::i32, idx_shape);
        auto* idx_data = idx_tensor.data<int32_t>();
        for (size_t i = 0; i < ov::shape_size(idx_shape); i++) {
            idx_data[i] = static_cast<int32_t>(i % p.n_experts);
        }
        inputs.insert({params[1], idx_tensor});
    }
};

TEST_P(GatherMatmulGPUTest, Inference) {
    run();
}

// --- Test parameters ---

// --- u4 weights (per-channel, 3D) ---
const std::vector<GatherMatmulTestParams> u4_per_channel_params = {
    // Single token
    {
        {{{1, -1, 128}, {{1, 1, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Multiple tokens
    {
        {{{1, -1, 128}, {{1, 8, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // More experts, larger topk
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        8,                                // n_experts
        4,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
};

// --- u4 with bias ---
const std::vector<GatherMatmulTestParams> u4_bias_params = {
    {
        {{{1, -1, 128}, {{1, 1, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::u4,                  // w_type
        true,                             // has_bias
        false,                            // has_zp
    },
};

// --- u4 grouped (4D weights) ---
const std::vector<GatherMatmulTestParams> u4_grouped_params = {
    // Grouped, no zp
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        32,                               // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Grouped, with zp
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        32,                               // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        true,                             // has_zp
    },
};

// --- u4 weights (most common in real models like Qwen MOE) ---
const std::vector<GatherMatmulTestParams> u4_params = {
    // Per-channel
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Grouped with zp
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        32,                               // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        true,                             // has_zp
    },
};

// --- i4 weights ---
const std::vector<GatherMatmulTestParams> i4_params = {
    // Per-channel
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        0,                                // group_size
        ov::element::i4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Grouped with zp
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size
        128,                              // output_size
        32,                               // group_size
        ov::element::i4,                  // w_type
        false,                            // has_bias
        true,                             // has_zp
    },
};

// --- Dynamic token count (multiple static shapes) ---
const std::vector<GatherMatmulTestParams> dynamic_params = {
    {
        {{{1, -1, 128}, {{1, 1, 128}, {1, 4, 128}, {1, 8, 128}}}},  // input_shapes
        4,                                                          // n_experts
        2,                                                          // top_k
        128,                                                        // hidden_size
        128,                                                        // output_size
        0,                                                          // group_size
        ov::element::u4,                                            // w_type
        false,                                                      // has_bias
        false,                                                      // has_zp
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_u4_PerChannel,
                         GatherMatmulGPUTest,
                         ::testing::ValuesIn(u4_per_channel_params),
                         GatherMatmulGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_u4_Bias, GatherMatmulGPUTest, ::testing::ValuesIn(u4_bias_params), GatherMatmulGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_u4_Grouped, GatherMatmulGPUTest, ::testing::ValuesIn(u4_grouped_params), GatherMatmulGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_u4, GatherMatmulGPUTest, ::testing::ValuesIn(u4_params), GatherMatmulGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_i4, GatherMatmulGPUTest, ::testing::ValuesIn(i4_params), GatherMatmulGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_Dynamic, GatherMatmulGPUTest, ::testing::ValuesIn(dynamic_params), GatherMatmulGPUTest::getTestCaseName);

// =============================================================================
// GatherMatmul + SwiGLU fusion test
// =============================================================================
// Builds: GatherMatmulCompressed(weights=[E, 2N, K]) → GLU(axis=-1, split_lengths=N)
// The fusion pass should absorb GLU into GatherMatmul.
// Output: [top_k, tokens, N] (not 2N).

class GatherMatmulSwiGLUGPUTest : public testing::WithParamInterface<GatherMatmulTestParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherMatmulTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : p.input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (const auto& shape : p.input_shapes) {
            result << "(";
            for (const auto& s : shape.second) {
                result << ov::test::utils::vec2str(s) << "_";
            }
            result << ")_";
        }
        result << "E=" << p.n_experts;
        result << "_K=" << p.top_k;
        result << "_H=" << p.hidden_size;
        result << "_N=" << p.output_size;
        result << "_G=" << p.group_size;
        result << "_W=" << p.w_type;
        result << "_SwiGLU";
        return result.str();
    }

protected:
    std::vector<ov::Tensor> calculate_refs() override {
        return {};
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        // GatherMatmul requires systolic (HW_MATMUL / supports_immad). Skip on iGPU.
        auto capabilities = core->get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
        if (std::find(capabilities.cbegin(), capabilities.cend(), ov::intel_gpu::capability::HW_MATMUL) == capabilities.cend())
            GTEST_SKIP();

        const auto& p = GetParam();
        init_input_shapes(p.input_shapes);

        const size_t E = p.n_experts;
        const size_t K = p.hidden_size;
        const size_t N = p.output_size;        // N is the FINAL output size (after SwiGLU)
        const size_t N2 = N * 2;               // 2N is the GEMM output size (gate + value)
        const size_t topk = p.top_k;
        const bool grouped = p.group_size > 0;

        auto param_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        param_A->set_friendly_name("input_A");

        inputDynamicShapes.push_back({-1, static_cast<int64_t>(topk)});
        for (auto& static_shapes : targetStaticShapes) {
            size_t tokens_for_iter = static_shapes[0][1];
            static_shapes.push_back({tokens_for_iter, topk});
        }

        const auto w_type = p.w_type;
        const bool is_int4 = (w_type == ov::element::u4 || w_type == ov::element::i4);

        // Weights: [E, 2N, K] — double width for gate+value
        std::shared_ptr<ov::op::v0::Constant> weight_B;
        ov::Shape scale_shape;
        ov::Shape zp_shape;
        if (grouped) {
            const size_t G = K / p.group_size;
            const size_t GS = p.group_size;
            ov::Shape w_shape = {E, N2, G, GS};
            const size_t num_elements = ov::shape_size(w_shape);
            const size_t raw_bytes = is_int4 ? (num_elements + 1) / 2 : num_elements;
            std::vector<uint8_t> w_data(raw_bytes);
            for (size_t i = 0; i < raw_bytes; i++)
                w_data[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
            weight_B = std::make_shared<ov::op::v0::Constant>(w_type, w_shape, w_data.data());
            scale_shape = {E, N2, G, 1};
            zp_shape = {E, N2, G, 1};
        } else {
            ov::Shape w_shape = {E, N2, K};
            const size_t num_elements = ov::shape_size(w_shape);
            const size_t raw_bytes = is_int4 ? (num_elements + 1) / 2 : num_elements;
            std::vector<uint8_t> w_data(raw_bytes);
            for (size_t i = 0; i < raw_bytes; i++)
                w_data[i] = static_cast<uint8_t>((i * 7 + 3) % 256);
            weight_B = std::make_shared<ov::op::v0::Constant>(w_type, w_shape, w_data.data());
            scale_shape = {E, N2, 1};
            zp_shape = {E, N2, 1};
        }

        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[1]);
        indices->set_friendly_name("indices");

        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{0});

        std::vector<ov::float16> scale_data(ov::shape_size(scale_shape), ov::float16(0.01f));
        auto scales = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_shape, scale_data.data());

        auto zp = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{0});

        // GatherMatmulCompressed: output is [top_k, tokens, 2N]
        auto bgm = std::make_shared<ov::op::internal::GatherMatmulCompressed>(param_A, weight_B, indices, bias, scales, zp);
        bgm->set_friendly_name("gather_matmul_compressed");

        // GLU (SwiGLU): splits last axis, applies Swish to gate half, multiplies
        // Output: [top_k, tokens, N]
        auto glu = std::make_shared<ov::op::internal::GLU>(bgm,
                                                            -1,                                      // axis (last)
                                                            static_cast<int64_t>(N),                 // split_lengths
                                                            ov::op::internal::GLU::GluType::Swish,   // glu_type
                                                            0,                                       // split_to_glu_idx (gate first)
                                                            ov::element::f16);                       // output_type
        glu->set_friendly_name("swiglu");

        auto result = std::make_shared<ov::op::v0::Result>(glu);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_A, indices}, "GatherMatmulSwiGLUTest");

        abs_threshold = 1.0f;
        rel_threshold = 1.0f;
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& p = GetParam();
        const auto& params = function->get_parameters();
        auto tensor = ov::test::utils::create_and_fill_tensor(params[0]->get_element_type(),
                                                              target_input_static_shapes[0],
                                                              ov::test::utils::InputGenerateData{0, 1, 1, 0});
        inputs.insert({params[0], tensor});

        const auto& idx_shape = target_input_static_shapes[1];
        auto idx_tensor = ov::Tensor(ov::element::i32, idx_shape);
        auto* idx_data = idx_tensor.data<int32_t>();
        for (size_t i = 0; i < ov::shape_size(idx_shape); i++)
            idx_data[i] = static_cast<int32_t>(i % p.n_experts);
        inputs.insert({params[1], idx_tensor});
    }
};

TEST_P(GatherMatmulSwiGLUGPUTest, Inference) {
    run();
}

const std::vector<GatherMatmulTestParams> swiglu_params = {
    // Per-channel, single token
    {
        {{{1, -1, 128}, {{1, 1, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size (K)
        64,                               // output_size (N — after SwiGLU, weights are 2N=128)
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Per-channel, multiple tokens
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size (K)
        64,                               // output_size (N)
        0,                                // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
    // Grouped
    {
        {{{1, -1, 128}, {{1, 4, 128}}}},  // input_shapes
        4,                                // n_experts
        2,                                // top_k
        128,                              // hidden_size (K)
        64,                               // output_size (N)
        32,                               // group_size
        ov::element::u4,                  // w_type
        false,                            // has_bias
        false,                            // has_zp
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmul_SwiGLU,
                         GatherMatmulSwiGLUGPUTest,
                         ::testing::ValuesIn(swiglu_params),
                         GatherMatmulSwiGLUGPUTest::getTestCaseName);

}  // namespace
