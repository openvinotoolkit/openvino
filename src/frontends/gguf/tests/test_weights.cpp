// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests for weights surfaced as GGML_OP_NONE (leaf) nodes.
//
// A weight is a regular node in the gguf frontend: a ggml leaf (op type "GGML_OP_NONE") that
// the decoder marks as a weight by providing the raw GGUF bytes (get_attribute<ov::Tensor>
// ("data")), the ggml quant type name (get_attribute<std::string>("quant_type")) and the
// logical [rows, cols] shape (get_output_shape()); the frontend's translate_weight does the
// dequant. These tests build a single GGML_OP_NONE model -- exactly like the single-op tests in
// test_ops.cpp -- run it on CPU and compare against the same real-ggml reference data used by
// test_dequant_vs_ggml.cpp.

#include <cmath>
#include <cstring>

#include "op_test_utils.hpp"
#include "quant/weights.hpp"

using namespace ov_gguf_test;

namespace {

// rows/cols and tolerance match the reference generator (see test_dequant_vs_ggml.cpp).
// Q5_K/Q6_K requantize to channel-wise Q8_0_C (matching the llama.cpp ggml-openvino CPU/GPU
// backend), so they diverge from ggml's faithful to_float by the Q8_0_C round-off rather
// than f16 noise.
constexpr size_t kRows = 4;
constexpr size_t kCols = 256;
constexpr float kTolFaithful = 3e-3f;
constexpr float kTolRequant = 1.5e-2f;
// Q4_K uses an INTEGER (u8) zero-point so the CPU plugin fuses the dequant into the MatMul
// (matching the original ggml-openvino backend); the integer zp diverges from ggml's faithful
// to_float by up to ~0.045 per weight.
constexpr float kTolIntZp = 5e-2f;

struct WeightCase {
    const char* stem;        // test_data prefix
    const char* quant_type;  // name passed to the frontend
    float tol;
};

ov::Tensor bytes_to_u8_tensor(const std::vector<uint8_t>& bytes) {
    ov::Tensor t(ov::element::u8, ov::Shape{bytes.size()});
    std::memcpy(t.data(), bytes.data(), bytes.size());
    return t;
}

}  // namespace

class GGUFWeight : public ::testing::TestWithParam<WeightCase> {};

// Convert a GGML_OP_NONE weight node and check the produced (constant-foldable) dequant against
// ggml's to_float reference.
TEST_P(GGUFWeight, MatchesGgmlToFloat) {
    const WeightCase c = GetParam();
    const auto qbytes = load_npy<uint8_t>(std::string(c.stem) + "_qbytes");
    const auto ref = load_npy<float>(std::string(c.stem) + "_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {kRows, kCols})
                     .attr<ov::Tensor>("data", bytes_to_u8_tensor(qbytes))
                     .attr<std::string>("quant_type", c.quant_type)
                     .build();

    // The weight node has no graph inputs; it folds to a constant. Run it to read the values.
    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), ref.size());

    const float* a = out.data<float>();
    float max_diff = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - ref[i]));
    EXPECT_LE(max_diff, c.tol) << c.stem << ": frontend weight dequant diverges from ggml to_float";
}

INSTANTIATE_TEST_SUITE_P(AllQuantTypes,
                         GGUFWeight,
                         ::testing::Values(WeightCase{"q4_0", "Q4_0", kTolFaithful},
                                           WeightCase{"q4_1", "Q4_1", kTolFaithful},
                                           WeightCase{"q5_0", "Q5_0", kTolFaithful},
                                           WeightCase{"q5_1", "Q5_1", kTolFaithful},
                                           WeightCase{"q8_0", "Q8_0", kTolFaithful},
                                           WeightCase{"q2_k", "Q2_K", kTolFaithful},
                                           WeightCase{"q3_k", "Q3_K", kTolFaithful},
                                           WeightCase{"q4_k", "Q4_K", kTolIntZp},
                                           WeightCase{"q5_k", "Q5_K", kTolRequant},
                                           WeightCase{"q6_k", "Q6_K", kTolRequant},
                                           WeightCase{"q2_0", "Q2_0", kTolFaithful}),
                         [](const ::testing::TestParamInfo<WeightCase>& i) {
                             return std::string(i.param.stem);
                         });

// token_embd / output are requantized to channel-wise Q8_0_C, and that path reads the zero-point
// as f16 -- Q2_0 used to hard-code u8 here, which threw for every ternary model.
TEST(GGUFWeightRequant, Q2_0AsTokenEmbd) {
    const auto qbytes = load_npy<uint8_t>("q2_0_qbytes");
    const auto ref = load_npy<float>("q2_0_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("token_embd.weight", ov::element::f32, {kRows, kCols})
                     .attr<ov::Tensor>("data", bytes_to_u8_tensor(qbytes))
                     .attr<std::string>("quant_type", "Q2_0")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), ref.size());

    const float* a = out.data<float>();
    float max_diff = 0.f;
    for (size_t i = 0; i < ref.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(a[i] - ref[i]));
    EXPECT_LE(max_diff, kTolRequant) << "Q2_0 token_embd requant diverges from ggml to_float";
}

// An F16 weight is wrapped directly as a constant (no dequant); round-trips the raw bytes.
TEST(GGUFWeightPlain, F16) {
    std::vector<ov::float16> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(ov::float16)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "F16")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(a[i], static_cast<float>(vals[i]), 1e-6f);
}

// An F32 weight is wrapped directly as a Constant (no dequant, no Convert); round-trips exactly.
TEST(GGUFWeightPlain, F32) {
    std::vector<float> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(float)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "F32")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_EQ(a[i], vals[i]);
}

// A BF16 weight is wrapped as a bf16 Constant then Convert'ed to f32 for the translators.
TEST(GGUFWeightPlain, BF16) {
    std::vector<ov::bfloat16> vals{1.0f, -2.0f, 3.5f, -4.25f, 0.0f, 7.0f};
    ov::Tensor data(ov::element::u8, ov::Shape{vals.size() * sizeof(ov::bfloat16)});
    std::memcpy(data.data(), vals.data(), data.get_byte_size());

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {2, 3})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "BF16")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), vals.size());
    const float* a = out.data<float>();
    for (size_t i = 0; i < vals.size(); ++i)
        EXPECT_NEAR(a[i], static_cast<float>(vals[i]), 1e-6f);
}

// Q8_K is a ggml intermediate activation-quantization type: it only ever appears as the
// on-the-fly quantized activation in a dot product, never as a weight tensor stored in a .gguf.
// The frontend therefore does not accept it as a weight -- gguf_type_from_name knows the name
// (so a stray reference is a clear error, not a silent misparse), but make_weight_node has no
// Q8_K path and rejects it. If a future model ever stores Q8_K weights, wire it into the
// make_weight_node switch and turn this into a value test.
TEST(GGUFWeightUnsupported, Q8KIsNotAStoredWeight) {
    // One Q8_K block = 292 bytes covers 256 weights.
    ov::Tensor data(ov::element::u8, ov::Shape{292});
    std::memset(data.data(), 0, data.get_byte_size());
    EXPECT_ANY_THROW({
        SingleOpBuilder()
            .op("GGML_OP_NONE")
            .output("w", ov::element::f32, {1, 256})
            .attr<ov::Tensor>("data", data)
            .attr<std::string>("quant_type", "Q8_K")
            .build();
    });
}

// A stored MoE expert weight is rank>2 ([1,n_expert,m,k]) and stays PACKED for MUL_MAT_ID (see
// GGUFOps.MulMatIdMxfp4Packed). But the llama.cpp cgraph path (e.g. test-backend-ops' GET_ROWS /
// MUL_MAT op tests) also feeds bare 2D MXFP4 tensors that are not stored weights at all, so
// make_weight_node must dequantize those too. f4e2m1 nibble -> value mirrors kF4E2M1 in
// test_ops.cpp; scale byte 127 -> e8m0 exponent 0 -> 2^0 = 1.0.
TEST(GGUFWeight, Mxfp4TwoDim) {
    static const float kF4E2M1[16] =
        {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    // 1 row, 1 block (32 cols): byte0 = e8m0 scale, bytes1..16 = nibble-packed codes (low nibble
    // = element i, high nibble = element i+16, for i in [0,16)).
    ov::Tensor data(ov::element::u8, ov::Shape{17});
    auto* bytes = data.data<uint8_t>();
    bytes[0] = 127;
    for (size_t i = 0; i < 16; ++i)
        bytes[1 + i] = static_cast<uint8_t>((i << 4) | i);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_NONE")
                     .output("w", ov::element::f32, {1, 32})
                     .attr<ov::Tensor>("data", data)
                     .attr<std::string>("quant_type", "MXFP4")
                     .build();

    auto out = run_on_cpu(model, {});
    ASSERT_EQ(out.get_size(), 32u);
    const float* a = out.data<float>();
    for (size_t j = 0; j < 32; ++j)
        EXPECT_FLOAT_EQ(a[j], kF4E2M1[j % 16]) << "element " << j;
}

// Regression for a fused-QKV bias (e.g. phi-3's blk.N.attn_qkv.bias) being silently dropped:
// register_fused_qkv split the fused weight into q/k/v parts but never touched a fused bias, so
// attention()'s add_bias calls (gated on !has_fused_qkv) never ran for these archs, and the
// bias never made it into the graph. split_fused_qkv_bias must slice the fused bias into
// exactly the q/k/v parts the weight split uses (n_q, n_k, n_v rows in that order).
TEST(GGUFFusedQkvBias, SplitsIntoExpectedRanges) {
    constexpr size_t n_q = 4, n_k = 2, n_v = 2;
    ov::Tensor bias(ov::element::f32, ov::Shape{n_q + n_k + n_v});
    auto* data = bias.data<float>();
    for (size_t i = 0; i < n_q + n_k + n_v; ++i) {
        data[i] = static_cast<float>(i);
    }
    std::unordered_map<std::string, ov::Tensor> weights{{"blk.0.attn_qkv.bias", bias}};

    auto parts = ov::frontend::gguf::split_fused_qkv_bias("blk.0.attn_qkv", weights, n_q, n_k, n_v);

    ASSERT_EQ(parts[0].get_shape(), ov::Shape{n_q});
    ASSERT_EQ(parts[1].get_shape(), ov::Shape{n_k});
    ASSERT_EQ(parts[2].get_shape(), ov::Shape{n_v});
    for (size_t i = 0; i < n_q; ++i) {
        EXPECT_FLOAT_EQ(parts[0].data<float>()[i], static_cast<float>(i));
    }
    for (size_t i = 0; i < n_k; ++i) {
        EXPECT_FLOAT_EQ(parts[1].data<float>()[i], static_cast<float>(n_q + i));
    }
    for (size_t i = 0; i < n_v; ++i) {
        EXPECT_FLOAT_EQ(parts[2].data<float>()[i], static_cast<float>(n_q + n_k + i));
    }
}

TEST(GGUFFusedQkvWeight, PreservesScalesForMxfp4AndQ8K) {
    const auto check = [](ov::frontend::gguf::GgufTensorType qtype,
                          const ov::element::Type& weight_type,
                          const ov::element::Type& scale_type,
                          size_t groups) {
        constexpr size_t rows = 6;
        constexpr size_t cols = 256;
        std::unordered_map<std::string, ov::Tensor> weights{
            {"fused.weight", ov::Tensor(weight_type, {rows, cols})},
            {"fused.scales", ov::Tensor(scale_type, {rows, groups})},
        };
        std::unordered_map<std::string, ov::frontend::gguf::GgufTensorType> qtypes{{"fused.qtype", qtype}};

        auto qkv = ov::frontend::gguf::split_fused_qkv_extracted("fused", weights, qtypes, 2, 2, 2);
        for (const auto& part : qkv) {
            ASSERT_TRUE(part.tensors.scales);
            EXPECT_EQ(part.tensors.scales.get_shape(), (ov::Shape{2, groups}));
        }

        auto q_gate = ov::frontend::gguf::split_interleaved_q_gate("fused", weights, qtypes, 1);
        for (const auto& part : q_gate) {
            ASSERT_TRUE(part.tensors.scales);
            EXPECT_EQ(part.tensors.scales.get_shape(), (ov::Shape{3, groups}));
        }
    };

    check(ov::frontend::gguf::GGUF_TYPE_MXFP4, ov::element::f4e2m1, ov::element::f8e8m0, 8);
    check(ov::frontend::gguf::GGUF_TYPE_Q8_K, ov::element::i8, ov::element::f32, 1);
}

TEST(GGUFWeight, RejectsMissingAuxiliaryTensors) {
    using namespace ov::frontend::gguf;

    WeightTensors without_scales{ov::Tensor(ov::element::i4, {1, 32}), {}, {}};
    EXPECT_THROW(make_weight_node(without_scales, GGUF_TYPE_Q4_0), ov::Exception);

    WeightTensors without_zero_point{ov::Tensor(ov::element::u32, {1, 4}), ov::Tensor(ov::element::f16, {1, 1}), {}};
    EXPECT_THROW(make_weight_node(without_zero_point, GGUF_TYPE_Q4_K), ov::Exception);
}
