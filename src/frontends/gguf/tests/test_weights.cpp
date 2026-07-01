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
                                           WeightCase{"q4_k", "Q4_K", kTolIntZp},
                                           WeightCase{"q5_k", "Q5_K", kTolRequant},
                                           WeightCase{"q6_k", "Q6_K", kTolRequant}),
                         [](const ::testing::TestParamInfo<WeightCase>& i) {
                             return std::string(i.param.stem);
                         });

// Known-failing (fill_q2_k / fill_q3_k diverge from ggml), kept DISABLED to match
// test_dequant_vs_ggml.cpp.
INSTANTIATE_TEST_SUITE_P(DISABLED_KnownFailingQuantTypes,
                         GGUFWeight,
                         ::testing::Values(WeightCase{"q2_k", "Q2_K", kTolFaithful},
                                           WeightCase{"q3_k", "Q3_K", kTolFaithful}),
                         [](const ::testing::TestParamInfo<WeightCase>& i) {
                             return std::string(i.param.stem);
                         });

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
