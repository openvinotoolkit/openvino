// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dequantization correctness tests with REAL ggml as the oracle.
//
// The reference data was produced offline by linking real ggml from llama.cpp
// (captured from real ggml): ggml quantizes smooth, asymmetric synthetic
// data into real GGUF-format blocks (_qbytes) and dequantizes those exact bytes
// (_deq).  The committed .npy files mean the tests need no ggml / llama.cpp at
// build or run time.
//
// Here we feed the SAME bytes through the frontend's fill + make_weight_node dequant
// subgraph and require it to match ggml's to_float output.
//
// Tolerance: ggml stores K-quant scales as f16 and the dequant subgraph runs in f16,
// so allow ~3e-3 (matching llama.cpp's MAX_QUANTIZATION_TOTAL_ERROR-class thresholds).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "quant/gguf.hpp"
#include "quant/weights.hpp"

using namespace ov::frontend::gguf;
using ov_gguf_test::load_npy;

TEST(GGUFDequant, ChannelwiseQ8ZeroRowRemainsZero) {
    const std::vector<float> row(32, 0.0f);
    std::vector<int8_t> weights(row.size(), 1);
    ov::float16 scale(1.0f);
    quantize_row_q8_0_c(row.data(), row.size(), weights.data(), scale);
    EXPECT_FLOAT_EQ(static_cast<float>(scale), 0.0f);
    EXPECT_EQ(weights, std::vector<int8_t>(row.size(), 0));
}

TEST(GGUFDequant, ChannelwiseQ8TinyRowsHaveFiniteQuantizedWeights) {
    for (const float maximum : {std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min()}) {
        SCOPED_TRACE(maximum);
        const std::vector<float> row{maximum, -maximum, 0.0f};
        std::vector<int8_t> weights(row.size());
        ov::float16 scale;
        quantize_row_q8_0_c(row.data(), row.size(), weights.data(), scale);
        EXPECT_EQ(weights, (std::vector<int8_t>{127, -127, 0}));
        EXPECT_FLOAT_EQ(static_cast<float>(scale), 0.0f);
    }
}

namespace {

std::vector<float> eval_as_f32(const std::shared_ptr<ov::Node>& node) {
    auto result = std::make_shared<ov::op::v0::Result>(node);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{});
    ov::pass::Manager pm;
    pm.register_pass<ov::pass::ConstantFolding>();
    pm.run_passes(model);
    auto folded =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(model->get_results()[0]->get_input_node_shared_ptr(0));
    EXPECT_NE(folded, nullptr) << "dequant graph did not constant-fold to a single Constant";
    if (!folded)
        return {};
    const float* d = folded->get_data_ptr<float>();
    return std::vector<float>(d, d + ov::shape_size(folded->get_shape()));
}

// Map the test's ggml type id to the quant-type name string the frontend API expects.
const char* type_name(uint32_t type) {
    switch (type) {
    case GGUF_TYPE_Q4_0:
        return "Q4_0";
    case GGUF_TYPE_Q4_1:
        return "Q4_1";
    case GGUF_TYPE_Q5_0:
        return "Q5_0";
    case GGUF_TYPE_Q5_1:
        return "Q5_1";
    case GGUF_TYPE_Q8_0:
        return "Q8_0";
    case GGUF_TYPE_Q8_1:
        return "Q8_1";
    case GGUF_TYPE_Q2_K:
        return "Q2_K";
    case GGUF_TYPE_Q3_K:
        return "Q3_K";
    case GGUF_TYPE_Q4_K:
        return "Q4_K";
    case GGUF_TYPE_Q5_K:
        return "Q5_K";
    case GGUF_TYPE_Q6_K:
        return "Q6_K";
    case GGUF_TYPE_Q2_0:
        return "Q2_0";
    case GGUF_TYPE_Q1_0:
        return "Q1_0";
    default:
        return "";
    }
}

// Build the frontend dequant of `rows x cols` weights from raw ggml block bytes `qbytes`
// via the public make_weight_node(data, quant_type, shape) entry point, then constant-fold
// it to f32 values (row-major, rows*cols).
std::vector<float> frontend_dequant(uint32_t type, const std::vector<uint8_t>& qbytes, uint64_t rows, uint64_t cols) {
    ov::Tensor data(ov::element::u8, ov::Shape{qbytes.size()});
    std::memcpy(data.data(), qbytes.data(), qbytes.size());
    auto node = make_weight_node(data, type_name(type), ov::Shape{rows, cols});
    return eval_as_f32(node);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float m = 0.f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

float mean_squared_error(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - b[i];
        sum += diff * diff;
    }
    return static_cast<float>(sum / a.size());
}

// One case: stem (test_data file prefix) + ggml quant enum + tolerance. rows/cols match the
// generator. Q5_K/Q6_K go through the channel-wise Q8_0_C requantization (matching the
// llama.cpp ggml-openvino CPU/GPU backend), so they diverge from ggml's faithful to_float by
// the Q8_0_C round-off (~1e-2) rather than by f16 noise (~3e-3).
struct DeqCase {
    const char* stem;
    uint32_t type;
    float tol;
};

constexpr uint64_t kRows = 4;
constexpr uint64_t kCols = 256;
constexpr float kTolFaithful = 3e-3f;   // f16-scale dequant noise
constexpr float kTolRequant = 1.5e-2f;  // channel-wise Q8_0_C requant round-off
// Q4_K is faithfully decoded and then requantized in its native 32-value groups to OpenVINO u4.
// Keep its worst error below the former rounded-zero-point path's 5e-2 tolerance.
constexpr float kTolU4Requant = 4e-2f;
// Q2_0: (code - 1) * d on both sides and the zero-point of 1 is exact, so hold it to bit-equality.
constexpr float kTolExact = 0.0f;

}  // namespace

class DequantVsGGML : public ::testing::TestWithParam<DeqCase> {};

TEST_P(DequantVsGGML, MatchesGgmlToFloat) {
    const DeqCase c = GetParam();
    const auto qbytes = load_npy<uint8_t>(std::string(c.stem) + "_qbytes");
    const auto ref = load_npy<float>(std::string(c.stem) + "_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    const auto ours = frontend_dequant(c.type, qbytes, kRows, kCols);
    ASSERT_EQ(ours.size(), ref.size());

    if (c.tol == 0.f) {
        for (size_t i = 0; i < ref.size(); ++i)
            EXPECT_EQ(ours[i], ref[i]) << "element " << i;
        return;
    }
    EXPECT_LE(max_abs_diff(ours, ref), c.tol)
        << c.stem << ": frontend dequant diverges from ggml to_float beyond tolerance";
}

// Guard both aggregate quality and the worst outlier against the real ggml CPU oracle. The
// previous SSE-only candidate selection lowered MSE but raised max error above 5e-2.
TEST(DequantVsGGML, Q4KRequantizationImprovesWithoutOutliers) {
    const auto qbytes = load_npy<uint8_t>("q4_k_qbytes");
    const auto ref = load_npy<float>("q4_k_deq");
    const auto ours = frontend_dequant(GGUF_TYPE_Q4_K, qbytes, kRows, kCols);

    ASSERT_EQ(ours.size(), ref.size());
    EXPECT_LE(mean_squared_error(ours, ref), 2.8e-4f);
    EXPECT_LE(max_abs_diff(ours, ref), kTolU4Requant);
}

TEST(DequantVsGGML, Q4KRejectsNonFiniteScaleMetadata) {
    const auto valid = load_npy<uint8_t>("q4_k_qbytes");
    constexpr uint16_t inf_f16 = 0x7c00;
    constexpr uint16_t nan_f16 = 0x7e00;

    for (const auto& [offset, bits] : {std::pair<size_t, uint16_t>{0, inf_f16}, {2, nan_f16}}) {
        auto malformed = valid;
        std::memcpy(malformed.data() + offset, &bits, sizeof(bits));
        EXPECT_ANY_THROW(frontend_dequant(GGUF_TYPE_Q4_K, malformed, kRows, kCols));
    }
}

// The faithful per-row K-quant dequant used as the Q8_0_C requant source must match ggml's
// to_float almost exactly (f16 super-scale widening only): assert tight agreement with the ggml
// reference. A loose result here means a byte-layout/index bug (which would silently corrupt the
// requant of token_embd/output/Q6_K/Q5_K and break swap-vs-master parity).
struct FaithfulCase {
    const char* stem;
    uint32_t type;
    void (*dq)(const uint8_t*, size_t, float*);
};

class FaithfulDequantVsGGML : public ::testing::TestWithParam<FaithfulCase> {};

TEST_P(FaithfulDequantVsGGML, MatchesGgmlToFloat) {
    const FaithfulCase c = GetParam();
    const auto qbytes = load_npy<uint8_t>(std::string(c.stem) + "_qbytes");
    const auto ref = load_npy<float>(std::string(c.stem) + "_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);
    // Reference is row-major [kRows, kCols]; the raw blocks for a row are contiguous.
    const size_t bytes_per_row = qbytes.size() / kRows;
    std::vector<float> ours(kRows * kCols);
    for (size_t r = 0; r < kRows; ++r) {
        c.dq(qbytes.data() + r * bytes_per_row, kCols, ours.data() + r * kCols);
    }
    EXPECT_LE(max_abs_diff(ours, ref), 3e-3f) << c.stem << ": faithful per-row dequant diverges from ggml to_float";
}

INSTANTIATE_TEST_SUITE_P(
    FaithfulKQuant,
    FaithfulDequantVsGGML,
    ::testing::Values(FaithfulCase{"q4_k", GGUF_TYPE_Q4_K, ov::frontend::gguf::dequant_row_q4_k_f32_for_test},
                      FaithfulCase{"q5_k", GGUF_TYPE_Q5_K, ov::frontend::gguf::dequant_row_q5_k_f32_for_test},
                      FaithfulCase{"q6_k", GGUF_TYPE_Q6_K, ov::frontend::gguf::dequant_row_q6_k_f32_for_test}),
    [](const ::testing::TestParamInfo<FaithfulCase>& i) {
        return std::string(i.param.stem);
    });

INSTANTIATE_TEST_SUITE_P(AllQuantTypes,
                         DequantVsGGML,
                         ::testing::Values(DeqCase{"q4_0", GGUF_TYPE_Q4_0, kTolFaithful},
                                           DeqCase{"q4_1", GGUF_TYPE_Q4_1, kTolFaithful},
                                           DeqCase{"q5_0", GGUF_TYPE_Q5_0, kTolFaithful},
                                           DeqCase{"q5_1", GGUF_TYPE_Q5_1, kTolFaithful},
                                           DeqCase{"q8_0", GGUF_TYPE_Q8_0, kTolFaithful},
                                           DeqCase{"q8_1", GGUF_TYPE_Q8_1, kTolExact},
                                           DeqCase{"q2_k", GGUF_TYPE_Q2_K, kTolFaithful},
                                           DeqCase{"q3_k", GGUF_TYPE_Q3_K, kTolFaithful},
                                           DeqCase{"q4_k", GGUF_TYPE_Q4_K, kTolU4Requant},
                                           DeqCase{"q5_k", GGUF_TYPE_Q5_K, kTolRequant},
                                           DeqCase{"q6_k", GGUF_TYPE_Q6_K, kTolRequant},
                                           // Q2_0 is bit-exact: both sides compute (code - 1) * d
                                           // from the same f16 scale, and the u8 zero-point of 1 is
                                           // represented exactly, so no dequant noise is introduced.
                                           DeqCase{"q2_0", GGUF_TYPE_Q2_0, kTolExact},
                                           // Q1_0 is bit-exact: both sides compute bit ? +d : -d from
                                           // the same f16 scale, with no zero-point rounding at all.
                                           DeqCase{"q1_0", GGUF_TYPE_Q1_0, kTolExact}),
                         [](const ::testing::TestParamInfo<DeqCase>& i) {
                             return std::string(i.param.stem);
                         });

// Gemma4 global attention can reuse one quantized tensor for both K and V.
// Constructing its first decompression graph must not reshape the shared scales.
TEST(GGUFDequant, SharedExtractedWeightsKeepGroupLayout) {
    constexpr size_t rows = 4, cols = 256, groups = cols / 32;
    for (const auto type : {GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K}) {
        const bool asymmetric = type == GGUF_TYPE_Q4_K;
        const std::string stem = asymmetric ? "q4_k" : "q4_0";
        SCOPED_TRACE(stem);
        const auto bytes = load_npy<uint8_t>(stem + "_qbytes");
        const auto reference = load_npy<float>(stem + "_deq");
        GgufTensor source{};
        source.type = type;
        source.ndim = 2;
        source.dim[0] = cols;
        source.dim[1] = rows;
        source.num_weights = rows * cols;
        source.bsize = bytes.size();
        source.weights_data = bytes.data();
        WeightTensors tensors;
        tensors.weight =
            ov::Tensor(asymmetric ? ov::element::u32 : ov::element::i4, {rows, asymmetric ? cols / 8 : cols});
        tensors.scales = ov::Tensor(ov::element::f16, {rows, groups});
        if (asymmetric) {
            tensors.zero_point = ov::Tensor(ov::element::f16, {rows, groups});
            gguf_fill_asym(source, tensors.weight, tensors.scales, tensors.zero_point);
        } else {
            gguf_fill_sym(source, tensors.weight, tensors.scales);
        }
        const auto first = make_weight_node(tensors, type, "key");
        EXPECT_EQ(tensors.scales.get_shape(), (ov::Shape{rows, groups}));
        if (asymmetric) {
            EXPECT_EQ(tensors.zero_point.get_shape(), (ov::Shape{rows, groups}));
        }
        const auto second = make_weight_node(tensors, type, "value");
        const auto a = eval_as_f32(first);
        const auto b = eval_as_f32(second);
        ASSERT_EQ(a.size(), reference.size());
        ASSERT_EQ(b.size(), reference.size());
        for (size_t i = 0; i < reference.size(); ++i) {
            EXPECT_EQ(a[i], b[i]);
            EXPECT_NEAR(a[i], reference[i], 3e-3f);
        }
    }
}

// Matmul weights take an integer zero-point, so Q4_1 is requantized to u4 and Q5_K to u8.
// The public byte-level entry point decodes Q4_1 and sends Q5_K through Q8_0_C instead, so
// fill as the loader does. Each 32-value group lands on a grid spanning [min(lo,0), max(hi,0)]:
// every value is within half a step of it, up to f16 scale rounding.
TEST(GGUFDequant, IntegerZeroPointRequantizationTracksGgml) {
    constexpr size_t rows = kRows, cols = kCols, groups = cols / 32;
    for (const auto& [type, stem, levels] :
         {std::tuple{GGUF_TYPE_Q4_1, "q4_1", 15.f}, std::tuple{GGUF_TYPE_Q5_K, "q5_k", 255.f}}) {
        SCOPED_TRACE(stem);
        const auto bytes = load_npy<uint8_t>(std::string(stem) + "_qbytes");
        const auto reference = load_npy<float>(std::string(stem) + "_deq");
        GgufTensor source{};
        source.type = type;
        source.ndim = 2;
        source.dim[0] = cols;
        source.dim[1] = rows;
        source.num_weights = rows * cols;
        source.bsize = bytes.size();
        source.weights_data = bytes.data();
        const auto qtype = static_cast<GgufTensorType>(type);
        ASSERT_EQ(gguf_zero_point_type("blk.0.ffn_up.weight", qtype), ov::element::u8);
        const bool packed = type == GGUF_TYPE_Q4_1;
        WeightTensors tensors;
        tensors.weight = ov::Tensor(packed ? ov::element::u32 : ov::element::u8, {rows, packed ? cols / 8 : cols});
        tensors.scales = ov::Tensor(ov::element::f16, {rows, groups});
        tensors.zero_point = ov::Tensor(ov::element::u8, {rows, groups});
        gguf_fill_asym(source, tensors.weight, tensors.scales, tensors.zero_point);
        const auto values = eval_as_f32(make_weight_node(tensors, qtype, "blk.0.ffn_up.weight"));
        ASSERT_EQ(values.size(), reference.size());
        for (size_t group = 0; group < rows * groups; ++group) {
            const auto begin = reference.begin() + group * 32;
            const auto [lo, hi] = std::minmax_element(begin, begin + 32);
            const float range = std::max(*hi, 0.f) - std::min(*lo, 0.f);
            for (size_t k = group * 32; k < group * 32 + 32; ++k)
                ASSERT_LE(std::fabs(values[k] - reference[k]), range * (0.5f / levels + 1e-3f)) << "group " << group;
        }
    }
}

TEST(DequantVsGGML, Q8_1PreservesFiniteScaleRangeAndIgnoresSum) {
    const std::vector<uint16_t> scale_bits{0, 0x8000, 0xbc00, 1, 0x7bff, 0x3555};
    std::vector<uint8_t> bytes(scale_bits.size() * 36);
    std::vector<float> reference(scale_bits.size() * 32);
    for (size_t group = 0; group < scale_bits.size(); ++group) {
        std::memcpy(bytes.data() + group * 36, &scale_bits[group], sizeof(uint16_t));
        const uint16_t sum_bits = group % 2 == 0 ? 0x7c00 : 0x7e00;
        std::memcpy(bytes.data() + group * 36 + 2, &sum_bits, sizeof(sum_bits));
        const float scale = static_cast<float>(ov::float16::from_bits(scale_bits[group]));
        for (size_t k = 0; k < 32; ++k) {
            const int8_t code = k == 0 ? -128 : k == 1 ? 127 : static_cast<int8_t>(static_cast<int>(k) - 16);
            std::memcpy(bytes.data() + group * 36 + 4 + k, &code, sizeof(code));
            reference[group * 32 + k] = static_cast<float>(code) * scale;
        }
    }
    const auto values = frontend_dequant(GGUF_TYPE_Q8_1, bytes, 1, reference.size());
    ASSERT_EQ(values.size(), reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(std::isfinite(values[i]));
        EXPECT_EQ(values[i], reference[i]) << "element " << i;
    }
    const size_t rounding_trap = (scale_bits.size() - 1) * 32 + 1;
    EXPECT_NE(reference[rounding_trap], static_cast<float>(ov::float16(reference[rounding_trap])));
}

TEST(DequantVsGGML, Q8_1RejectsInvalidDimensionsAndPayloadSizes) {
    const std::vector<uint8_t> block(36);
    for (const uint64_t cols : {0, 31, 33}) {
        SCOPED_TRACE(cols);
        EXPECT_THROW(frontend_dequant(GGUF_TYPE_Q8_1, block, 1, cols), ov::Exception);
    }
    for (const size_t size : {35, 37, 40, 71, 73, 80}) {
        SCOPED_TRACE(size);
        const uint64_t rows = size < 64 ? 1 : 2;
        EXPECT_THROW(frontend_dequant(GGUF_TYPE_Q8_1, std::vector<uint8_t>(size), rows, 32), ov::Exception);
    }
}

TEST(GGUFQ5KRequant, SymmetricGroupsTrackRealGgml) {
    const auto bytes = load_npy<uint8_t>("q5_k_qbytes");
    const auto reference = load_npy<float>("q5_k_deq");
    ASSERT_EQ(reference.size(), kRows * kCols);
    GgufTensor source{};
    source.type = GGUF_TYPE_Q5_K;
    source.num_weights = reference.size();
    source.bsize = bytes.size();
    source.weights_data = bytes.data();
    constexpr size_t group_size = 32;
    WeightTensors tensors{ov::Tensor(ov::element::i8, {kRows, kCols}),
                          ov::Tensor(ov::element::f16, {kRows, kCols / group_size}),
                          {}};
    gguf_fill_q5_k_i8(source, tensors.weight, tensors.scales);
    ASSERT_FALSE(tensors.zero_point);
    const auto node = make_weight_node(tensors, GGUF_TYPE_Q5_K);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{node}, ov::ParameterVector{});
    bool has_i8 = false;
    for (const auto& op : model->get_ops()) {
        EXPECT_STRNE(op->get_type_name(), "Subtract");
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op))
            has_i8 |= constant->get_element_type() == ov::element::i8;
    }
    EXPECT_TRUE(has_i8);
    const auto expanded = eval_as_f32(node);
    ASSERT_EQ(expanded.size(), reference.size());
    for (size_t group = 0; group < reference.size() / group_size; ++group) {
        float maxabs = 0.0f;
        for (size_t k = group * group_size; k < (group + 1) * group_size; ++k)
            maxabs = std::max(maxabs, std::fabs(reference[k]));
        const ov::float16 expected_scale(maxabs / 127.0f);
        EXPECT_EQ(tensors.scales.data<ov::float16>()[group].to_bits(), expected_scale.to_bits());
        const float scale = static_cast<float>(expected_scale);
        for (size_t k = group * group_size; k < (group + 1) * group_size; ++k) {
            const int code = tensors.weight.data<int8_t>()[k];
            EXPECT_GE(code, -127);
            EXPECT_LE(code, 127);
            const float expected_code = scale > 0.0f ? std::round(reference[k] / scale) : 0.0f;
            EXPECT_EQ(code, static_cast<int>(std::clamp(expected_code, -127.0f, 127.0f)));
            EXPECT_LE(std::fabs(expanded[k] - reference[k]), scale * 0.501f + maxabs * 1e-3f + 1e-7f);
        }
    }
}

TEST(GGUFQ5KRequant, ZeroAndUnderflowGroupsStayFinite) {
    auto bytes = load_npy<uint8_t>("q5_k_qbytes");
    GgufTensor source{};
    source.type = GGUF_TYPE_Q5_K;
    source.num_weights = kRows * kCols;
    source.bsize = bytes.size();
    source.weights_data = bytes.data();
    for (bool tiny : {false, true}) {
        std::fill(bytes.begin(), bytes.end(), uint8_t{0});
        if (tiny) {
            for (size_t block = 0; block < bytes.size() / 176; ++block) {
                bytes[block * 176] = 1;
                std::fill_n(bytes.begin() + block * 176 + 4, 4, uint8_t{1});
                std::fill_n(bytes.begin() + block * 176 + 48, 128, uint8_t{0x11});
            }
        }
        constexpr size_t group_size = 32;
        WeightTensors tensors{ov::Tensor(ov::element::i8, {kRows, kCols}),
                              ov::Tensor(ov::element::f16, {kRows, kCols / group_size}),
                              {}};
        gguf_fill_q5_k_i8(source, tensors.weight, tensors.scales);
        for (size_t k = 0; k < tensors.weight.get_size(); ++k)
            EXPECT_EQ(tensors.weight.data<int8_t>()[k], 0);
        for (size_t k = 0; k < tensors.scales.get_size(); ++k)
            EXPECT_EQ(tensors.scales.data<ov::float16>()[k].to_bits(), 0);
    }
}

TEST(GGUFQ5KRequant, RejectsNonFiniteSource) {
    auto bytes = load_npy<uint8_t>("q5_k_qbytes");
    const uint16_t infinity = 0x7c00;
    std::memcpy(bytes.data(), &infinity, sizeof(infinity));
    GgufTensor source{};
    source.type = GGUF_TYPE_Q5_K;
    source.num_weights = kRows * kCols;
    source.bsize = bytes.size();
    source.weights_data = bytes.data();
    ov::Tensor weights(ov::element::i8, {kRows, kCols});
    ov::Tensor scales(ov::element::f16, {kRows, kCols / 32});
    EXPECT_THROW(gguf_fill_q5_k_i8(source, weights, scales), ov::Exception);
}
