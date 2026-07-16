// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dequantization correctness tests with REAL ggml as the oracle.
//
// The reference data was produced offline by linking real ggml from llama.cpp
// (see tests/gen_ggml_reference.c): ggml quantizes smooth, asymmetric synthetic
// data into real GGUF-format blocks (_qbytes) and dequantizes those exact bytes
// (_deq).  The committed .npy files mean the tests need no ggml / llama.cpp at
// build or run time.
//
// Here we feed the SAME bytes through the frontend's fill + make_weight_node dequant
// subgraph and require it to match ggml's to_float output.
//
// Tolerance: ggml stores K-quant scales as f16 and the dequant subgraph runs in f16,
// so allow ~3e-3 (matching llama.cpp's MAX_QUANTIZATION_TOTAL_ERROR-class thresholds).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "quant/gguf.hpp"
#include "quant/weights.hpp"

using namespace ov::frontend::gguf;
using ov_gguf_test::load_npy;

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
// Q4_K uses an INTEGER (u8) zero-point so the CPU plugin fuses the dequant into the MatMul
// (matching the original ggml-openvino backend). The integer zp rounds min to a multiple of
// scale, so the dequant diverges from ggml's faithful to_float by up to ~0.045 per weight.
constexpr float kTolIntZp = 5e-2f;

}  // namespace

class DequantVsGGML : public ::testing::TestWithParam<DeqCase> {};

TEST_P(DequantVsGGML, MatchesGgmlToFloat) {
    const DeqCase c = GetParam();
    const auto qbytes = load_npy<uint8_t>(std::string(c.stem) + "_qbytes");
    const auto ref = load_npy<float>(std::string(c.stem) + "_deq");
    ASSERT_EQ(ref.size(), kRows * kCols);

    const auto ours = frontend_dequant(c.type, qbytes, kRows, kCols);
    ASSERT_EQ(ours.size(), ref.size());

    EXPECT_LE(max_abs_diff(ours, ref), c.tol)
        << c.stem << ": frontend dequant diverges from ggml to_float beyond tolerance";
}

INSTANTIATE_TEST_SUITE_P(AllQuantTypes,
                         DequantVsGGML,
                         ::testing::Values(DeqCase{"q4_0", GGUF_TYPE_Q4_0, kTolFaithful},
                                           DeqCase{"q4_1", GGUF_TYPE_Q4_1, kTolFaithful},
                                           DeqCase{"q5_0", GGUF_TYPE_Q5_0, kTolFaithful},
                                           DeqCase{"q5_1", GGUF_TYPE_Q5_1, kTolFaithful},
                                           DeqCase{"q8_0", GGUF_TYPE_Q8_0, kTolFaithful},
                                           DeqCase{"q2_k", GGUF_TYPE_Q2_K, kTolFaithful},
                                           DeqCase{"q3_k", GGUF_TYPE_Q3_K, kTolFaithful},
                                           DeqCase{"q4_k", GGUF_TYPE_Q4_K, kTolIntZp},
                                           DeqCase{"q5_k", GGUF_TYPE_Q5_K, kTolRequant},
                                           DeqCase{"q6_k", GGUF_TYPE_Q6_K, kTolRequant}),
                         [](const ::testing::TestParamInfo<DeqCase>& i) {
                             return std::string(i.param.stem);
                         });
