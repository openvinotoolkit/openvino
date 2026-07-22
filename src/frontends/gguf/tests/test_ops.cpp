// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Per-op conversion tests for the GGUF frontend.
//
// Each test builds a one-op model through SingleOpBuilder (which drives
// ov::frontend::gguf::FrontEnd::convert via an in-memory SingleOpDecoder), runs it on
// CPU and checks the result against a reference computed in plain C++.  No .gguf file,
// ggml or llama.cpp is involved.

#include <algorithm>
#include <cmath>
#include <functional>

#include "op_test_utils.hpp"

using namespace ov_gguf_test;

namespace {

// ── Elementwise binary ops (two equally-shaped f32 inputs) ──────────────────────
// add / mul / sub / div share the exact same graph shape and driver, so they are
// parameterized over (op type, reference lambda) rather than duplicated.

struct BinaryCase {
    const char* name;
    const char* op_type;
    std::function<float(float, float)> ref;
};

class GGUFBinaryElementwise : public ::testing::TestWithParam<BinaryCase> {};

TEST_P(GGUFBinaryElementwise, MatchesReference) {
    const BinaryCase c = GetParam();
    auto model = SingleOpBuilder()
                     .op(c.op_type)
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    // Inputs chosen to be valid for every op (non-zero divisor for div).
    std::vector<float> a{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> b{2, 4, 5, 8, 10, 12, 14, 16};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({2, 4}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = c.ref(a[i], b[i]);
    expect_near(out, expected);
}

INSTANTIATE_TEST_SUITE_P(GGUFOps,
                         GGUFBinaryElementwise,
                         ::testing::Values(BinaryCase{"add", "GGML_OP_ADD", [](float x, float y) { return x + y; }},
                                           BinaryCase{"mul", "GGML_OP_MUL", [](float x, float y) { return x * y; }},
                                           BinaryCase{"sub", "GGML_OP_SUB", [](float x, float y) { return x - y; }},
                                           BinaryCase{"div", "GGML_OP_DIV", [](float x, float y) { return x / y; }}),
                         [](const ::testing::TestParamInfo<BinaryCase>& i) { return std::string(i.param.name); });

// ── Elementwise unary ops (single f32 input) ────────────────────────────────────
// silu / gelu(tanh) / tanh / softplus share the same one-input graph and driver.

struct UnaryCase {
    const char* name;
    const char* op_type;
    std::function<float(float)> ref;
    float atol;
};

class GGUFUnaryElementwise : public ::testing::TestWithParam<UnaryCase> {};

TEST_P(GGUFUnaryElementwise, MatchesReference) {
    const UnaryCase c = GetParam();
    auto model = SingleOpBuilder()
                     .op(c.op_type)
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x{-2, -1, -0.5f, 0, 0.5f, 1, 2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = c.ref(x[i]);
    expect_near(out, expected, c.atol);
}

// ggml GELU is the tanh approximation, but the frontend maps GGML_UNARY_OP_GELU to v7::Gelu(TANH)
// which is close enough to the exact (erf) form to check against it at 1e-3.
// Softplus uses 1e-3 to cover ARM CPU fp16 execution (small outputs where fp16 spacing ~1e-3
// dominates); on x86 fp32 the exact reference still matches comfortably within that bound.
INSTANTIATE_TEST_SUITE_P(
    GGUFOps,
    GGUFUnaryElementwise,
    ::testing::Values(
        UnaryCase{"silu", "GGML_UNARY_OP_SILU", [](float x) { return x / (1.0f + std::exp(-x)); }, 1e-4f},
        UnaryCase{"gelu",
                  "GGML_UNARY_OP_GELU",
                  [](float x) { return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))); },
                  1e-3f},
        UnaryCase{"tanh", "GGML_UNARY_OP_TANH", [](float x) { return std::tanh(x); }, 1e-4f},
        UnaryCase{"softplus",
                  "GGML_UNARY_OP_SOFTPLUS",
                  [](float x) { return std::log1p(std::exp(-std::abs(x))) + std::max(x, 0.0f); },
                  1e-3f}),
    [](const ::testing::TestParamInfo<UnaryCase>& i) { return std::string(i.param.name); });

// Scale: out = in * scale + bias (scale/bias in op-params slots 0,1).
TEST(GGUFOps, Scale) {
    const float scale = 2.5f;
    const float bias = 1.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<float>("scale", scale)
                     .attr<float>("bias", bias)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = x[i] * scale + bias;
    expect_near(out, expected);
}

// RMS norm over the last axis (eps in op-params slot 0).
TEST(GGUFOps, RmsNorm) {
    const float eps = 1e-6f;
    const size_t cols = 4;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_RMS_NORM")
                     .input("x", ov::element::f32, {2, cols})
                     .output("out", ov::element::f32, {2, cols})
                     .attr<float>("eps", eps)
                     .build();

    std::vector<float> x{1, 2, 3, 4, -1, -2, -3, -4};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, cols}, x)}});

    std::vector<float> expected(x.size());
    for (size_t r = 0; r < 2; ++r) {
        float ss = 0.f;
        for (size_t c = 0; c < cols; ++c)
            ss += x[r * cols + c] * x[r * cols + c];
        float scale = 1.0f / std::sqrt(ss / cols + eps);
        for (size_t c = 0; c < cols; ++c)
            expected[r * cols + c] = x[r * cols + c] * scale;
    }
    expect_near(out, expected, 1e-4f);
}

// RMS norm rank-4 multi-token: each (token) row over the last axis independently. Mirrors OLMoE's
// QK-norm input layout [1,1,tok,width].
TEST(GGUFOps, RmsNormRank4MultiTok) {
    const float eps = 1e-6f;
    const size_t T = 3, W = 8;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_RMS_NORM")
                     .input("x", ov::element::f32, {1, 1, T, W})
                     .output("out", ov::element::f32, {1, 1, T, W})
                     .attr<float>("eps", eps)
                     .build();

    std::vector<float> x(T * W);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.3f * static_cast<float>((int)i - 5) + ((i % 2) ? 0.7f : -0.4f);
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, T, W}, x)}});

    std::vector<float> expected(x.size());
    for (size_t t = 0; t < T; ++t) {
        float ss = 0.f;
        for (size_t c = 0; c < W; ++c)
            ss += x[t * W + c] * x[t * W + c];
        float scale = 1.0f / std::sqrt(ss / W + eps);
        for (size_t c = 0; c < W; ++c)
            expected[t * W + c] = x[t * W + c] * scale;
    }
    expect_near(out, expected, 1e-4f);
}

// Softmax along last axis (single-input form; shape rank 3 as the translator expects).
TEST(GGUFOps, SoftMax) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOFT_MAX")
                     .input("x", ov::element::f32, {1, 2, 4})
                     .output("out", ov::element::f32, {1, 2, 4})
                     .attr<float>("scale", 1.0f)
                     .attr<float>("max_bias", 0.0f)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 4, 3, 2, 1};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t r = 0; r < 2; ++r) {
        float mx = -1e30f;
        for (size_t c = 0; c < 4; ++c)
            mx = std::max(mx, x[r * 4 + c]);
        float sum = 0.f;
        for (size_t c = 0; c < 4; ++c)
            sum += std::exp(x[r * 4 + c] - mx);
        for (size_t c = 0; c < 4; ++c)
            expected[r * 4 + c] = std::exp(x[r * 4 + c] - mx) / sum;
    }
    expect_near(out, expected, 1e-5f);
}

// Softmax with ALiBi (max_bias > 0): per-head slope applied to the mask.
TEST(GGUFOps, SoftMaxAlibi) {
    const uint32_t n_head = 6;
    const size_t T = 2, Kd = 4;
    const float scale = 1.0f, max_bias = 8.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOFT_MAX")
                     .input("x", ov::element::f32, {n_head, T, Kd})
                     .input("mask", ov::element::f32, {1, T, Kd})
                     .output("out", ov::element::f32, {n_head, T, Kd})
                     .attr<float>("scale", scale)
                     .attr<float>("max_bias", max_bias)
                     .build();

    std::vector<float> x(n_head * T * Kd);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.1f * static_cast<float>(i) - 2.0f;
    std::vector<float> mask{0, 0, -1e9f, -1e9f, 0, 0, 0, -1e9f};
    auto out = run_on_cpu(model,
                          {{"x", make_f32_tensor({n_head, T, Kd}, x)}, {"mask", make_f32_tensor({1, T, Kd}, mask)}});

    const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(n_head)));
    const float m0 = std::pow(2.0f, -max_bias / n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
    std::vector<float> expected(x.size());
    for (uint32_t h = 0; h < n_head; ++h) {
        float slope = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1)) : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
        for (size_t t = 0; t < T; ++t) {
            float mx = -1e30f;
            std::vector<float> z(Kd);
            for (size_t k = 0; k < Kd; ++k) {
                z[k] = scale * x[(h * T + t) * Kd + k] + slope * mask[t * Kd + k];
                mx = std::max(mx, z[k]);
            }
            float sum = 0.f;
            for (size_t k = 0; k < Kd; ++k) {
                z[k] = std::exp(z[k] - mx);
                sum += z[k];
            }
            for (size_t k = 0; k < Kd; ++k)
                expected[(h * T + t) * Kd + k] = z[k] / sum;
        }
    }
    expect_near(out, expected, 1e-4f);
}

// SwiGLU GLU: split last axis in half -> silu(a) * b.
TEST(GGUFOps, SwiGLU) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU")
                     .input("x", ov::element::f32, {2, 8})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<bool>("swapped", false)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 1, 1, 1, 1};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 8}, x)}});

    std::vector<float> expected(8);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            float a = x[r * 8 + c];
            float b = x[r * 8 + 4 + c];
            float silu = a / (1.0f + std::exp(-a));
            expected[r * 4 + c] = silu * b;
        }
    }
    expect_near(out, expected, 1e-4f);
}

// Clamp: elementwise clamp to [min, max] (bounds from typed attributes).
TEST(GGUFOps, Clamp) {
    const float min = -1.0f, max = 2.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CLAMP")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<float>("clamp_min", min)
                     .attr<float>("clamp_max", max)
                     .build();

    std::vector<float> x{-3, -1.5f, -0.5f, 0, 0.5f, 1.5f, 2.5f, 5};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = std::min(std::max(x[i], min), max);
    expect_near(out, expected);
}

// LayerNorm over the last axis: (x - mean) / sqrt(var + eps).
TEST(GGUFOps, Norm) {
    const float eps = 1e-5f;
    const size_t cols = 4;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_NORM")
                     .input("x", ov::element::f32, {2, cols})
                     .output("out", ov::element::f32, {2, cols})
                     .attr<float>("eps", eps)
                     .build();

    std::vector<float> x{1, 2, 3, 4, -2, 0, 2, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, cols}, x)}});

    std::vector<float> expected(x.size());
    for (size_t r = 0; r < 2; ++r) {
        float mean = 0.f;
        for (size_t c = 0; c < cols; ++c)
            mean += x[r * cols + c];
        mean /= cols;
        float var = 0.f;
        for (size_t c = 0; c < cols; ++c)
            var += (x[r * cols + c] - mean) * (x[r * cols + c] - mean);
        var /= cols;
        float inv = 1.0f / std::sqrt(var + eps);
        for (size_t c = 0; c < cols; ++c)
            expected[r * cols + c] = (x[r * cols + c] - mean) * inv;
    }
    expect_near(out, expected, 1e-4f);
}

// L2 norm over the last axis: x / max(sqrt(sum(x^2)), eps).
TEST(GGUFOps, L2Norm) {
    const float eps = 1e-12f;
    const size_t cols = 4;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_L2_NORM")
                     .input("x", ov::element::f32, {2, cols})
                     .output("out", ov::element::f32, {2, cols})
                     .attr<float>("eps", eps)
                     .build();

    std::vector<float> x{1, 2, 3, 4, -1, -2, -3, -4};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, cols}, x)}});

    std::vector<float> expected(x.size());
    for (size_t r = 0; r < 2; ++r) {
        float ss = 0.f;
        for (size_t c = 0; c < cols; ++c)
            ss += x[r * cols + c] * x[r * cols + c];
        float denom = std::max(std::sqrt(ss), eps);
        for (size_t c = 0; c < cols; ++c)
            expected[r * cols + c] = x[r * cols + c] / denom;
    }
    expect_near(out, expected, 1e-4f);
}

// SumRows: reduce-sum over the last axis, keeping rank.
TEST(GGUFOps, SumRows) {
    const size_t cols = 4;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SUM_ROWS")
                     .input("x", ov::element::f32, {2, cols})
                     .output("out", ov::element::f32, {2, 1})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, cols}, x)}});

    std::vector<float> expected{1 + 2 + 3 + 4, 5 + 6 + 7 + 8};
    expect_near(out, expected);
}

// Concat: join two inputs along a ggml axis (here ggml dim 0 == innermost == OV last axis).
TEST(GGUFOps, Concat) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CONCAT")
                     .input("a", ov::element::f32, {2, 3})
                     .input("b", ov::element::f32, {2, 2})
                     .output("out", ov::element::f32, {2, 5})
                     .attr<int>("concat_axis", 0)
                     .build();

    std::vector<float> a{1, 2, 3, 4, 5, 6};
    std::vector<float> b{7, 8, 9, 10};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 3}, a)}, {"b", make_f32_tensor({2, 2}, b)}});

    // Row-wise concat on the last axis: [1,2,3 | 7,8], [4,5,6 | 9,10].
    std::vector<float> expected{1, 2, 3, 7, 8, 4, 5, 6, 9, 10};
    expect_near(out, expected);
}

// Argsort: indices that sort the last axis; output is integer indices, checked manually.
TEST(GGUFOps, Argsort) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ARGSORT")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::i32, {1, 1, 2, 4})
                     .attr<int>("sort_order", 0)  // ascending
                     .build();

    std::vector<float> x{4, 1, 3, 2, 10, 40, 20, 30};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});

    ASSERT_EQ(out.get_element_type(), ov::element::i32);
    ASSERT_EQ(out.get_size(), 8u);
    const int32_t* a = out.data<int32_t>();
    // Row 0: values 4,1,3,2 -> ascending indices 1,3,2,0. Row 1: 10,40,20,30 -> 0,2,3,1.
    std::vector<int32_t> expected{1, 3, 2, 0, 0, 2, 3, 1};
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(a[i], expected[i]) << "mismatch at index " << i;
}

// Transpose with an explicit decoder-supplied permutation (perm attribute).
TEST(GGUFOps, TransposePerm) {
    // Swap the last two axes of a [1,1,2,3] tensor -> [1,1,3,2].
    auto model = SingleOpBuilder()
                     .op("GGML_OP_TRANSPOSE")
                     .input("x", ov::element::f32, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 3, 2})
                     .attr<std::vector<int64_t>>("perm", {0, 1, 3, 2})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 3}, x)}});

    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    std::vector<float> expected{1, 4, 2, 5, 3, 6};
    expect_near(out, expected);
}

// Repeat: tile src to fill the output shape (integer multiples per axis).
TEST(GGUFOps, Repeat) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_REPEAT")
                     .input("x", ov::element::f32, {1, 1, 1, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 3})
                     .build();

    std::vector<float> x{1, 2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 3}, x)}});

    std::vector<float> expected{1, 2, 3, 1, 2, 3};
    expect_near(out, expected);
}

// AddId: gather bias rows by ids and add to input (MoE bias).
TEST(GGUFOps, AddId) {
    // input [1, n_token=2, n_used=1, n_embd=3]; bias [1,1,n_expert=3,n_embd=3]; ids [1,1,2,1].
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ADD_ID")
                     .input("x", ov::element::f32, {1, 2, 1, 3})
                     .input("bias", ov::element::f32, {1, 1, 3, 3})
                     .input("ids", ov::element::i32, {1, 1, 2, 1})
                     .output("out", ov::element::f32, {1, 2, 1, 3})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6};
    std::vector<float> bias{10, 20, 30, 40, 50, 60, 70, 80, 90};  // experts 0,1,2
    std::vector<int32_t> ids{2, 0};                               // token0->expert2, token1->expert0

    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, 2, 1});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"x", make_f32_tensor({1, 2, 1, 3}, x)},
                           {"bias", make_f32_tensor({1, 1, 3, 3}, bias)},
                           {"ids", ids_t}});

    // token0 += expert2 (70,80,90); token1 += expert0 (10,20,30)
    std::vector<float> expected{1 + 70, 2 + 80, 3 + 90, 4 + 10, 5 + 20, 6 + 30};
    expect_near(out, expected);
}

// Pad: constant (zero) pad on the last axis.
TEST(GGUFOps, Pad) {
    // Pad last axis by 1 at the end: op_params[0]=begin(dim0), [1]=end(dim0) in ggml order.
    // ggml dim0 is the innermost == OV last axis. pads_begin/end mapping in the op: pad end of last
    // axis is pads[1].
    auto model = SingleOpBuilder()
                     .op("GGML_OP_PAD")
                     .input("x", ov::element::f32, {1, 1, 2, 2})
                     .output("out", ov::element::f32, {1, 1, 2, 3})
                     .attr<std::vector<int32_t>>("pad_params", {0, 1, 0, 0, 0, 0, 0, 0})
                     .attr<bool>("pad_circular", false)
                     .build();

    std::vector<float> x{1, 2, 3, 4};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 2}, x)}});

    // Each row [a,b] -> [a,b,0]
    std::vector<float> expected{1, 2, 0, 3, 4, 0};
    expect_near(out, expected);
}

// SwiGLU-OAI (gpt-oss): clamped, alpha-scaled gated SiLU times (clamped up + 1).
TEST(GGUFOps, SwigluOai) {
    const float alpha = 1.702f, limit = 7.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU_OAI")
                     .input("x", ov::element::f32, {2, 8})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<bool>("swapped", false)
                     .attr<float>("glu_alpha", alpha)
                     .attr<float>("glu_limit", limit)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 0.5f, 1.5f, 2.5f, 9};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 8}, x)}});

    std::vector<float> expected(8);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            float g = x[r * 8 + c];
            float u = x[r * 8 + 4 + c];
            float gate = std::min(g, limit);
            float glu = gate / (1.0f + std::exp(-(gate * alpha)));
            float up = std::min(std::max(u, -limit), limit);
            expected[r * 4 + c] = glu * (up + 1.0f);
        }
    }
    expect_near(out, expected, 1e-4f);
}

// MulMatId (generic dequantized experts): per-token expert matmul.
TEST(GGUFOps, MulMatId) {
    // weights [1, n_expert=2, m=2, k=3]; activations [1, n_tokens=2, 1, k=3];
    // ids [1, 1, n_tokens=2, n_used=1]; output [1, 1, n_tokens=2, m=2] (n_used collapsed).
    const int64_t n_expert = 2, m = 2, k = 3, n_tokens = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT_ID")
                     .input("w", ov::element::f32, {1, n_expert, m, k})
                     .input("a", ov::element::f32, {1, n_tokens, 1, k})
                     .input("ids", ov::element::i32, {1, 1, n_tokens, 1})
                     .output("out", ov::element::f32, {1, 1, n_tokens, m})
                     .build();

    // expert 0 rows: [1,2,3],[4,5,6]; expert 1 rows: [7,8,9],[10,11,12]
    std::vector<float> w{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> a{1, 0, 1, 0, 1, 0};  // token0 act=[1,0,1], token1 act=[0,1,0]
    std::vector<int32_t> ids{1, 0};           // token0 -> expert1, token1 -> expert0

    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tokens, 1});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"w", make_f32_tensor({1, (size_t)n_expert, (size_t)m, (size_t)k}, w)},
                           {"a", make_f32_tensor({1, (size_t)n_tokens, 1, (size_t)k}, a)},
                           {"ids", ids_t}});

    // token0 (expert1, act [1,0,1]): row0 dot = 7+9=16, row1 dot = 10+12=22
    // token1 (expert0, act [0,1,0]): row0 dot = 2, row1 dot = 5
    std::vector<float> expected{16, 22, 2, 5};
    expect_near(out, expected, 1e-4f);
}

// MulMatId with n_used=2 (multiple experts per token): output [1, n_tokens, n_used, m].
TEST(GGUFOps, MulMatIdNused2) {
    const int64_t n_expert = 3, m = 2, k = 2, n_tokens = 2, n_used = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT_ID")
                     .input("w", ov::element::f32, {1, n_expert, m, k})
                     .input("a", ov::element::f32, {1, n_tokens, 1, k})
                     .input("ids", ov::element::i32, {1, 1, n_tokens, n_used})
                     .output("out", ov::element::f32, {1, n_tokens, n_used, m})
                     .build();

    // experts: e0 rows [1,0],[0,1]; e1 rows [2,0],[0,2]; e2 rows [3,0],[0,3]
    std::vector<float> w{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3};
    std::vector<float> a{1, 2, 3, 4};       // token0 act=[1,2], token1 act=[3,4]
    std::vector<int32_t> ids{0, 1, 2, 0};    // tok0 uses experts {0,1}; tok1 uses {2,0}

    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tokens, (size_t)n_used});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"w", make_f32_tensor({1, (size_t)n_expert, (size_t)m, (size_t)k}, w)},
                           {"a", make_f32_tensor({1, (size_t)n_tokens, 1, (size_t)k}, a)},
                           {"ids", ids_t}});

    // out[token][used][row] = dot(expert(ids[token][used]).row, act[token])
    // tok0 act [1,2]: e0 -> [1*1+0*2, 0*1+1*2]=[1,2]; e1 -> [2,4]
    // tok1 act [3,4]: e2 -> [9,12]; e0 -> [3,4]
    std::vector<float> expected{1, 2, 2, 4, 9, 12, 3, 4};
    expect_near(out, expected, 1e-4f);
}

// GetRows batched gather for MoE routing weights: data [1, n_tok, n_expert, 1], indices
// [1,1,n_tok,n_used] -> per-token gather of n_used of the n_expert probs -> [1, n_tok, n_used, 1].
TEST(GGUFOps, GetRowsMoeWeights) {
    const int64_t n_tok = 2, n_expert = 4, n_used = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GET_ROWS")
                     .input("probs", ov::element::f32, {1, n_tok, n_expert, 1})
                     .input("ids", ov::element::i32, {1, 1, n_tok, n_used})
                     .output("out", ov::element::f32, {1, n_tok, n_used, 1})
                     .build();

    // token0 probs [0.1,0.2,0.3,0.4], token1 probs [0.5,0.6,0.7,0.8]
    std::vector<float> probs{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<int32_t> ids{3, 1, 0, 2};  // tok0 picks experts {3,1}; tok1 picks {0,2}

    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tok, (size_t)n_used});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"probs", make_f32_tensor({1, (size_t)n_tok, (size_t)n_expert, 1}, probs)},
                           {"ids", ids_t}});

    // tok0: probs[3],probs[1] = 0.4,0.2 ; tok1: probs[0],probs[2] = 0.5,0.7
    std::vector<float> expected{0.4f, 0.2f, 0.5f, 0.7f};
    expect_near(out, expected, 1e-5f);
}

// SsmConv: depthwise causal 1D conv. sx [1, n_s=1, d_inner=2, ncs=4], weight [1,1,2,d_conv=3]
// -> out [1, 1, n_t=2, 2].
TEST(GGUFOps, SsmConv) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SSM_CONV")
                     .input("sx", ov::element::f32, {1, 1, 2, 4})
                     .input("c", ov::element::f32, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .build();

    // channel 0 sx = [1,2,3,4], channel 1 sx = [5,6,7,8]
    std::vector<float> sx{1, 2, 3, 4, 5, 6, 7, 8};
    // channel 0 filter = [1,0,0], channel 1 filter = [0,0,1]
    std::vector<float> c{1, 0, 0, 0, 0, 1};
    auto out = run_on_cpu(model, {{"sx", make_f32_tensor({1, 1, 2, 4}, sx)}, {"c", make_f32_tensor({1, 1, 2, 3}, c)}});

    // Depthwise conv, kernel size 3, 2 output positions:
    //   ch0 (filter [1,0,0]): pos0 = 1*1+2*0+3*0 = 1 ; pos1 = 2*1+3*0+4*0 = 2
    //   ch1 (filter [0,0,1]): pos0 = 5*0+6*0+7*1 = 7 ; pos1 = 6*0+7*0+8*1 = 8
    // Output layout [1,1,n_t,d_inner]: [[ch0_p0,ch1_p0],[ch0_p1,ch1_p1]] = [1,7, 2,8]
    std::vector<float> expected{1, 7, 2, 8};
    expect_near(out, expected, 1e-4f);
}

// Softmax with attention sinks (gpt-oss): a 2nd input shaped [1,1,1,n_head] is a per-head sink,
// NOT a mask. It is appended as one hidden logit column, softmax runs over the widened last axis,
// and the sink column is dropped -- so each row sums to (1 - sink_weight) instead of 1.
TEST(GGUFOps, SoftMaxSinks) {
    // logits [1, n_head=2, T=1, Kd=3]; sinks [1,1,1,n_head=2].
    const size_t n_head = 2, T = 1, Kd = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOFT_MAX")
                     .input("x", ov::element::f32, {1, n_head, T, Kd})
                     .input("sinks", ov::element::f32, {1, 1, 1, n_head})
                     .output("out", ov::element::f32, {1, n_head, T, Kd})
                     .attr<float>("scale", 1.0f)
                     .attr<float>("max_bias", 0.0f)
                     .build();

    std::vector<float> x{1, 2, 3, 3, 2, 1};  // head0 row [1,2,3], head1 row [3,2,1]
    std::vector<float> sinks{0.5f, -0.5f};    // one sink logit per head
    ov::Tensor sinks_t(ov::element::f32, ov::Shape{1, 1, 1, n_head});
    std::copy(sinks.begin(), sinks.end(), sinks_t.data<float>());
    auto out =
        run_on_cpu(model, {{"x", make_f32_tensor({1, n_head, T, Kd}, x)}, {"sinks", sinks_t}});

    // Reference: softmax over [logits..., sink] then drop the sink column.
    std::vector<float> expected(n_head * T * Kd);
    for (size_t h = 0; h < n_head; ++h) {
        std::vector<float> z{x[h * Kd + 0], x[h * Kd + 1], x[h * Kd + 2], sinks[h]};
        float mx = *std::max_element(z.begin(), z.end());
        float sum = 0.f;
        for (float& v : z) {
            v = std::exp(v - mx);
            sum += v;
        }
        for (size_t k = 0; k < Kd; ++k)
            expected[h * Kd + k] = z[k] / sum;  // sink column (z[3]) dropped
    }
    expect_near(out, expected, 1e-5f);
}

// Softmax with BOTH a mask and attention sinks (3 inputs): mask add, then sink column.
TEST(GGUFOps, SoftMaxMaskAndSinks) {
    const size_t n_head = 2, T = 1, Kd = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOFT_MAX")
                     .input("x", ov::element::f32, {1, n_head, T, Kd})
                     .input("mask", ov::element::f32, {1, 1, T, Kd})
                     .input("sinks", ov::element::f32, {1, 1, 1, n_head})
                     .output("out", ov::element::f32, {1, n_head, T, Kd})
                     .attr<float>("scale", 1.0f)
                     .attr<float>("max_bias", 0.0f)
                     .build();

    std::vector<float> x{1, 2, 3, 3, 2, 1};
    std::vector<float> mask{0, 0, -1e9f};  // last key position masked out
    std::vector<float> sinks{0.5f, -0.5f};
    ov::Tensor sinks_t(ov::element::f32, ov::Shape{1, 1, 1, n_head});
    std::copy(sinks.begin(), sinks.end(), sinks_t.data<float>());
    auto out = run_on_cpu(model,
                          {{"x", make_f32_tensor({1, n_head, T, Kd}, x)},
                           {"mask", make_f32_tensor({1, 1, T, Kd}, mask)},
                           {"sinks", sinks_t}});

    std::vector<float> expected(n_head * T * Kd);
    for (size_t h = 0; h < n_head; ++h) {
        std::vector<float> z{x[h * Kd + 0] + mask[0], x[h * Kd + 1] + mask[1], x[h * Kd + 2] + mask[2], sinks[h]};
        float mx = *std::max_element(z.begin(), z.end());
        float sum = 0.f;
        for (float& v : z) {
            v = std::exp(v - mx);
            sum += v;
        }
        for (size_t k = 0; k < Kd; ++k)
            expected[h * Kd + k] = z[k] / sum;
    }
    expect_near(out, expected, 1e-5f);
}

// RESHAPE op_case 3: flatten-for-SET_ROWS, [F, tok, 1, 1] -> [1, F*tok?, -1, 1] with the token
// count landing on the dynamic axis (constant folds to the static output shape here). Verifies the
// (previously stubbed) case emits a correct reshape rather than throwing.
TEST(GGUFOps, ReshapeCase3) {
    // input [1,1,tok=2,F=4] -> output [1, F=4, tok=2, 1] (op_case 3 places -1 on the token axis).
    auto model = SingleOpBuilder()
                     .op("GGML_OP_RESHAPE")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 4, 2, 1})
                     .op_case(3)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});
    // Row-major reshape preserves element order.
    expect_near(out, x, 0.0f);
}

// SET_ROWS into a flattened KV-cache row (row_size taken from the dst input, not the op output):
// dst cache [1,1,ctx=3,row=2], data [1,1,n=2,row=2] written at indices {2,0}.
TEST(GGUFOps, SetRowsFlattenedCache) {
    const size_t ctx = 3, row = 2, n = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SET_ROWS")
                     .input("data", ov::element::f32, {1, 1, n, row})
                     .input("ind", ov::element::i64, {1, 1, 1, n})
                     .input("dst", ov::element::f32, {1, 1, ctx, row})
                     .output("out", ov::element::f32, {1, 1, ctx, row})
                     .build();

    std::vector<float> data{10, 11, 20, 21};  // row for idx target 2, then idx target 0
    std::vector<int64_t> ind{2, 0};
    std::vector<float> dst(ctx * row, -1.0f);

    ov::Tensor ind_t(ov::element::i64, ov::Shape{1, 1, 1, n});
    std::copy(ind.begin(), ind.end(), ind_t.data<int64_t>());
    auto out = run_on_cpu(model,
                          {{"data", make_f32_tensor({1, 1, n, row}, data)},
                           {"ind", ind_t},
                           {"dst", make_f32_tensor({1, 1, ctx, row}, dst)}});

    // data row0 -> dst[ind[0]=2] = [10,11]; data row1 -> dst[ind[1]=0] = [20,21]; idx1 untouched.
    std::vector<float> expected{20, 21, -1, -1, 10, 11};
    expect_near(out, expected, 0.0f);
}

// Reference for the frontend's make_sin_cos on the NEOX (non-imrope) path: for a single position p,
// theta[j] = p * freq_scale * factor[j], factor[0]=1, factor[j]=theta_scale^j,
// theta_scale = freq_base^(-2/n_dims); cos/sin scaled by attn_factor. ext_factor is 0 here.
static void neox_ref_cos_sin(int n_dims,
                             float freq_base,
                             float freq_scale,
                             float attn_factor,
                             int pos,
                             std::vector<float>& cos_out,
                             std::vector<float>& sin_out) {
    const int half = n_dims / 2;
    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    cos_out.resize(half);
    sin_out.resize(half);
    float factor = 1.0f;
    for (int j = 0; j < half; ++j) {
        const float theta = pos * freq_scale * factor;
        cos_out[j] = std::cos(theta) * attn_factor;
        sin_out[j] = std::sin(theta) * attn_factor;
        factor *= theta_scale;
    }
}

// RoPE NEOX partial rotary (gemma4 MatFormer): head_dim=6 but only the first n_dims=4 are rotated;
// elements [4,6) pass through unchanged. Drives the real make_sin_cos path (data + pos), so the fix
// (Slice rot-block / pass-through remainder) is exercised end-to-end through the op.
TEST(GGUFOps, RopeNeoxPartialRotary) {
    const int64_t head_dim = 6, n_rot = 4, T = 1, n_head = 1, pos = 2;
    RopeConfig cfg;
    cfg.n_dims = static_cast<int>(n_rot);
    cfg.freq_base = 10000.0f;
    cfg.freq_scale = 1.0f;
    cfg.ext_factor = 0.0f;
    cfg.attn_factor = 1.0f;
    cfg.n_ctx_orig = 4096;
    const int op_case = (1 << 16);  // NEOX mode lives in the op_case high 16 bits.

    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .op_case(op_case)
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data{1, 2, 3, 4, 100, 200};  // [d0,d1, d2,d3, p0,p1]; only d0..d3 rotate.
    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)T});
    pos_t.data<int32_t>()[0] = pos;
    auto out = run_on_cpu(model,
                          {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)},
                           {"pos", pos_t}});

    std::vector<float> cos_v, sin_v;
    neox_ref_cos_sin(static_cast<int>(n_rot), cfg.freq_base, cfg.freq_scale, cfg.attn_factor, pos, cos_v, sin_v);
    // NEOX split of the rotary block [d0,d1,d2,d3]: x0=[d0,d1], x1=[d2,d3].
    //   first  = x0*cos - x1*sin ; second = x0*sin + x1*cos ; then pass-through [100,200].
    std::vector<float> expected(head_dim);
    for (int j = 0; j < 2; ++j) {
        expected[j] = data[j] * cos_v[j] - data[j + 2] * sin_v[j];
        expected[j + 2] = data[j] * sin_v[j] + data[j + 2] * cos_v[j];
    }
    expected[4] = 100;  // pass-through remainder unchanged
    expected[5] = 200;
    expect_near(out, expected, 1e-4f);
}

// RoPE NEOX full rotary (n_dims == head_dim): no pass-through remainder (regression guard that the
// partial-rotary change did not alter the common full-head path).
TEST(GGUFOps, RopeNeoxFullRotary) {
    const int64_t head_dim = 4, T = 1, n_head = 1, pos = 3;
    RopeConfig cfg;
    cfg.n_dims = static_cast<int>(head_dim);
    cfg.freq_base = 10000.0f;
    cfg.freq_scale = 1.0f;
    cfg.ext_factor = 0.0f;
    cfg.attn_factor = 1.0f;
    cfg.n_ctx_orig = 4096;
    const int op_case = (1 << 16);

    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .op_case(op_case)
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data{1, 2, 3, 4};
    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)T});
    pos_t.data<int32_t>()[0] = pos;
    auto out = run_on_cpu(model,
                          {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)},
                           {"pos", pos_t}});

    std::vector<float> cos_v, sin_v;
    neox_ref_cos_sin(static_cast<int>(head_dim), cfg.freq_base, cfg.freq_scale, cfg.attn_factor, pos, cos_v, sin_v);
    std::vector<float> expected(head_dim);
    for (int j = 0; j < 2; ++j) {
        expected[j] = data[j] * cos_v[j] - data[j + 2] * sin_v[j];
        expected[j + 2] = data[j] * sin_v[j] + data[j + 2] * cos_v[j];
    }
    expect_near(out, expected, 1e-4f);
}

// The f4e2m1 nibble -> value lookup used by the packed MXFP4 dequant (mirrors mul_mat_id.cpp).
static const float kF4E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// MUL_MAT_ID over PACKED MXFP4 experts (gpt-oss MoE): expert weights are raw u8 rank-5 blocks
// [1, n_expert, m, k_blocks, 17] (byte0 = e8m0 scale exponent, bytes1..16 = 32 nibble-packed
// f4e2m1 quants). The dequant (nibble unpack + LUT + 2^(exp-127) scale) runs on-graph. This test
// builds real MXFP4 block bytes and checks the per-token expert matmul against a plain reference.
TEST(GGUFOps, MulMatIdMxfp4Packed) {
    const int64_t n_expert = 2, m = 2, k_blocks = 1, qk = 32, cols = k_blocks * qk, n_tokens = 2;
    const uint8_t scale_exp = 127;  // e8m0 127 -> 2^(127-127) = 1.0
    const float scale = 1.0f;

    // Choose nibble codes per (expert,row,element). Keep it simple: each row's first element is a
    // distinct nonzero code, the rest zero, so the matmul isolates one activation lane.
    // codes[e][row][col] in [0,15].
    auto code = [&](int e, int row, int col) -> uint8_t {
        if (col != 0) {
            return 0;  // f4 value 0.0
        }
        // expert0 row0 -> 2 (=1.0), row1 -> 4 (=2.0); expert1 row0 -> 5 (=3.0), row1 -> 6 (=4.0)
        static const uint8_t tbl[2][2] = {{2, 4}, {5, 6}};
        return tbl[e][row];
    };

    // Pack into [n_expert, m, k_blocks, 17] u8 (rank-5 with leading 1).
    const size_t block_bytes = 17;
    std::vector<uint8_t> packed(n_expert * m * k_blocks * block_bytes, 0);
    for (int e = 0; e < n_expert; ++e) {
        for (int row = 0; row < m; ++row) {
            size_t base = ((e * m + row) * k_blocks + 0) * block_bytes;
            packed[base + 0] = scale_exp;
            // 16 quant bytes, each low nibble = element 2*b, high nibble = element 2*b+1.
            for (int b = 0; b < 16; ++b) {
                uint8_t lo = code(e, row, 2 * b);
                uint8_t hi = code(e, row, 2 * b + 1);
                packed[base + 1 + b] = static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
            }
        }
    }

    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT_ID")
                     .input("w", ov::element::u8, {1, n_expert, m, k_blocks, block_bytes})
                     .input("a", ov::element::f32, {1, n_tokens, 1, cols})
                     .input("ids", ov::element::i32, {1, 1, n_tokens, 1})
                     .output("out", ov::element::f32, {1, 1, n_tokens, m})
                     .build();

    // activations: token0 = e0 pattern, token1 = e1 pattern; only lane 0 is nonzero.
    std::vector<float> a(n_tokens * cols, 0.0f);
    a[0 * cols + 0] = 1.0f;  // token0 act lane0 = 1
    a[1 * cols + 0] = 1.0f;  // token1 act lane0 = 1
    std::vector<int32_t> ids{1, 0};  // token0 -> expert1, token1 -> expert0

    ov::Tensor w_t(ov::element::u8, ov::Shape{1, (size_t)n_expert, (size_t)m, (size_t)k_blocks, block_bytes});
    std::memcpy(w_t.data<uint8_t>(), packed.data(), packed.size());
    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tokens, 1});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"w", w_t},
                           {"a", make_f32_tensor({1, (size_t)n_tokens, 1, (size_t)cols}, a)},
                           {"ids", ids_t}});

    // Reference: out[token][row] = sum_col dequant(expert=ids[token], row, col) * act[token][col].
    auto dequant = [&](int e, int row, int col) -> float {
        return kF4E2M1[code(e, row, col)] * scale;
    };
    std::vector<float> expected(n_tokens * m);
    for (int t = 0; t < n_tokens; ++t) {
        int e = ids[t];
        for (int row = 0; row < m; ++row) {
            float acc = 0.f;
            for (int col = 0; col < cols; ++col)
                acc += dequant(e, row, col) * a[t * cols + col];
            expected[t * m + row] = acc;
        }
    }
    // token0 (expert1): row0=3.0*1=3, row1=4.0*1=4 ; token1 (expert0): row0=1.0, row1=2.0
    expect_near(out, expected, 1e-4f);
}

// MulMat: batched matmul A @ B^T. The translator binds B=input(0), A=input(1) and emits
// MatMul(A, B, transpose_b=true), so out[t][n] = sum_k a[t][k] * b[n][k].
TEST(GGUFOps, MulMat) {
    // b (weight) [1,1,N=2,K=3]; a (activation) [1,1,T=2,K=3]; out [1,1,T=2,N=2].
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT")
                     .input("b", ov::element::f32, {1, 1, 2, 3})
                     .input("a", ov::element::f32, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .build();

    std::vector<float> b{1, 2, 3, 4, 5, 6};  // rows n0=[1,2,3], n1=[4,5,6]
    std::vector<float> a{1, 0, 1, 0, 1, 0};  // tokens t0=[1,0,1], t1=[0,1,0]
    auto out = run_on_cpu(model, {{"b", make_f32_tensor({1, 1, 2, 3}, b)}, {"a", make_f32_tensor({1, 1, 2, 3}, a)}});

    // t0=[1,0,1]: n0=1+3=4, n1=4+6=10 ; t1=[0,1,0]: n0=2, n1=5
    std::vector<float> expected{4, 10, 2, 5};
    expect_near(out, expected, 1e-4f);
}

// FlashAttnExt: scaled-dot-product attention. Inputs q,k,v,mask; output softmax(q@k^T*scale + mask)@v
// transposed to [1,T,H,D]. The op runs the SDPA in fp16 internally, so the tolerance is fp16-scale.
TEST(GGUFOps, FlashAttnExt) {
    // q [1,H=1,T=2,D=2]; k,v [1,H=1,Tk=2,D=2]; mask [1,1,T=2,Tk=2]. scale = 1.0.
    const float scale = 1.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_FLASH_ATTN_EXT")
                     .input("q", ov::element::f32, {1, 1, 2, 2})
                     .input("k", ov::element::f32, {1, 1, 2, 2})
                     .input("v", ov::element::f32, {1, 1, 2, 2})
                     .input("mask", ov::element::f32, {1, 1, 2, 2})
                     .output("out", ov::element::f32, {1, 2, 1, 2})
                     .attr<float>("scale", scale)
                     .build();

    std::vector<float> q{1, 0, 0, 1};
    std::vector<float> k{1, 0, 0, 1};
    std::vector<float> v{1, 2, 3, 4};
    std::vector<float> mask{0, 0, 0, 0};
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({1, 1, 2, 2}, q)},
                           {"k", make_f32_tensor({1, 1, 2, 2}, k)},
                           {"v", make_f32_tensor({1, 1, 2, 2}, v)},
                           {"mask", make_f32_tensor({1, 1, 2, 2}, mask)}});

    // Reference SDPA (H=1). scores[t][s] = scale * dot(q[t], k[s]) + mask[t][s]; softmax over s;
    // out[t][d] = sum_s p[t][s] * v[s][d]. Output layout [1,T,1,D] == flat over (t,d).
    const int T = 2, Tk = 2, D = 2;
    std::vector<float> expected(T * D);
    for (int t = 0; t < T; ++t) {
        std::vector<float> z(Tk);
        float mx = -1e30f;
        for (int s = 0; s < Tk; ++s) {
            float dot = 0.f;
            for (int d = 0; d < D; ++d)
                dot += q[t * D + d] * k[s * D + d];
            z[s] = scale * dot + mask[t * Tk + s];
            mx = std::max(mx, z[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < Tk; ++s) {
            z[s] = std::exp(z[s] - mx);
            sum += z[s];
        }
        for (int d = 0; d < D; ++d) {
            float acc = 0.f;
            for (int s = 0; s < Tk; ++s)
                acc += (z[s] / sum) * v[s * D + d];
            expected[t * D + d] = acc;
        }
    }
    expect_near(out, expected, 2e-2f);  // fp16 SDPA
}

// GatedDeltaNet (qwen3next linear attention) as a recurrent scan. Minimal scalar case
// B=H=S=1, T=2 exercises the full per-token gated delta update + output packing
// [attn_t0, attn_t1, final_state].
TEST(GGUFOps, GatedDeltaNet) {
    const int64_t B = 1, T = 2, H = 1, S = 1;
    auto shp = ov::PartialShape{B, T, H, S};
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .input("q", ov::element::f32, shp)
                     .input("k", ov::element::f32, shp)
                     .input("v", ov::element::f32, shp)
                     .input("g", ov::element::f32, shp)
                     .input("beta", ov::element::f32, shp)
                     .input("state", ov::element::f32, {1, 1, 1, 1})
                     .output("out", ov::element::f32, {1, 1, T * B + S * B, S * H})
                     .build();

    // Per-token scalars (see reference math below). g2 = ln(2) so exp(g)=2.
    const float ln2 = std::log(2.0f);
    std::vector<float> q{2, 1}, k{1, 2}, v{3, 1}, g{0, ln2}, beta{0.5f, 1};
    std::vector<float> state0{0};
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({1, (size_t)T, 1, 1}, q)},
                           {"k", make_f32_tensor({1, (size_t)T, 1, 1}, k)},
                           {"v", make_f32_tensor({1, (size_t)T, 1, 1}, v)},
                           {"g", make_f32_tensor({1, (size_t)T, 1, 1}, g)},
                           {"beta", make_f32_tensor({1, (size_t)T, 1, 1}, beta)},
                           {"state", make_f32_tensor({1, 1, 1, 1}, state0)}});

    // Scalar recurrence (scale = 1/sqrt(S) = 1):
    //   decayed = state * exp(g); delta = (v - decayed*k) * beta;
    //   state' = decayed + delta*k; attn = state' * q.
    float state = state0[0];
    std::vector<float> expected;
    for (int t = 0; t < T; ++t) {
        float decayed = state * std::exp(g[t]);
        float delta = (v[t] - decayed * k[t]) * beta[t];
        state = decayed + delta * k[t];
        expected.push_back(state * q[t]);  // attn_t
    }
    expected.push_back(state);  // final state, packed after the attn outputs
    expect_near(out, expected, 1e-4f);
}

// Im2col 1D: unfold a width-KW sliding window over the innermost image axis. For IC=1, stride=1,
// no pad/dilation, image [1,2,3] with KW=2 yields the two windows [1,2] and [2,3].
TEST(GGUFOps, Im2col1D) {
    // kernel [1,1,IC=1,KW=2] (only its shape is read); image [1,N=1,1,IW=3]; out [1,1,OW=2,IC*KW=2].
    auto model = SingleOpBuilder()
                     .op("GGML_OP_IM2COL")
                     .input("kernel", ov::element::f32, {1, 1, 1, 2})
                     .input("image", ov::element::f32, {1, 1, 1, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     // {s0,s1,p0,p1,d0,d1,is_2D}: stride 1, no pad, no dilation, 1D.
                     .attr<std::vector<int32_t>>("im2col_params", {1, 0, 0, 0, 1, 0, 0})
                     .build();

    // The kernel is only read for its shape, so its Parameter is pruned during conversion and has no
    // runtime port -- feed only the image.
    std::vector<float> image{1, 2, 3};
    auto out = run_on_cpu(model, {{"image", make_f32_tensor({1, 1, 1, 3}, image)}});

    // window at ow=0 -> [img0,img1]=[1,2]; ow=1 -> [img1,img2]=[2,3].
    std::vector<float> expected{1, 2, 2, 3};
    expect_near(out, expected, 1e-4f);
}

// Cpy: a ggml copy is a dtype convert to the destination type. i32 -> f32 upcast round-trips
// the integer values exactly.
TEST(GGUFOps, Cpy) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CPY")
                     .input("x", ov::element::i32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<int32_t> x{1, 2, 3, 4, 5, 6, 7, 8};
    ov::Tensor x_t(ov::element::i32, ov::Shape{2, 4});
    std::copy(x.begin(), x.end(), x_t.data<int32_t>());
    auto out = run_on_cpu(model, {{"x", x_t}});

    std::vector<float> expected;
    expected.reserve(x.size());
    for (int32_t v : x) {
        expected.push_back(static_cast<float>(v));
    }
    expect_near(out, expected, 0.0f);
}

// Cont (op_case 1/2): after a PERMUTE/TRANSPOSE the OV tensor is already logically contiguous, so
// CONT is an identity passthrough of its input.
TEST(GGUFOps, Cont) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CONT")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .op_case(1)  // input from PERMUTE
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});
    expect_near(out, x, 0.0f);
}

// GeGLU: split the last axis in half -> gelu(a) * b, where gelu is ggml's tanh approximation.
TEST(GGUFOps, GeGLU) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_GEGLU")
                     .input("x", ov::element::f32, {2, 8})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<bool>("swapped", false)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, 1, 1, 1, 1};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 8}, x)}});

    // gelu_tanh(a) = 0.5*a*(1 + tanh(sqrt(2/pi)*(a + 0.044715*a^3)))
    auto gelu_tanh = [](float a) {
        const float pi = 3.14159265358979323846f;
        const float k = std::sqrt(2.0f / pi);
        return 0.5f * a * (1.0f + std::tanh(k * (a + 0.044715f * a * a * a)));
    };
    std::vector<float> expected(8);
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            float a = x[r * 8 + c];
            float b = x[r * 8 + 4 + c];
            expected[r * 4 + c] = gelu_tanh(a) * b;
        }
    }
    expect_near(out, expected, 1e-3f);
}

// Add1: ggml adds a (broadcast) scalar to the input. The translator is the generic 2-input Add,
// which broadcasts the [1,1] operand across the [2,4] input.
TEST(GGUFOps, Add1) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ADD1")
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {1, 1})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{5};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({1, 1}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] + b[0];
    expect_near(out, expected);
}

// Cumsum: prefix sum along ggml dim 0 = the last OV axis. Each row accumulates independently.
TEST(GGUFOps, Cumsum) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CUMSUM")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected{1, 3, 6, 10, 5, 11, 18, 26};
    expect_near(out, expected);
}

// Sqr: element-wise square.
TEST(GGUFOps, Sqr) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SQR")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{1, -2, 3, -4, 0.5f, -0.25f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = x[i] * x[i];
    expect_near(out, expected);
}

// Sqrt: element-wise square root (non-negative inputs).
TEST(GGUFOps, Sqrt) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SQRT")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{0.0f, 1.0f, 4.0f, 9.0f, 2.0f, 0.25f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = std::sqrt(x[i]);
    expect_near(out, expected);
}

// Diag: a [.,.,1,n] vector becomes a [.,.,n,n] diagonal matrix (row axis = OV axis 2).
TEST(GGUFOps, Diag) {
    const int64_t n = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIAG")
                     .input("x", ov::element::f32, {1, 1, 1, n})
                     .output("out", ov::element::f32, {1, 1, n, n})
                     .build();

    std::vector<float> x{2, 5, 7};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, static_cast<size_t>(n)}, x)}});

    std::vector<float> expected(n * n, 0.0f);
    for (int64_t i = 0; i < n; ++i)
        expected[i * n + i] = x[i];
    expect_near(out, expected);
}

// Unary Sigmoid: 1 / (1 + exp(-x)) via the 1to1 template.
TEST(GGUFOps, UnarySigmoid) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_SIGMOID")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = 1.0f / (1.0f + std::exp(-x[i]));
    expect_near(out, expected);
}

// Unary Exp: element-wise exponential (moderate inputs to stay well inside f32 range).
TEST(GGUFOps, UnaryExp) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_EXP")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = std::exp(x[i]);
    expect_near(out, expected);
}

// Unary Neg: element-wise negation.
TEST(GGUFOps, UnaryNeg) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_NEG")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{1, -2, 3, -4, 0.5f, -0.25f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = -x[i];
    expect_near(out, expected);
}

// Tri: zero out elements outside a triangular region of a square matrix. tri_type selects the region:
// 0=UPPER_DIAG (col>=row), 1=UPPER (col>row), 2=LOWER_DIAG (col<=row), 3=LOWER (col<row).
TEST(GGUFOps, TriLowerDiag) {
    const int64_t n = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_TRI")
                     .input("x", ov::element::f32, {1, 1, n, n})
                     .output("out", ov::element::f32, {1, 1, n, n})
                     .attr<int>("tri_type", 2)  // LOWER_DIAG: keep col <= row
                     .build();

    // Row-major [n,n] with distinct values.
    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, static_cast<size_t>(n), static_cast<size_t>(n)}, x)}});

    std::vector<float> expected(n * n, 0.0f);
    for (int64_t row = 0; row < n; ++row)
        for (int64_t col = 0; col < n; ++col)
            if (col <= row)
                expected[row * n + col] = x[row * n + col];
    expect_near(out, expected);
}

// Tri UPPER (strict): keep col > row.
TEST(GGUFOps, TriUpper) {
    const int64_t n = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_TRI")
                     .input("x", ov::element::f32, {1, 1, n, n})
                     .output("out", ov::element::f32, {1, 1, n, n})
                     .attr<int>("tri_type", 1)  // UPPER: keep col > row
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, static_cast<size_t>(n), static_cast<size_t>(n)}, x)}});

    std::vector<float> expected(n * n, 0.0f);
    for (int64_t row = 0; row < n; ++row)
        for (int64_t col = 0; col < n; ++col)
            if (col > row)
                expected[row * n + col] = x[row * n + col];
    expect_near(out, expected);
}

// Fill: set every element to a constant scalar; output has the input's shape.
TEST(GGUFOps, Fill) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_FILL")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .attr<float>("fill_value", -1.5f)
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6};  // ignored; only shape matters
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size(), -1.5f);
    expect_near(out, expected);
}

}  // namespace
