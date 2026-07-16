// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Per-op conversion tests for the GGUF frontend.
//
// Each test builds a one-op model through SingleOpBuilder (which drives
// ov::frontend::gguf::FrontEnd::convert via an in-memory SingleOpDecoder), runs it on
// CPU and checks the result against a reference computed in plain C++.  No .gguf file,
// ggml or llama.cpp is involved.

#include <cmath>

#include "op_test_utils.hpp"

using namespace ov_gguf_test;

namespace {

// Elementwise binary ops over two equally-shaped f32 inputs.

TEST(GGUFOps, Add) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ADD")
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{8, 7, 6, 5, 4, 3, 2, 1};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({2, 4}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] + b[i];
    expect_near(out, expected);
}

TEST(GGUFOps, Mul) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL")
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> b{2, 2, 2, 2, 3, 3, 3, 3};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({2, 4}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] * b[i];
    expect_near(out, expected);
}

TEST(GGUFOps, Sub) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SUB")
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> a{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> b{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({2, 4}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] - b[i];
    expect_near(out, expected);
}

TEST(GGUFOps, Div) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIV")
                     .input("a", ov::element::f32, {2, 4})
                     .input("b", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> a{10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<float> b{2, 4, 5, 8, 10, 12, 14, 16};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 4}, a)}, {"b", make_f32_tensor({2, 4}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] / b[i];
    expect_near(out, expected);
}

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

// Unary ops.
TEST(GGUFOps, Silu) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_SILU")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x{-2, -1, -0.5f, 0, 0.5f, 1, 2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = x[i] / (1.0f + std::exp(-x[i]));
    expect_near(out, expected, 1e-4f);
}

TEST(GGUFOps, Gelu) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_GELU")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x{-2, -1, -0.5f, 0, 0.5f, 1, 2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    // Exact (erf) GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] / std::sqrt(2.0f)));
    expect_near(out, expected, 1e-3f);
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

}  // namespace
