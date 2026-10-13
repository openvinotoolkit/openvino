// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Per-op conversion tests for the GGUF frontend.
//
// Each test builds a one-op model through SingleOpBuilder (which drives
// ov::frontend::gguf::FrontEnd::convert via an in-memory SingleOpDecoder), runs it on
// CPU and checks the result against a reference. No .gguf file or llama.cpp is involved
// at test time; layout-sensitive expected values are generated separately with ggml CPU.

#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>

#include "op_table.hpp"
#include "op_test_utils.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/frontend/gguf/make_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "openvino/op/eye.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/runtime/variable_state.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "utils.hpp"

using namespace ov_gguf_test;

namespace {

// A cgraph decoder retains capture-time shapes while its Parameters accept subsequent token counts.
class CapturedInputDecoder : public SingleOpDecoder {
public:
    CapturedInputDecoder(const std::shared_ptr<GgufDecoder>& decoder,
                         std::map<std::string, ov::PartialShape> captured_shapes,
                         int64_t view_offset = 0)
        : SingleOpDecoder(*std::dynamic_pointer_cast<SingleOpDecoder>(decoder)),
          m_captured_shapes(std::move(captured_shapes)),
          m_view_offset(view_offset) {}

    ov::PartialShape get_input_shape(const std::string& name) const override {
        const auto it = m_captured_shapes.find(name);
        return it == m_captured_shapes.end() ? SingleOpDecoder::get_input_shape(name) : it->second;
    }

    int64_t get_input_view_element_offset(const std::string&) const override {
        return m_view_offset;
    }

private:
    std::map<std::string, ov::PartialShape> m_captured_shapes;
    int64_t m_view_offset;
};

static std::shared_ptr<ov::Model> build_captured_op(
    const SingleOpBuilder& builder,
    const std::map<std::string, ov::PartialShape>& captured_shapes,
    int64_t view_offset = 0) {
    std::shared_ptr<GgufDecoder> decoder =
        std::make_shared<CapturedInputDecoder>(builder.decoder(), captured_shapes, view_offset);
    FrontEnd fe;
    return fe.convert(fe.load(decoder));
}

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
                         ::testing::Values(BinaryCase{"add",
                                                      "GGML_OP_ADD",
                                                      [](float x, float y) {
                                                          return x + y;
                                                      }},
                                           BinaryCase{"mul",
                                                      "GGML_OP_MUL",
                                                      [](float x, float y) {
                                                          return x * y;
                                                      }},
                                           BinaryCase{"sub",
                                                      "GGML_OP_SUB",
                                                      [](float x, float y) {
                                                          return x - y;
                                                      }},
                                           BinaryCase{"div",
                                                      "GGML_OP_DIV",
                                                      [](float x, float y) {
                                                          return x / y;
                                                      }}),
                         [](const ::testing::TestParamInfo<BinaryCase>& i) {
                             return std::string(i.param.name);
                         });

// ── Elementwise unary ops (single f32 input) ────────────────────────────────────
// Every registered GGML_UNARY_OP_* plus the elementwise GGML_OP_{LOG,SIN,COS} share the same
// one-input graph and driver, so they are parameterized over (op type, reference lambda).

struct UnaryCase {
    const char* name;
    const char* op_type;
    std::function<float(float)> ref;
    float atol;
    // Input values; empty means the default sign-spanning ramp below.  Ops with a restricted
    // domain (log) supply their own.
    std::vector<float> x{};
};

class GGUFUnaryElementwise : public ::testing::TestWithParam<UnaryCase> {};

TEST_P(GGUFUnaryElementwise, MatchesReference) {
    const UnaryCase c = GetParam();
    auto model = SingleOpBuilder()
                     .op(c.op_type)
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x = c.x.empty() ? std::vector<float>{-2, -1, -0.5f, 0, 0.5f, 1, 2, 3} : c.x;
    ASSERT_EQ(x.size(), 8u) << "UnaryCase input must match the [2,4] graph shape";
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = c.ref(x[i]);
    expect_near(out, expected, c.atol);
}

// References are the scalar ggml kernels from ggml/src/ggml-cpu/vec.h, so a translator that picks
// a different-but-plausible formula for the same name is caught (GELU vs GELU_QUICK are distinct
// approximations, not interchangeable).  GELU is checked against the exact erf form instead: the
// frontend maps it to v7::Gelu(TANH), which agrees with ggml's tanh kernel and with erf to 1e-3.
// Softplus uses 1e-3 to cover ARM CPU fp16 execution (small outputs where fp16 spacing ~1e-3
// dominates); on x86 fp32 the exact reference still matches comfortably within that bound.
INSTANTIATE_TEST_SUITE_P(GGUFOps,
                         GGUFUnaryElementwise,
                         ::testing::Values(UnaryCase{"silu",
                                                     "GGML_UNARY_OP_SILU",
                                                     [](float x) {
                                                         return x / (1.0f + std::exp(-x));
                                                     },
                                                     1e-4f},
                                           UnaryCase{"gelu",
                                                     "GGML_UNARY_OP_GELU",
                                                     [](float x) {
                                                         return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
                                                     },
                                                     1e-3f},
                                           // ggml_gelu_quick_f32: x*(1/(1+expf(-1.702f*x))) -- NOT the tanh GELU above.
                                           UnaryCase{"gelu_quick",
                                                     "GGML_UNARY_OP_GELU_QUICK",
                                                     [](float x) {
                                                         return x / (1.0f + std::exp(-1.702f * x));
                                                     },
                                                     1e-4f},
                                           UnaryCase{"tanh",
                                                     "GGML_UNARY_OP_TANH",
                                                     [](float x) {
                                                         return std::tanh(x);
                                                     },
                                                     1e-4f},
                                           UnaryCase{"relu",
                                                     "GGML_UNARY_OP_RELU",
                                                     [](float x) {
                                                         return x > 0.0f ? x : 0.0f;
                                                     },
                                                     1e-4f},
                                           // ggml_vec_elu_f32: (x > 0) ? x : expm1f(x), i.e. ELU with alpha == 1.
                                           UnaryCase{"elu",
                                                     "GGML_UNARY_OP_ELU",
                                                     [](float x) {
                                                         return x > 0.0f ? x : std::expm1(x);
                                                     },
                                                     1e-4f},
                                           UnaryCase{"sin",
                                                     "GGML_OP_SIN",
                                                     [](float x) {
                                                         return std::sin(x);
                                                     },
                                                     1e-4f},
                                           UnaryCase{"cos",
                                                     "GGML_OP_COS",
                                                     [](float x) {
                                                         return std::cos(x);
                                                     },
                                                     1e-4f},
                                           UnaryCase{"softplus",
                                                     "GGML_UNARY_OP_SOFTPLUS",
                                                     [](float x) {
                                                         return std::log1p(std::exp(-std::abs(x))) + std::max(x, 0.0f);
                                                     },
                                                     1e-3f},
                                           // Log's domain is x > 0, so this case overrides the default ramp.
                                           UnaryCase{"log",
                                                     "GGML_OP_LOG",
                                                     [](float x) {
                                                         return std::log(x);
                                                     },
                                                     1e-4f,
                                                     {0.25f, 0.5f, 1.0f, 2.0f, 3.0f, 10.0f, 100.0f, 1e-3f}}),
                         [](const ::testing::TestParamInfo<UnaryCase>& i) {
                             return std::string(i.param.name);
                         });

// Log is only defined for x > 0, so it gets its own inputs rather than the shared range above.
TEST(GGUFOps, Log) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_LOG")
                     .input("x", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> x{0.25f, 0.5f, 1, 2, 4, 10};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = std::log(x[i]);
    expect_near(out, expected, 1e-4f);
}

// get_dimensions must read the shape of the given output, not of the producing node's output 0.
// TOP_K and ARGSORT hand downstream translators output(1) of a TopK, so taking the node instead
// of the output would silently measure the values port.
TEST(GGUFOps, GetDimensionsKeepsOutputPort) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, 4});
    auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto topk = std::make_shared<ov::op::v11::TopK>(data,
                                                    k,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i32);

    auto dims = ov::frontend::gguf::get_dimensions(topk->output(1), {0});
    auto shape_of = dims->input_value(0).get_node_shared_ptr();
    ASSERT_EQ(shape_of->get_input_size(), 1u);
    EXPECT_EQ(shape_of->input_value(0).get_index(), 1u) << "get_dimensions measured the wrong port";
}

// ── Real-ggml reference data (test_data/*_ggml_{input,expected}.npy) ────────────
// The parameterized cases above encode the ggml formula by hand in C++.  These cases instead run
// the actual ggml kernel's output, captured offline from real ggml linked against
// libggml, over a [-6, 6] ramp -- so a misreading of the kernel cannot be baked into both sides.
// This is the check that distinguishes GELU_QUICK's sigmoid form from the tanh form: they differ
// by 2.2e-2 in the negative tail, far outside these tolerances.
//
// Tolerances are set by ggml's own arithmetic, not by OV's.  ggml evaluates GELU and GELU_QUICK
// through an fp16 lookup table (GGML_GELU_FP16), which costs ~2e-3 / ~3.3e-3 against the exact
// fp32 form; SILU runs in fp32 in the reference build and needs only 1e-5.
struct GgmlRefCase {
    const char* name;  // also the test_data/<name>_ggml_{input,expected}.npy stem prefix
    const char* op_type;
    float atol;
};

class GGUFUnaryVsGgml : public ::testing::TestWithParam<GgmlRefCase> {};

TEST_P(GGUFUnaryVsGgml, MatchesGgmlKernel) {
    const GgmlRefCase c = GetParam();
    const auto x = load_npy<float>(std::string(c.name) + "_ggml_input");
    const auto expected = load_npy<float>(std::string(c.name) + "_ggml_expected");
    ASSERT_EQ(x.size(), expected.size());
    ASSERT_FALSE(x.empty());

    // The captured reference is a [4, 32] ramp.
    const ov::Shape shape{4, 32};
    ASSERT_EQ(ov::shape_size(shape), x.size());

    auto model = SingleOpBuilder()
                     .op(c.op_type)
                     .input("x", ov::element::f32, shape)
                     .output("out", ov::element::f32, shape)
                     .build();
    auto out = run_on_cpu(model, {{"x", make_f32_tensor(shape, x)}});
    expect_near(out, expected, c.atol);
}

INSTANTIATE_TEST_SUITE_P(GGUFOps,
                         GGUFUnaryVsGgml,
                         ::testing::Values(GgmlRefCase{"silu", "GGML_UNARY_OP_SILU", 1e-5f},
                                           GgmlRefCase{"gelu", "GGML_UNARY_OP_GELU", 2.5e-3f},
                                           GgmlRefCase{"gelu_quick", "GGML_UNARY_OP_GELU_QUICK", 4e-3f}),
                         [](const ::testing::TestParamInfo<GgmlRefCase>& i) {
                             return std::string(i.param.name);
                         });

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
    auto out =
        run_on_cpu(model, {{"x", make_f32_tensor({n_head, T, Kd}, x)}, {"mask", make_f32_tensor({1, T, Kd}, mask)}});

    const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(n_head)));
    const float m0 = std::pow(2.0f, -max_bias / n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
    std::vector<float> expected(x.size());
    for (uint32_t h = 0; h < n_head; ++h) {
        float slope = h < n_head_log2 ? static_cast<float>(std::pow(m0, h + 1))
                                      : static_cast<float>(std::pow(m1, 2 * (h - n_head_log2) + 1));
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

TEST(GGUFOps, TopKIndexTypeComesFromOperation) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_TOP_K")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::dynamic, {1, 1, 2, 2})
                     .attr<int64_t>("k", 2)
                     .build();

    std::vector<float> x{4, 1, 3, 2, 10, 40, 20, 30};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});

    ASSERT_EQ(out.get_element_type(), ov::element::i32);
    const int32_t* a = out.data<int32_t>();
    std::vector<int32_t> expected{0, 2, 1, 3};
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(a[i], expected[i]) << "mismatch at index " << i;
}

// A pass-through VIEW must not rename its producer: that node belongs to another ggml tensor
// (here a model input), and the helper appends, so the name would compound on every such view.
TEST(GGUFOps, ViewPassThroughKeepsProducerName) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("myview", ov::element::f32, {1, 1, 2, 4})
                     .attr<ov::Shape>("input_ggml_shape", ov::Shape{1, 1, 2, 4})
                     .build();

    ASSERT_EQ(model->get_parameters().size(), 1u);
    EXPECT_EQ(model->get_parameters()[0]->get_friendly_name(), "x");
}

// The norm axis is the literal -1, so a dynamic token dim must still convert and run.
TEST(GGUFOps, NormDynamicShape) {
    const float eps = 1e-5f;
    const size_t cols = 4;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_NORM")
                     .input("x", ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(cols)})
                     .output("out", ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(cols)})
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
                     .output("out", ov::element::dynamic, {1, 1, 2, 4})
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

// PERMUTE: the plain head/token axis swap, perm {0,2,1,3}.
TEST(GGUFOps, PermuteCase1SwapsHeadAndTokenAxes) {
    // [1, tok=2, heads=3, hs=2] -> [1, heads=3, tok=2, hs=2]
    auto model = SingleOpBuilder()
                     .op("GGML_OP_PERMUTE")
                     .input("x", ov::element::f32, {1, 2, 3, 2})
                     .output("out", ov::element::f32, {1, 3, 2, 2})
                     .attr<std::vector<int64_t>>("perm", {0, 2, 1, 3})
                     .build();

    std::vector<float> x(12);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i);
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 3, 2}, x)}});

    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 3, 2, 2}));
    // out[0,h,t,d] == x[0,t,h,d]; with x flat == index, x[0,t,h,d] = ((t*3)+h)*2+d.
    std::vector<float> expected;
    for (int64_t h = 0; h < 3; ++h)
        for (int64_t t = 0; t < 2; ++t)
            for (int64_t d = 0; d < 2; ++d)
                expected.push_back(static_cast<float>((t * 3 + h) * 2 + d));
    expect_near(out, expected, 0.0f);
}

// A VIEW with no descriptor is a pure reinterpretation of the same buffer: a pass-through.
TEST(GGUFOps, ViewDefaultIsPassThrough) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 4})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});
    expect_near(out, x, 0.0f);
}

// VIEW slice descriptor: the decoder resolves ggml's ne/nb/offset into a {axis, start, len} slice plus the
// view's own output layout.  Here it selects columns [1, 3) of a [1,1,2,4] source.
TEST(GGUFOps, ViewCase3SlicesAndReshapes) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .attr<ov::Shape>("input_ggml_shape", ov::Shape{1, 1, 2, 4})
                     .attr<std::vector<int64_t>>("view_slice", {3, 1, 2})
                     .attr<std::vector<int64_t>>("view_reshape", {1, 1, 2, 2})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});

    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 1, 2, 2}));
    std::vector<float> expected{2, 3, 6, 7};
    expect_near(out, expected, 0.0f);
}

// A VIEW slice also has to restore the source's original ggml shape when the OV input arrives at
// a different rank (a preceding op already reshaped it), otherwise the slice lands on the wrong
// axis.  Same slice as above, but the input is presented flat.
TEST(GGUFOps, ViewCase3RestoresGgmlShapeBeforeSlicing) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .attr<ov::Shape>("input_ggml_shape", ov::Shape{1, 1, 2, 4})
                     .attr<std::vector<int64_t>>("view_slice", {3, 1, 2})
                     .attr<std::vector<int64_t>>("view_reshape", {1, 1, 2, 2})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 1, 2, 2}));
    std::vector<float> expected{2, 3, 6, 7};
    expect_near(out, expected, 0.0f);
}

TEST(GGUFOps, ViewInputRequiresDirectionalShapeMatch) {
    using namespace ov::frontend::gguf;
    auto process = [](const ov::PartialShape& expected, const ov::PartialShape& actual) {
        auto decoder = SingleOpBuilder()
                           .op("GGML_OP_VIEW")
                           .input("x", ov::element::f32, expected)
                           .output("out", ov::element::f32, expected)
                           .attr<int64_t>("view_offset", 1)
                           .decoder();
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, actual);
        input->set_friendly_name("x");
        input->output(0).set_names({"x"});
        auto tensors = std::make_shared<TensorMap>();
        (*tensors)["x"] = input;
        NodeContext context(decoder, tensors);
        return std::make_pair(input, process_view_input(context, 0, 2));
    };

    const auto [unresolved_input, unresolved] = process(ov::PartialShape{1, 3, 2, 4}, ov::PartialShape{1, -1, 2, 4});
    auto slice = ov::as_type_ptr<ov::op::v8::Slice>(unresolved.get_node_shared_ptr());
    ASSERT_NE(slice, nullptr);
    ASSERT_EQ(unresolved.get_partial_shape(), (ov::PartialShape{1, -1, 2, 2}));
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(unresolved)},
                                             ov::ParameterVector{unresolved_input});
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 3, 2, 4}, data)}});
    expect_near(out, {1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22}, 0.0f);

    for (const auto& actual : {ov::PartialShape{1, -1, 2, 4}, ov::PartialShape{1, 3, 2, 4}}) {
        const auto [input, materialized] = process(ov::PartialShape{1, -1, 2, 4}, actual);
        EXPECT_EQ(materialized, input);
    }
}

// TopK: indices of the k largest values per row, with k supplied by the operation descriptor.
//
// ggml's own kernel deliberately swaps the first two result slots ("emphasize that the order is not
// important", ops.cpp), so only the SET of returned indices is contractual -- the test compares
// sorted index sets rather than positions.
TEST(GGUFOps, TopK) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_TOP_K")
                     .input("x", ov::element::f32, {1, 1, 2, 5})
                     .output("out", ov::element::i32, {1, 1, 2, 3})
                     .attr<int64_t>("k", 3)
                     .build();

    std::vector<float> x{1, 9, 3, 7, 5, 50, 10, 40, 20, 30};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 5}, x)}});

    ASSERT_EQ(out.get_element_type(), ov::element::i32);
    ASSERT_EQ(out.get_size(), 6u);
    const int32_t* a = out.data<int32_t>();
    // Row 0: values 1,9,3,7,5 -> top 3 are 9,7,5 at indices 1,3,4.
    // Row 1: values 50,10,40,20,30 -> top 3 are 50,40,30 at indices 0,2,4.
    std::vector<int32_t> row0(a, a + 3), row1(a + 3, a + 6);
    std::sort(row0.begin(), row0.end());
    std::sort(row1.begin(), row1.end());
    EXPECT_EQ(row0, (std::vector<int32_t>{1, 3, 4}));
    EXPECT_EQ(row1, (std::vector<int32_t>{0, 2, 4}));
}

TEST(GGUFOps, TopKExplicitKInfersOutputShape) {
    const std::vector<float> values{1, 9, 3, 7, 5, 50, 10, 40, 20, 30};
    // Sorted index sets from ggml_top_k on llama.cpp CPU, revision 03fa73cb27.
    const std::vector<std::vector<int32_t>> expected_one{{1}, {0}};
    const std::vector<std::vector<int32_t>> expected_three{{1, 3, 4}, {0, 2, 4}};
    for (int64_t k : {1, 3}) {
        SCOPED_TRACE(k);
        auto model = SingleOpBuilder()
                         .op("GGML_OP_TOP_K")
                         .input("x", ov::element::f32, {1, 1, -1, 5})
                         .output("out", ov::element::i32, ov::PartialShape::dynamic())
                         .attr<int64_t>("k", k)
                         .build();
        EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{1, 1, -1, k}));
        const auto& expected = k == 1 ? expected_one : expected_three;
        for (size_t rows : {1u, 2u}) {
            SCOPED_TRACE(rows);
            const std::vector<float> data(values.begin(), values.begin() + rows * 5);
            auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, rows, 5}, data)}});
            ASSERT_EQ(out.get_shape(), (ov::Shape{1, 1, rows, static_cast<size_t>(k)}));
            const auto* indices = out.data<int32_t>();
            for (size_t row = 0; row < rows; ++row) {
                std::vector<int32_t> selected(indices + row * k, indices + (row + 1) * k);
                std::sort(selected.begin(), selected.end());
                EXPECT_EQ(selected, expected[row]);
            }
        }
    }
}

TEST(GGUFOps, TopKRejectsUnknownK) {
    for (const auto& shape : {ov::PartialShape::dynamic(), ov::PartialShape{1, 1, 2, -1}}) {
        auto builder = SingleOpBuilder()
                           .op("GGML_OP_TOP_K")
                           .input("x", ov::element::f32, {1, 1, 2, 5})
                           .output("out", ov::element::i32, shape);
        EXPECT_THROW(builder.build(), ov::Exception);
    }
}

// Repeat: tile src using explicit operation repeat factors.
TEST(GGUFOps, Repeat) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_REPEAT")
                     .input("x", ov::element::f32, {1, 1, 1, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 3})
                     .attr<std::vector<int64_t>>("repeats", {1, 1, 2, 1})
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
    auto out = run_on_cpu(
        model,
        {{"x", make_f32_tensor({1, 2, 1, 3}, x)}, {"bias", make_f32_tensor({1, 1, 3, 3}, bias)}, {"ids", ids_t}});

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

TEST(GGUFOps, SwigluClampOneInputSwapped) {
    const float limit = 2.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU_CLAMP")
                     .input("x", ov::element::f32, {1, 8})
                     .output("out", ov::element::f32, {1, 4})
                     .attr<bool>("swapped", true)
                     .attr<float>("glu_limit", limit)
                     .build();

    std::vector<float> x{3.0f, -4.0f, 0.5f, 1.0f, 1.0f, 2.0f, -3.0f, 4.0f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 8}, x)}});
    std::vector<float> expected(4);
    for (size_t i = 0; i < expected.size(); ++i) {
        const float gate = std::min(x[4 + i], limit);
        const float up = std::min(std::max(x[i], -limit), limit);
        expected[i] = gate / (1.0f + std::exp(-gate)) * up;
    }
    expect_near(out, expected);
}

TEST(GGUFOps, SwigluClampTwoInputs) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU_CLAMP")
                     .input("gate", ov::element::f32, {1, 3})
                     .input("up", ov::element::f32, {1, 3})
                     .output("out", ov::element::f32, {1, 3})
                     .attr<bool>("swapped", false)
                     .attr<float>("glu_limit", 3.0f)
                     .build();
    std::vector<float> gate{1.0f, 4.0f, -2.0f};
    std::vector<float> up{5.0f, -5.0f, 0.5f};
    auto out = run_on_cpu(model, {{"gate", make_f32_tensor({1, 3}, gate)}, {"up", make_f32_tensor({1, 3}, up)}});
    std::vector<float> expected(3);
    for (size_t i = 0; i < expected.size(); ++i) {
        const float g = std::min(gate[i], 3.0f);
        const float u = std::min(std::max(up[i], -3.0f), 3.0f);
        expected[i] = g / (1.0f + std::exp(-g)) * u;
    }
    expect_near(out, expected);
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
    std::vector<int32_t> ids{1, 0};          // token0 -> expert1, token1 -> expert0

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
    std::vector<float> a{1, 2, 3, 4};      // token0 act=[1,2], token1 act=[3,4]
    std::vector<int32_t> ids{0, 1, 2, 0};  // tok0 uses experts {0,1}; tok1 uses {2,0}

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

// MulMatId whose activations already carry the expanded expert axis ([n_token, n_used, k]) instead of
// a singleton -- the layout ggml uses for the ffn_down matmul, where each (token, expert) pair has its
// own activation row. Reshaping this form to one row per token would silently drop data, so the
// translator has to leave it as it is.
TEST(GGUFOps, MulMatIdActivationsPerExpert) {
    const int64_t n_expert = 3, m = 2, k = 2, n_tokens = 2, n_used = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT_ID")
                     .input("w", ov::element::f32, {1, n_expert, m, k})
                     .input("a", ov::element::f32, {1, n_tokens, n_used, k})
                     .input("ids", ov::element::i32, {1, 1, n_tokens, n_used})
                     .output("out", ov::element::f32, {1, n_tokens, n_used, m})
                     .build();

    // experts: e0 rows [1,0],[0,1]; e1 rows [2,0],[0,2]; e2 rows [3,0],[0,3]
    std::vector<float> w{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3};
    // a[token][used] -- a distinct activation row per (token, expert) pair.
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> ids{0, 1, 2, 0};  // tok0 uses experts {0,1}; tok1 uses {2,0}

    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tokens, (size_t)n_used});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out = run_on_cpu(model,
                          {{"w", make_f32_tensor({1, (size_t)n_expert, (size_t)m, (size_t)k}, w)},
                           {"a", make_f32_tensor({1, (size_t)n_tokens, (size_t)n_used, (size_t)k}, a)},
                           {"ids", ids_t}});

    // out[token][used][row] = dot(expert(ids[token][used]).row, a[token][used])
    // tok0: e0 . [1,2] = [1,2];  e1 . [3,4] = [6,8]
    // tok1: e2 . [5,6] = [15,18]; e0 . [7,8] = [7,8]
    std::vector<float> expected{1, 2, 6, 8, 15, 18, 7, 8};
    expect_near(out, expected, 1e-4f);
}

// MulMatId over an empty token axis. Every non-final chunk of a chunked prefill produces no logits, so
// its inp_out_ids gather leaves the last layer's activations with 0 rows; that empty extent must flow
// through to an empty result rather than being broadcast up to n_used (which OpenVINO rejects).
TEST(GGUFOps, MulMatIdEmptyTokens) {
    const int64_t n_expert = 3, m = 2, k = 2, n_used = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT_ID")
                     .input("w", ov::element::f32, {1, n_expert, m, k})
                     .input("a", ov::element::f32, {1, ov::Dimension::dynamic(), 1, k})
                     .input("ids", ov::element::i32, {1, 1, ov::Dimension::dynamic(), n_used})
                     .output("out", ov::element::f32, {1, ov::Dimension::dynamic(), n_used, m})
                     .build();

    std::vector<float> w{1, 0, 0, 1, 2, 0, 0, 2, 3, 0, 0, 3};
    auto out = run_on_cpu(model,
                          {{"w", make_f32_tensor({1, (size_t)n_expert, (size_t)m, (size_t)k}, w)},
                           {"a", ov::Tensor(ov::element::f32, ov::Shape{1, 0, 1, (size_t)k})},
                           {"ids", ov::Tensor(ov::element::i32, ov::Shape{1, 1, 0, (size_t)n_used})}});

    EXPECT_EQ(out.get_shape(), (ov::Shape{1, 0, (size_t)n_used, (size_t)m}));
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
    auto out =
        run_on_cpu(model, {{"probs", make_f32_tensor({1, (size_t)n_tok, (size_t)n_expert, 1}, probs)}, {"ids", ids_t}});

    // tok0: probs[3],probs[1] = 0.4,0.2 ; tok1: probs[0],probs[2] = 0.5,0.7
    std::vector<float> expected{0.4f, 0.2f, 0.5f, 0.7f};
    expect_near(out, expected, 1e-5f);
}

TEST(GGUFOps, SSMScanMamba2) {
    // Compile once with dynamic batch/token axes. Oracle: ssm_scan_oracle.c, real ggml CPU,
    // two groups, four heads, unequal head/state widths and state slot permutation {2,0}.
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SSM_SCAN")
                     .input("s", ov::element::f32, {-1, 4, 3, 8})
                     .input("x", ov::element::f32, {-1, -1, 4, 3})
                     .input("dt", ov::element::f32, {1, -1, -1, 4})
                     .input("a", ov::element::f32, {1, 1, 4, 1})
                     .input("b", ov::element::f32, {-1, -1, 2, 8})
                     .input("c", ov::element::f32, {-1, -1, 2, 8})
                     .input("ids", ov::element::i32, {1, 1, 1, -1})
                     .output("out", ov::element::f32, {1, 1, 1, -1})
                     .build();
    size_t fused = 0;
    for (const auto& node : model->get_ops())
        fused += ov::is_type<ov::op::internal::SelectiveSSM>(node);
    ASSERT_EQ(fused, 1);
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t tokens : {3, 1}) {
        const std::vector<std::string> names{"s", "x", "dt", "a", "b", "c"};
        const std::vector<ov::Shape> shapes{{3, 4, 3, 8},
                                            {2, tokens, 4, 3},
                                            {1, 2, tokens, 4},
                                            {1, 1, 4, 1},
                                            {2, tokens, 2, 8},
                                            {2, tokens, 2, 8}};
        for (int j = 0; j < 6; ++j) {
            ov::Tensor input(ov::element::f32, shapes[j]);
            for (size_t i = 0; i < input.get_size(); ++i)
                input.data<float>()[i] =
                    j == 3 ? -.1f * (i + 1) : (static_cast<int>((i * (j + 3) + j) % 29) - 14) * .07f;
            if (j == 2) {
                input.data<float>()[0] = -25.f;
                input.data<float>()[1] = 25.f;
            }
            request.set_tensor(names[j], input);
        }
        ov::Tensor ids(ov::element::i32, {1, 1, 1, 2});
        ids.data<int32_t>()[0] = 2;
        ids.data<int32_t>()[1] = 0;
        request.set_tensor("ids", ids);
        request.infer();
        const auto out = request.get_output_tensor();
        const auto expected = load_npy<float>("ssm_scan_t" + std::to_string(tokens) + "_expected");
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
        // ARM's FP32 SoftPlus approximation produces up to 8e-4 absolute error through the scan.
        // A relative tolerance alone cannot cover the resulting near-zero state values.
        expect_near(out, expected, 1e-3f);
#else
        expect_near(out, expected);
#endif
    }
}

TEST(GGUFOps, SSMScanRejectsMamba1Decay) {
    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_SSM_SCAN")
                     .input("s", ov::element::f32, {1, 4, 1, 8})
                     .input("x", ov::element::f32, {1, 2, 4, 1})
                     .input("dt", ov::element::f32, {1, 1, 2, 4})
                     .input("a", ov::element::f32, {1, 1, 4, 8})
                     .input("b", ov::element::f32, {1, 2, 1, 8})
                     .input("c", ov::element::f32, {1, 2, 1, 8})
                     .input("ids", ov::element::i32, {1, 1, 1, 1})
                     .output("out", ov::element::f32, {1, 1, 1, 40})
                     .build(),
                 ov::Exception);
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

// SsmConv at the real qwen3.5/qwen3-next conv dims (d_conv=4, n_t=5, d_inner=3): the size-2/3
// SsmConv above collapses the token axis (n_t=2) and the kernel (d_conv=3); this exercises the
// model's actual window length with n_t>1 across multiple channels. Reference comes from the
// standalone ggml-CPU oracle (ssm_conv_oracle.c), not a hand-derived conv, so it pins the OV
// GroupConvolution + transpose/reshape layout against ggml's own kernel.
TEST(GGUFOps, SsmConvRealDims) {
    const size_t d_conv = 4, d_inner = 3, n_t = 5, ncs = n_t + d_conv - 1;  // ncs=8
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SSM_CONV")
                     .input("sx", ov::element::f32, {1, 1, d_inner, ncs})
                     .input("c", ov::element::f32, {1, 1, d_inner, d_conv})
                     .output("out", ov::element::f32, {1, 1, n_t, d_inner})
                     .build();

    // sx channel-major [d_inner][ncs]: ch0 = 1..8, ch1 = 11..18, ch2 = 21..28
    std::vector<float> sx(d_inner * ncs);
    for (size_t ch = 0; ch < d_inner; ++ch)
        for (size_t j = 0; j < ncs; ++j)
            sx[ch * ncs + j] = static_cast<float>(ch * 10 + j + 1);
    // c per-channel [d_inner][d_conv]: ch0 weighted, ch1 picks window-start, ch2 picks window-end
    std::vector<float> c{0.5f, 0.25f, 0.125f, 0.0625f, 1, 0, 0, 0, 0, 0, 0, 1};
    auto out = run_on_cpu(
        model,
        {{"sx", make_f32_tensor({1, 1, d_inner, ncs}, sx)}, {"c", make_f32_tensor({1, 1, d_inner, d_conv}, c)}});

    // ggml-CPU oracle (ssm_conv_oracle.c) flat output, layout [n_t][d_inner]:
    //   row t = [ch0_t, ch1_t, ch2_t]. ch1 = sx window-start (11..15), ch2 = window-end (24..28).
    std::vector<float> expected{1.625f, 11, 24, 2.5625f, 12, 25, 3.5f, 13, 26, 4.4375f, 14, 27, 5.375f, 15, 28};
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
    std::vector<float> sinks{0.5f, -0.5f};   // one sink logit per head
    ov::Tensor sinks_t(ov::element::f32, ov::Shape{1, 1, 1, n_head});
    std::copy(sinks.begin(), sinks.end(), sinks_t.data<float>());
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, n_head, T, Kd}, x)}, {"sinks", sinks_t}});

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

// RESHAPE flatten-for-SET_ROWS, [F, tok, 1, 1] -> [1, F*tok?, -1, 1] with the token
// count landing on the dynamic axis (constant folds to the static output shape here). Verifies the
// (previously stubbed) case emits a correct reshape rather than throwing.
TEST(GGUFOps, ReshapeCase3) {
    // input [1,1,tok=2,F=4] -> output [1, F=4, tok=2, 1] (the target places -1 on the token axis).
    auto model = SingleOpBuilder()
                     .op("GGML_OP_RESHAPE")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 4, 2, 1})
                     .attr<std::vector<int64_t>>("reshape_target", {1, 4, -1, 1})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});
    // Row-major reshape preserves element order.
    expect_near(out, x, 0.0f);
}

// The attention split/merge RESHAPEs must be LAYOUT-POLYMORPHIC:
// valid both for plain SDPA inference, which feeds a batch-major activation ([1, tokens, ..]), and
// after ov::pass::SDPAToPagedAttention, which moves the token count into dim 0 ([tokens, 1, ..]).
// Those two are the same buffer, so the op must copy the leading dim through rather than pin a
// literal 1 -- pinning silently rewrites a token-major activation into a batch-major one and the
// PagedAttention operands come out token-count-squared.
//
// Feeding the same values under both arrangements and requiring the same output element order is
// exactly that property, and it fails on a literal-1 reshape.
TEST(GGUFOps, ReshapeCase1SplitHeadsIsLayoutPolymorphic) {
    const int64_t tokens = 2, heads = 2, head_size = 2;
    const std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};

    // Batch-major (what genai feeds a plain SDPA model): [1, 1, tokens, heads*head_size].
    auto sdpa = SingleOpBuilder()
                    .op("GGML_OP_RESHAPE")
                    .input("x", ov::element::f32, {1, 1, tokens, heads * head_size})
                    .output("out", ov::element::f32, {1, tokens, heads, head_size})
                    .attr<std::vector<int64_t>>("reshape_target", {0, -1, heads, head_size})
                    .attr("special_zero", true)
                    .build();
    auto sdpa_out = run_on_cpu(sdpa, {{"x", make_f32_tensor({1, 1, (size_t)tokens, (size_t)(heads * head_size)}, x)}});
    EXPECT_EQ(sdpa_out.get_shape(), (ov::Shape{1, (size_t)tokens, (size_t)heads, (size_t)head_size}));

    // Token-major (the layout SDPAToPagedAttention establishes): [tokens, 1, 1, heads*head_size].
    // The op must keep the tokens in dim 0 instead of moving them to dim 1.
    auto pa = SingleOpBuilder()
                  .op("GGML_OP_RESHAPE")
                  .input("x", ov::element::f32, {tokens, 1, 1, heads * head_size})
                  .output("out", ov::element::f32, {1, tokens, heads, head_size})
                  .attr<std::vector<int64_t>>("reshape_target", {0, -1, heads, head_size})
                  .attr("special_zero", true)
                  .build();
    auto pa_out = run_on_cpu(pa, {{"x", make_f32_tensor({(size_t)tokens, 1, 1, (size_t)(heads * head_size)}, x)}});
    EXPECT_EQ(pa_out.get_shape(), (ov::Shape{(size_t)tokens, 1, (size_t)heads, (size_t)head_size}));

    // Same buffer in, same buffer out.
    expect_near(sdpa_out, x, 0.0f);
    expect_near(pa_out, x, 0.0f);
}

TEST(GGUFOps, ReshapeCase2MergeHeadsIsLayoutPolymorphic) {
    const int64_t tokens = 2, heads = 2, head_size = 2;
    const std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};

    // Non-stateful ggml keeps activations rank-4 throughout: [1, tokens, H, S] -> [1, 1, tokens, H*S].
    auto sdpa = SingleOpBuilder()
                    .op("GGML_OP_RESHAPE")
                    .input("x", ov::element::f32, {1, tokens, heads, head_size})
                    .output("out", ov::element::f32, {1, 1, tokens, heads * head_size})
                    .attr<std::vector<int64_t>>("reshape_target", {0, 1, -1, heads * head_size})
                    .attr("special_zero", true)
                    .build();
    auto sdpa_out =
        run_on_cpu(sdpa, {{"x", make_f32_tensor({1, (size_t)tokens, (size_t)heads, (size_t)head_size}, x)}});
    EXPECT_EQ(sdpa_out.get_shape(), (ov::Shape{1, 1, (size_t)tokens, (size_t)(heads * head_size)}));

    // Token-major: the tokens stay in dim 0.
    auto pa = SingleOpBuilder()
                  .op("GGML_OP_RESHAPE")
                  .input("x", ov::element::f32, {tokens, 1, heads, head_size})
                  .output("out", ov::element::f32, {1, 1, tokens, heads * head_size})
                  .attr<std::vector<int64_t>>("reshape_target", {0, 1, -1, heads * head_size})
                  .attr("special_zero", true)
                  .build();
    auto pa_out = run_on_cpu(pa, {{"x", make_f32_tensor({(size_t)tokens, 1, (size_t)heads, (size_t)head_size}, x)}});
    EXPECT_EQ(pa_out.get_shape(), (ov::Shape{(size_t)tokens, 1, 1, (size_t)(heads * head_size)}));

    expect_near(sdpa_out, x, 0.0f);
    expect_near(pa_out, x, 0.0f);
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

// Real KV layout is [batch, context, heads, head_size], while SET_ROWS indices address the
// flattened context/head rows. Append one token's two heads at context slot 1 and preserve slot 0.
TEST(GGUFOps, SetRowsAppendsToKvCache) {
    constexpr size_t context = 2, heads = 2, head_size = 2;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SET_ROWS")
                     .input("data", ov::element::f32, {1, 1, heads, head_size})
                     .input("ind", ov::element::i64, {1, 1, 1, heads})
                     .input("dst", ov::element::f32, {1, context, heads, head_size})
                     .output("out", ov::element::f32, {1, context, heads, head_size})
                     .build();

    std::vector<float> data{10, 11, 20, 21};
    std::vector<int64_t> ind{2, 3};
    std::vector<float> dst{1, 2, 3, 4, -1, -1, -1, -1};

    ov::Tensor ind_t(ov::element::i64, ov::Shape{1, 1, 1, heads});
    std::copy(ind.begin(), ind.end(), ind_t.data<int64_t>());
    auto out = run_on_cpu(model,
                          {{"data", make_f32_tensor({1, 1, heads, head_size}, data)},
                           {"ind", ind_t},
                           {"dst", make_f32_tensor({1, context, heads, head_size}, dst)}});

    std::vector<float> expected{1, 2, 3, 4, 10, 11, 20, 21};
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
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .attr("rope_mode", std::string("neox"))
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data{1, 2, 3, 4, 100, 200};  // [d0,d1, d2,d3, p0,p1]; only d0..d3 rotate.
    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)T});
    pos_t.data<int32_t>()[0] = pos;
    auto out =
        run_on_cpu(model,
                   {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)}, {"pos", pos_t}});

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
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .attr("rope_mode", std::string("neox"))
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data{1, 2, 3, 4};
    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)T});
    pos_t.data<int32_t>()[0] = pos;
    auto out =
        run_on_cpu(model,
                   {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)}, {"pos", pos_t}});

    std::vector<float> cos_v, sin_v;
    neox_ref_cos_sin(static_cast<int>(head_dim), cfg.freq_base, cfg.freq_scale, cfg.attn_factor, pos, cos_v, sin_v);
    std::vector<float> expected(head_dim);
    for (int j = 0; j < 2; ++j) {
        expected[j] = data[j] * cos_v[j] - data[j + 2] * sin_v[j];
        expected[j + 2] = data[j] * sin_v[j] + data[j + 2] * cos_v[j];
    }
    expect_near(out, expected, 1e-4f);
}

// IMROPE (interleaved M-RoPE, qwen3.5 / qwen3vl) at qwen3.5's exact config: head_dim=64,
// n_dims=64, sections={11,11,10,0}, freq_base=1e7. Text decode uses the SAME position id for all
// four mrope sections, so IMROPE reduces to theta[j]=pos*theta_scale^j with split-halves pairing
// — identical to NEOX. This guards the frontend's imrope path (make_sin_cos j%3 gather + rope.cpp
// TYPE_IMROPE split/sub/add/concat) against ggml's own kernel. Expected values come from the
// standalone ggml-CPU oracle imrope_oracle.c (ggml_rope_multi, GGML_ROPE_TYPE_IMROPE).
TEST(GGUFOps, ImropeTextRealDims) {
    const int64_t head_dim = 64, T = 2, n_head = 1;
    RopeConfig cfg;
    cfg.n_dims = static_cast<int>(head_dim);
    cfg.freq_base = 1e7f;
    cfg.freq_scale = 1.0f;
    cfg.ext_factor = 0.0f;
    cfg.attn_factor = 1.0f;
    cfg.beta_fast = 32.0f;
    cfg.beta_slow = 1.0f;
    cfg.n_ctx_orig = 262144;
    // pos: mrope needs 4*T ints, section-major [t-sect | h-sect | w-sect | e-sect]; text uses the
    // same position per section, so token t -> pos t in every section: {0,1, 0,1, 0,1, 0,1}.
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, 4 * T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .attr("rope_mode", std::string("imrope"))
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data(T * head_dim);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < head_dim; ++e)
            data[t * head_dim + e] = static_cast<float>(t + 1) + 0.01f * e;

    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)(4 * T)});
    int32_t pos_vals[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    std::copy(pos_vals, pos_vals + 8, pos_t.data<int32_t>());

    auto out =
        run_on_cpu(model,
                   {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)}, {"pos", pos_t}});

    // From imrope_oracle.c (ggml-CPU IMROPE), layout [T][head_dim].
    std::vector<float> expected{// token 0 (pos 0) — identity
                                1.000000f,
                                1.010000f,
                                1.020000f,
                                1.030000f,
                                1.040000f,
                                1.050000f,
                                1.060000f,
                                1.070000f,
                                1.080000f,
                                1.090000f,
                                1.100000f,
                                1.110000f,
                                1.120000f,
                                1.130000f,
                                1.140000f,
                                1.150000f,
                                1.160000f,
                                1.170000f,
                                1.180000f,
                                1.190000f,
                                1.200000f,
                                1.210000f,
                                1.220000f,
                                1.230000f,
                                1.240000f,
                                1.250000f,
                                1.260000f,
                                1.270000f,
                                1.280000f,
                                1.290000f,
                                1.300000f,
                                1.310000f,
                                1.320000f,
                                1.330000f,
                                1.340000f,
                                1.350000f,
                                1.360000f,
                                1.370000f,
                                1.380000f,
                                1.390000f,
                                1.400000f,
                                1.410000f,
                                1.420000f,
                                1.430000f,
                                1.440000f,
                                1.450000f,
                                1.460000f,
                                1.470000f,
                                1.480000f,
                                1.490000f,
                                1.500000f,
                                1.510000f,
                                1.520000f,
                                1.530000f,
                                1.540000f,
                                1.550000f,
                                1.560000f,
                                1.570000f,
                                1.580000f,
                                1.590000f,
                                1.600000f,
                                1.610000f,
                                1.620000f,
                                1.630000f,
                                // token 1 (pos 1)
                                -0.871608f,
                                0.330166f,
                                1.051162f,
                                1.466389f,
                                1.708109f,
                                1.852569f,
                                1.941705f,
                                1.998783f,
                                2.036994f,
                                2.063982f,
                                2.084241f,
                                2.100448f,
                                2.114208f,
                                2.126487f,
                                2.137869f,
                                2.148707f,
                                2.159216f,
                                2.169524f,
                                2.179711f,
                                2.189825f,
                                2.199894f,
                                2.209936f,
                                2.219961f,
                                2.229976f,
                                2.239986f,
                                2.249991f,
                                2.259995f,
                                2.269997f,
                                2.279998f,
                                2.289999f,
                                2.299999f,
                                2.309999f,
                                2.936443f,
                                3.059410f,
                                2.907070f,
                                2.737353f,
                                2.610280f,
                                2.527328f,
                                2.477454f,
                                2.449871f,
                                2.436607f,
                                2.432320f,
                                2.433586f,
                                2.438262f,
                                2.445020f,
                                2.453050f,
                                2.461852f,
                                2.471125f,
                                2.480683f,
                                2.490415f,
                                2.500252f,
                                2.510153f,
                                2.520093f,
                                2.530056f,
                                2.540034f,
                                2.550021f,
                                2.560013f,
                                2.570008f,
                                2.580004f,
                                2.590003f,
                                2.600002f,
                                2.610001f,
                                2.620001f,
                                2.630001f};

    expect_near(out, expected, 1e-4f);
}

// IMROPE with PARTIAL rotary (n_dims < head_dim) at qwen3.5's real full-attention config:
// head_dim=256 but rope.dimension_count=64. Only the first 64 dims of every head are rotated; the
// remaining 192 pass through UNCHANGED. The head_dim==n_dims test above never exercised this, and
// the frontend used to rotate the whole head (splitting all 256 dims) -- corrupting every
// full-attention layer and, downstream, every token. Expected from imrope_partial_oracle.c
// (ggml_rope_multi, GGML_ROPE_TYPE_IMROPE, n_dims=64, head_dim=256): token 0 is identity, token 1's
// first 64 dims are the same rotated values as ImropeTextRealDims and its tail is the raw ramp.
TEST(GGUFOps, ImropePartialRotary) {
    const int64_t head_dim = 256, T = 2, n_head = 1;
    const int n_dims = 64;  // partial rotary
    RopeConfig cfg;
    cfg.n_dims = n_dims;
    cfg.freq_base = 1e7f;
    cfg.freq_scale = 1.0f;
    cfg.ext_factor = 0.0f;
    cfg.attn_factor = 1.0f;
    cfg.beta_fast = 32.0f;
    cfg.beta_slow = 1.0f;
    cfg.n_ctx_orig = 262144;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, T, n_head, head_dim})
                     .input("pos", ov::element::i32, {1, 1, 1, 4 * T})
                     .output("out", ov::element::f32, {1, T, n_head, head_dim})
                     .attr("rope_mode", std::string("imrope"))
                     .attr<RopeConfig>("rope_config", cfg)
                     .build();

    std::vector<float> data(T * head_dim);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < head_dim; ++e)
            data[t * head_dim + e] = static_cast<float>(t + 1) + 0.01f * e;

    ov::Tensor pos_t(ov::element::i32, ov::Shape{1, 1, 1, (size_t)(4 * T)});
    int32_t pos_vals[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    std::copy(pos_vals, pos_vals + 8, pos_t.data<int32_t>());

    auto out =
        run_on_cpu(model,
                   {{"data", make_f32_tensor({1, (size_t)T, (size_t)n_head, (size_t)head_dim}, data)}, {"pos", pos_t}});

    // token 1's first 64 rotated dims (from imrope_partial_oracle.c; identical to ImropeTextRealDims).
    const float rot1[64] = {-0.871608f, 0.330166f, 1.051162f, 1.466389f, 1.708109f, 1.852569f, 1.941705f, 1.998783f,
                            2.036994f,  2.063982f, 2.084241f, 2.100448f, 2.114208f, 2.126487f, 2.137869f, 2.148707f,
                            2.159216f,  2.169524f, 2.179711f, 2.189825f, 2.199894f, 2.209936f, 2.219961f, 2.229976f,
                            2.239986f,  2.249991f, 2.259995f, 2.269997f, 2.279998f, 2.289999f, 2.299999f, 2.309999f,
                            2.936443f,  3.059410f, 2.907070f, 2.737353f, 2.610280f, 2.527328f, 2.477454f, 2.449871f,
                            2.436607f,  2.432320f, 2.433586f, 2.438262f, 2.445020f, 2.453050f, 2.461852f, 2.471125f,
                            2.480683f,  2.490415f, 2.500252f, 2.510153f, 2.520093f, 2.530056f, 2.540034f, 2.550021f,
                            2.560013f,  2.570008f, 2.580004f, 2.590003f, 2.600002f, 2.610001f, 2.620001f, 2.630001f};

    std::vector<float> expected(T * head_dim);
    for (int e = 0; e < head_dim; ++e)
        expected[e] = 1.0f + 0.01f * e;  // token 0: pos 0 -> identity
    for (int e = 0; e < head_dim; ++e)
        expected[head_dim + e] = (e < n_dims) ? rot1[e] : (2.0f + 0.01f * e);  // token 1: rotate head, pass tail

    expect_near(out, expected, 1e-4f);
}

// The f4e2m1 nibble -> value lookup used by the packed MXFP4 dequant (mirrors mul_mat_id.cpp).
static const float kF4E2M1[16] =
    {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

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
    a[0 * cols + 0] = 1.0f;          // token0 act lane0 = 1
    a[1 * cols + 0] = 1.0f;          // token1 act lane0 = 1
    std::vector<int32_t> ids{1, 0};  // token0 -> expert1, token1 -> expert0

    ov::Tensor w_t(ov::element::u8, ov::Shape{1, (size_t)n_expert, (size_t)m, (size_t)k_blocks, block_bytes});
    std::memcpy(w_t.data<uint8_t>(), packed.data(), packed.size());
    ov::Tensor ids_t(ov::element::i32, ov::Shape{1, 1, (size_t)n_tokens, 1});
    std::copy(ids.begin(), ids.end(), ids_t.data<int32_t>());
    auto out =
        run_on_cpu(model,
                   {{"w", w_t}, {"a", make_f32_tensor({1, (size_t)n_tokens, 1, (size_t)cols}, a)}, {"ids", ids_t}});

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
                     .output("out", ov::element::dynamic, {1, 1, 2, 2})
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

// A FLASH_ATTN_EXT without a mask ("has_mask" false, so a fourth operand is the sinks) attends to every
// key: it must match the same operation with an all-zero additive mask, for both the soft-capped
// decomposition and the attention-sinks path.
TEST(GGUFOps, FlashAttnExtMasklessMatchesZeroMask) {
    const std::vector<float> q{0.5f, -1.0f, 0.25f, 2.0f};                    // [1, 2, 1, 2]
    const std::vector<float> k{1.0f, 0.5f, -0.5f, 1.5f};                     // [1, 1, 2, 2]
    const std::vector<float> v{0.1f, 0.2f, 0.3f, 0.4f};                      // [1, 1, 2, 2]
    const std::vector<float> sinks{0.3f, -0.7f};                             // [2]
    const std::vector<float> zero_mask(2, 0.0f);                             // [1, 1, 1, 2]
    for (bool with_sinks : {false, true}) {
        SCOPED_TRACE(with_sinks ? "sinks" : "soft cap");
        auto build = [&](bool masked) {
            auto builder = SingleOpBuilder()
                               .op("GGML_OP_FLASH_ATTN_EXT")
                               .input("q", ov::element::f32, {1, 2, 1, 2})
                               .input("k", ov::element::f32, {1, 1, 2, 2})
                               .input("v", ov::element::f32, {1, 1, 2, 2});
            if (masked) {
                builder.input("mask", ov::element::f32, {1, 1, 1, 2});
            }
            if (with_sinks) {
                builder.input("sinks", ov::element::f32, {2});
            } else {
                builder.attr<float>("kq_soft_cap", 10.0f);
            }
            return builder.output("out", ov::element::f32, {1, 1, 2, 2})
                .attr<float>("scale", 1.0f)
                .attr<bool>("has_mask", masked)
                .build();
        };
        std::map<std::string, ov::Tensor> inputs{{"q", make_f32_tensor({1, 2, 1, 2}, q)},
                                                 {"k", make_f32_tensor({1, 1, 2, 2}, k)},
                                                 {"v", make_f32_tensor({1, 1, 2, 2}, v)}};
        if (with_sinks) {
            inputs["sinks"] = make_f32_tensor({2}, sinks);
        }
        const auto maskless = run_on_cpu(build(false), inputs);
        inputs["mask"] = make_f32_tensor({1, 1, 1, 2}, zero_mask);
        const auto masked = run_on_cpu(build(true), inputs);
        ASSERT_EQ(maskless.get_shape(), masked.get_shape());
        std::vector<float> expected(masked.data<float>(), masked.data<float>() + masked.get_size());
        expect_near(maskless, expected, 1e-3f);
    }
}

TEST(GGUFOps, FlashAttnExtAcceptsF32MaskMatchingHeadCount) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_FLASH_ATTN_EXT")
                     .input("q", ov::element::f32, {1, 2, 1, 2})
                     .input("k", ov::element::f32, {1, 1, 2, 2})
                     .input("v", ov::element::f32, {1, 1, 2, 2})
                     .input("mask", ov::element::f32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .attr<float>("scale", 1.0f)
                     .build();
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({1, 2, 1, 2}, {1, 0, 0, 1})},
                           {"k", make_f32_tensor({1, 1, 2, 2}, {1, 0, 0, 1})},
                           {"v", make_f32_tensor({1, 1, 2, 2}, {1, 2, 3, 4})},
                           {"mask", make_f32_tensor({1, 1, 1, 2}, {0, -100})}});
    expect_near(out, {1, 2, 1, 2}, 2e-2f);
}

// FlashAttnExt with gpt-oss attention sinks (5th input), in ggml's operand layout (llama.cpp cgraph)
// layout where q/k/v already arrive as [B, n_head, T, D]. Regression test for a bug where the sink
// logit was reshaped to [1, q_shape[2], 1, 1] instead of [1, q_shape[head_axis], 1, 1]: q_shape[2] is
// T on this layout (head_axis == 1), not n_head, so with n_head != T the reshape either throws a
// shape mismatch or silently broadcasts the wrong values into the softmax denominator. Using
// n_head=2 and T=3 (both != 1, and different from each other) makes either failure mode observable.
void check_flash_attn_ext_with_sinks(bool f32_attention) {
    const size_t n_head = 2, T = 3, Tk = 3, D = 2;
    const float scale = 1.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_FLASH_ATTN_EXT")
                     .input("q", ov::element::f32, {1, n_head, T, D})
                     .input("k", ov::element::f32, {1, n_head, Tk, D})
                     .input("v", ov::element::f32, {1, n_head, Tk, D})
                     .input("mask", ov::element::f32, {1, 1, T, Tk})
                     .input("sinks", ov::element::f32, {n_head})
                     .output("out", ov::element::f32, {1, T, n_head, D})
                     .attr<float>("scale", scale)
                     .attr<bool>("f32_attention", f32_attention)
                     .build();

    std::vector<float> q, k, v;
    for (size_t i = 0; i < n_head * T * D; ++i)
        q.push_back(0.1f * static_cast<float>(i) - 1.0f);
    for (size_t i = 0; i < n_head * Tk * D; ++i) {
        k.push_back(0.2f * static_cast<float>(i) - 1.0f);
        v.push_back(0.05f * static_cast<float>(i));
    }
    std::vector<float> mask(T * Tk, 0.0f);
    std::vector<float> sinks{0.5f, -0.3f};

    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({1, n_head, T, D}, q)},
                           {"k", make_f32_tensor({1, n_head, Tk, D}, k)},
                           {"v", make_f32_tensor({1, n_head, Tk, D}, v)},
                           {"mask", make_f32_tensor({1, 1, T, Tk}, mask)},
                           {"sinks", make_f32_tensor({n_head}, sinks)}});

    // Reference: per head, augment the softmax denominator with the head's sink logit (dropped from
    // the weighted value sum), same convention as SoftMaxSinks. Output layout [1,T,H,D].
    std::vector<float> expected(T * n_head * D);
    for (size_t h = 0; h < n_head; ++h) {
        for (size_t t = 0; t < T; ++t) {
            std::vector<float> z(Tk + 1);
            for (size_t s = 0; s < Tk; ++s) {
                float dot = 0.f;
                for (size_t d = 0; d < D; ++d)
                    dot += q[(h * T + t) * D + d] * k[(h * Tk + s) * D + d];
                z[s] = scale * dot + mask[t * Tk + s];
            }
            z[Tk] = sinks[h];
            float mx = *std::max_element(z.begin(), z.end());
            float sum = 0.f;
            for (float& zi : z) {
                zi = std::exp(zi - mx);
                sum += zi;
            }
            for (size_t d = 0; d < D; ++d) {
                float acc = 0.f;
                for (size_t s = 0; s < Tk; ++s)
                    acc += (z[s] / sum) * v[(h * Tk + s) * D + d];
                expected[(t * n_head + h) * D + d] = acc;
            }
        }
    }
    expect_near(out, expected, 2e-2f);
}

TEST(GGUFOps, FlashAttnExtWithSinksCgraphLayout) {
    check_flash_attn_ext_with_sinks(false);
}

TEST(GGUFOps, FlashAttnExtWithSinksF32Attention) {
    check_flash_attn_ext_with_sinks(true);
}

// GatedDeltaNet, reference (Loop) path. With head size S=1 the gate last-dim equals S_v, so this is
// the per-key-dimension gating case (kda) that the fused op does not support and which therefore
// lowers to the serializable Loop scan. Minimal scalar case B=H=S=1, T=2 exercises the full
// per-token gated delta update + output packing [attn_t0, attn_t1, final_state]. This model stays
// serializable, so we also assert no internal GatedDeltaNet op is present.
TEST(GGUFOps, GatedDeltaNetRefFallback) {
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

    // Fallback path is core-op only: no internal fused op, and it uses a Loop scan.
    bool has_fused = false, has_loop = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) == std::string("GatedDeltaNet"))
            has_fused = true;
        if (std::string(node->get_type_name()) == std::string("Loop"))
            has_loop = true;
    }
    EXPECT_FALSE(has_fused);
    EXPECT_TRUE(has_loop);
}

// K > 1 (multiple recurrent-state snapshot slots, speculative-decode rollback) reserves K state
// blocks in the ggml output while both frontend paths pack one, so conversion must fail loudly
// rather than emit a graph whose snapshot views read the wrong block.
TEST(GGUFOps, GatedDeltaNetMultipleSnapshotSlotsRejected) {
    const int64_t B = 1, T = 2, H = 1, D = 2, K = 2;
    auto qkv_shp = ov::PartialShape{B, T, H, D};
    auto gate_shp = ov::PartialShape{B, T, H, 1};
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_GATED_DELTA_NET")
                       .attr<int64_t>("gdn_state_slots", K)
                       .input("q", ov::element::f32, qkv_shp)
                       .input("k", ov::element::f32, qkv_shp)
                       .input("v", ov::element::f32, qkv_shp)
                       .input("g", ov::element::f32, gate_shp)
                       .input("beta", ov::element::f32, gate_shp)
                       .input("state", ov::element::f32, {B, H, D, D})
                       .output("out", ov::element::f32, {1, 1, (T + K * D) * B, D * H});
    EXPECT_THROW(builder.build(), ov::Exception);
}

// GatedDeltaNet, fused-op path. A scalar gate (gate last-dim 1 with head size Dv>1) selects the
// fused ov::op::internal::GatedDeltaNet op instead of the Loop scan. B=H=1, T=2, D=Dv=2. We assert
// the fused op is actually emitted and that its result matches the core reference recurrence.
// NOTE: a model on this path contains an internal op and is therefore NOT IR-serializable
// (see src/frontends/gguf/docs/runtime.md).
TEST(GGUFOps, GatedDeltaNetFused) {
    const int64_t B = 1, T = 2, H = 1, D = 2;  // head size D = Dv = 2, scalar gate
    auto qkv_shp = ov::PartialShape{B, T, H, D};
    auto gate_shp = ov::PartialShape{B, T, H, 1};   // scalar gate -> fused path
    auto state_shp = ov::PartialShape{B, H, D, D};  // ggml [B, H_v, value_dim, key_dim]
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .input("q", ov::element::f32, qkv_shp)
                     .input("k", ov::element::f32, qkv_shp)
                     .input("v", ov::element::f32, qkv_shp)
                     .input("g", ov::element::f32, gate_shp)
                     .input("beta", ov::element::f32, gate_shp)
                     .input("state", ov::element::f32, state_shp)
                     .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H})
                     .build();

    // Assert the fused internal op is present (this is the whole point of this path).
    bool has_fused = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (std::string(node->get_type_name()) == std::string("GatedDeltaNet"))
            has_fused = true;
    }
    ASSERT_TRUE(has_fused) << "fused ov::op::internal::GatedDeltaNet was not emitted";

    // Per-token, per-dim inputs (row-major over [T, D]).
    std::vector<float> q{1, 0, 0, 1}, k{1, 0, 0, 1}, v{1, 2, 3, 4};
    std::vector<float> g{0.0f, 0.0f}, beta{1.0f, 1.0f};  // exp(g)=1, full update
    std::vector<float> state0(D * D, 0.0f);
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, q)},
                           {"k", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, k)},
                           {"v", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, v)},
                           {"g", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, g)},
                           {"beta", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, beta)},
                           {"state", make_f32_tensor({(size_t)B, (size_t)H, (size_t)D, (size_t)D}, state0)}});

    // Reference: mirror ov::reference::gated_delta_net for B=H=1. state[d][dv]; q scaled by
    // 1/sqrt(D). For each token: decay state by exp(g); h_k[dv] = sum_d state[d][dv]*k[d];
    // update state[d][dv] += k[d]*(v[dv]-h_k[dv])*beta; attn[dv] = sum_d state[d][dv]*q_scaled[d].
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<std::vector<float>> st(D, std::vector<float>(D, 0.0f));  // st[d][dv]
    std::vector<float> attn;                                             // [T*D]
    for (int t = 0; t < T; ++t) {
        std::vector<float> qs(D), kv(D);
        for (int d = 0; d < D; ++d) {
            qs[d] = q[t * D + d] * scale;
            kv[d] = k[t * D + d];
        }
        float gexp = std::exp(g[t]);
        for (int d = 0; d < D; ++d)
            for (int dv = 0; dv < D; ++dv)
                st[d][dv] *= gexp;
        for (int dv = 0; dv < D; ++dv) {
            float h_k = 0.0f;
            for (int d = 0; d < D; ++d)
                h_k += st[d][dv] * kv[d];
            float upd = (v[t * D + dv] - h_k) * beta[t];
            for (int d = 0; d < D; ++d)
                st[d][dv] += kv[d] * upd;
        }
        for (int dv = 0; dv < D; ++dv) {
            float a = 0.0f;
            for (int d = 0; d < D; ++d)
                a += st[d][dv] * qs[d];
            attn.push_back(a);
        }
    }
    // Packed output layout: [attn (T*D) | final_state (D*D)] flattened. The frontend transposes the
    // op's [key_dim, value_dim] state back to ggml's [value_dim, key_dim] before flattening, so the
    // state is packed value-dim outer, key-dim inner.
    std::vector<float> expected = attn;
    for (int dv = 0; dv < D; ++dv)
        for (int d = 0; d < D; ++d)
            expected.push_back(st[d][dv]);
    expect_near(out, expected, 1e-4f);
}

// Requested Q/K normalization follows ggml, x / max(||x||, eps), also for vectors below eps-scale
// norms: q = k = [1e-4, 0] normalize to [1, 0], so one update with v = [1, 0] attends 1/sqrt(2).
TEST(GGUFOps, GatedDeltaNetFusedQkNormFollowsGgmlForSmallVectors) {
    const int64_t B = 1, T = 1, H = 1, D = 2;
    auto qkv_shp = ov::PartialShape{B, T, H, D};
    auto gate_shp = ov::PartialShape{B, T, H, 1};
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .input("q", ov::element::f32, qkv_shp)
                     .input("k", ov::element::f32, qkv_shp)
                     .input("v", ov::element::f32, qkv_shp)
                     .input("g", ov::element::f32, gate_shp)
                     .input("beta", ov::element::f32, gate_shp)
                     .input("state", ov::element::f32, ov::PartialShape{B, H, D, D})
                     .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H})
                     .attr("fuse_qk_l2norm", true)
                     .attr("qk_l2_norm_eps", 1e-6f)
                     .build();
    const ov::Shape qkv{1, 1, 1, 2}, gate{1, 1, 1, 1};
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor(qkv, {1e-4f, 0})},
                           {"k", make_f32_tensor(qkv, {1e-4f, 0})},
                           {"v", make_f32_tensor(qkv, {1, 0})},
                           {"g", make_f32_tensor(gate, {0})},
                           {"beta", make_f32_tensor(gate, {1})},
                           {"state", make_f32_tensor({1, 1, 2, 2}, {0, 0, 0, 0})}});
    ASSERT_GE(out.get_size(), 2u);
    EXPECT_NEAR(out.data<const float>()[0], 1.0f / std::sqrt(2.0f), 1e-4f);
    EXPECT_NEAR(out.data<const float>()[1], 0.0f, 1e-6f);
}

// GatedDeltaNet, fused-op path with MULTIPLE HEADS (H=2). qwen3-next real dims are H=32; the
// single-head tests above (H=1) collapse the head axis so they never exercise the per-head packing
// of attn/state into the flat output. This reproduces the multi-head case: for B=1, T tokens and
// H heads of size D, the packed output is [1,1,(T+D), D*H] -- attention on the first T rows (each
// row = H heads x D values, head-outer) and the H final states on the last D rows. Independent
// per-head recurrences let a head-axis packing/transpose bug surface as a mismatch.
TEST(GGUFOps, GatedDeltaNetFusedMultiHead) {
    const int64_t B = 1, T = 2, H = 2, D = 2;  // scalar gate (last-dim 1), head size D>1 -> fused path
    auto qkv_shp = ov::PartialShape{B, T, H, D};
    auto gate_shp = ov::PartialShape{B, T, H, 1};
    auto state_shp = ov::PartialShape{B, H, D, D};  // ggml [B, H_v, value_dim, key_dim]
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .input("q", ov::element::f32, qkv_shp)
                     .input("k", ov::element::f32, qkv_shp)
                     .input("v", ov::element::f32, qkv_shp)
                     .input("g", ov::element::f32, gate_shp)
                     .input("beta", ov::element::f32, gate_shp)
                     .input("state", ov::element::f32, state_shp)
                     .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H})
                     .build();

    bool has_fused = false;
    for (const auto& node : model->get_ordered_ops())
        if (std::string(node->get_type_name()) == std::string("GatedDeltaNet"))
            has_fused = true;
    ASSERT_TRUE(has_fused) << "fused ov::op::internal::GatedDeltaNet was not emitted";

    // Distinct per-head, per-token, per-dim inputs (row-major over [T, H, D]).
    std::vector<float> q{1, 0, 0, 1, /*t1*/ 0, 1, 1, 0};
    std::vector<float> k{1, 0, 0, 1, /*t1*/ 1, 1, 0, 1};
    std::vector<float> v{1, 2, 3, 4, /*t1*/ 5, 6, 7, 8};
    std::vector<float> g{0.0f, 0.0f, 0.0f, 0.0f};     // exp(g)=1
    std::vector<float> beta{1.0f, 1.0f, 1.0f, 1.0f};  // full update
    std::vector<float> state0(H * D * D, 0.0f);
    auto idx = [&](int t, int h, int d) {
        return (t * H + h) * D + d;
    };  // [T,H,D]
    auto gidx = [&](int t, int h) {
        return t * H + h;
    };  // [T,H]

    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, q)},
                           {"k", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, k)},
                           {"v", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, v)},
                           {"g", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, g)},
                           {"beta", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, beta)},
                           {"state", make_f32_tensor({(size_t)B, (size_t)H, (size_t)D, (size_t)D}, state0)}});

    (void)idx;
    (void)gidx;
    // Authoritative reference: the flat output of ggml's own CPU GGML_OP_GATED_DELTA_NET kernel for
    // these exact inputs (generated by the standalone oracle in
    // src/frontends/gguf/tests/gdn_oracle.c, run against llama.cpp's libggml-cpu). Using ggml's
    // kernel -- rather than a hand-derived recurrence -- avoids baking the frontend's own layout
    // assumptions into the expectation. The packed layout is [S_v*H, T + K*S_v] flattened: T=2
    // attention rows (width S_v*H, head-outer/value-inner) then K*S_v=2 final-state rows. Note the
    // state is packed head-outer/value-dim/key-dim -- a multi-head order the H=1 tests can't probe.
    std::vector<float> expected{// attention, token 0 and token 1 (each row: head0[sv0,sv1], head1[sv0,sv1])
                                0.707107f,
                                1.414214f,
                                2.121320f,
                                2.828427f,
                                2.828427f,
                                2.828427f,
                                0.000000f,
                                0.000000f,
                                // final state, two rows (head-outer): head0 [dv0(d0,d1), dv1(d0,d1)], head1 [...]
                                5.000000f,
                                4.000000f,
                                6.000000f,
                                4.000000f,
                                0.000000f,
                                7.000000f,
                                0.000000f,
                                8.000000f};
    expect_near(out, expected, 1e-4f);
}

// GatedDeltaNet, DECOMPOSED (Loop reference) path with MULTIPLE HEADS. Same inputs and same ggml-CPU
// authoritative reference as GatedDeltaNetFusedMultiHead, but forces the serializable Loop scan
// (force_ref attribute) even for the scalar-gate case. Both paths must match ggml; running the real
// model showed they produce DIFFERENT wrong outputs, so at least one mishandles the head axis. This
// isolates which. The Loop path is the one used when a model must stay IR-serializable, so its
// multi-head correctness matters independently of the fused op.
TEST(GGUFOps, GatedDeltaNetRefMultiHead) {
    const int64_t B = 1, T = 2, H = 2, D = 2;
    auto qkv_shp = ov::PartialShape{B, T, H, D};
    auto gate_shp = ov::PartialShape{B, T, H, 1};
    auto state_shp = ov::PartialShape{B, H, D, D};

    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .attr<bool>("force_ref", true)
                     .input("q", ov::element::f32, qkv_shp)
                     .input("k", ov::element::f32, qkv_shp)
                     .input("v", ov::element::f32, qkv_shp)
                     .input("g", ov::element::f32, gate_shp)
                     .input("beta", ov::element::f32, gate_shp)
                     .input("state", ov::element::f32, state_shp)
                     .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H})
                     .build();

    // This path must be the core-op Loop scan, not the fused internal op.
    bool has_fused = false, has_loop = false;
    for (const auto& node : model->get_ordered_ops()) {
        const std::string tn = node->get_type_name();
        if (tn == "GatedDeltaNet")
            has_fused = true;
        if (tn == "Loop")
            has_loop = true;
    }
    ASSERT_FALSE(has_fused) << "expected the decomposed path, but the fused op was emitted";
    ASSERT_TRUE(has_loop) << "expected the Loop-scan decomposition";

    std::vector<float> q{1, 0, 0, 1, 0, 1, 1, 0};
    std::vector<float> k{1, 0, 0, 1, 1, 1, 0, 1};
    std::vector<float> v{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> g{0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> beta{1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> state0(H * D * D, 0.0f);

    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, q)},
                           {"k", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, k)},
                           {"v", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, v)},
                           {"g", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, g)},
                           {"beta", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, beta)},
                           {"state", make_f32_tensor({(size_t)B, (size_t)H, (size_t)D, (size_t)D}, state0)}});

    // Same ggml-CPU oracle values as the fused multi-head test.
    std::vector<float> expected{0.707107f,
                                1.414214f,
                                2.121320f,
                                2.828427f,
                                2.828427f,
                                2.828427f,
                                0.000000f,
                                0.000000f,
                                5.000000f,
                                4.000000f,
                                6.000000f,
                                4.000000f,
                                0.000000f,
                                7.000000f,
                                0.000000f,
                                8.000000f};
    expect_near(out, expected, 1e-4f);
}

// GatedDeltaNet with GQA (H_v != H_k) -- the head-mapping path that EVERY existing GDN test misses.
// Real qwen3.5 has num_v_heads=32, num_k_heads=16 (ratio 2): each V head pairs with Q/K head
// (v_head % H_k). All the tests above use H_k==H_v, so the frontend's GQA repeat (fused: Tile;
// ref: Broadcast) was never checked against ggml. A wrong pairing (grouped vs interleaved, or
// wrong modulus) computes every non-first head against the wrong key -- a sharp, per-token error
// that a sum-based cross-backend metric smears into apparent "drift". This uses H_v=4, H_k=2 with
// distinct per-head inputs; expected = ggml-CPU oracle gdn_gqa_oracle.c.
static const std::vector<float> kGdnGqaExpected{
    // attention rows (T=2), each row = 4 v-heads x D=2, head-outer/value-inner
    0.707107f,
    1.414214f,
    2.121320f,
    2.828427f,
    3.535534f,
    4.242640f,
    4.949748f,
    5.656854f,
    5.656854f,
    5.656854f,
    0.000000f,
    0.000000f,
    5.656854f,
    5.656854f,
    0.000000f,
    0.000000f,
    // final-state rows (D=2), head-outer: each head [dv0(d0,d1), dv1(d0,d1)]
    9.000000f,
    8.000000f,
    10.000000f,
    8.000000f,
    0.000000f,
    11.000000f,
    0.000000f,
    12.000000f,
    13.000000f,
    8.000000f,
    14.000000f,
    8.000000f,
    0.000000f,
    15.000000f,
    0.000000f,
    16.000000f};

static void run_gdn_gqa(bool force_ref) {
    const int64_t B = 1, T = 2, H_v = 4, H_k = 2, D = 2;
    auto q_shp = ov::PartialShape{B, T, H_k, D};
    auto v_shp = ov::PartialShape{B, T, H_v, D};
    auto gate_shp = ov::PartialShape{B, T, H_v, 1};
    auto state_shp = ov::PartialShape{B, H_v, D, D};  // ggml [B, H_v, value_dim, key_dim]

    auto model = SingleOpBuilder()
                     .op("GGML_OP_GATED_DELTA_NET")
                     .attr<bool>("force_ref", force_ref)
                     .input("q", ov::element::f32, q_shp)
                     .input("k", ov::element::f32, q_shp)
                     .input("v", ov::element::f32, v_shp)
                     .input("g", ov::element::f32, gate_shp)
                     .input("beta", ov::element::f32, gate_shp)
                     .input("state", ov::element::f32, state_shp)
                     .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H_v})
                     .build();

    // q/k: [T, H_k, D]; v: [T, H_v, D] -- distinct per head so the GQA pairing is observable.
    std::vector<float> q{1, 0, 0, 1, 0, 1, 1, 0};
    std::vector<float> k{1, 0, 0, 1, 1, 1, 0, 1};
    std::vector<float> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<float> g(H_v * T, 0.0f);
    std::vector<float> beta(H_v * T, 1.0f);
    std::vector<float> state0(H_v * D * D, 0.0f);

    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H_k, (size_t)D}, q)},
                           {"k", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H_k, (size_t)D}, k)},
                           {"v", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H_v, (size_t)D}, v)},
                           {"g", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H_v, 1}, g)},
                           {"beta", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H_v, 1}, beta)},
                           {"state", make_f32_tensor({(size_t)B, (size_t)H_v, (size_t)D, (size_t)D}, state0)}});
    expect_near(out, kGdnGqaExpected, 1e-4f);
}

TEST(GGUFOps, GatedDeltaNetFusedGQA) {
    run_gdn_gqa(/*force_ref=*/false);
}

TEST(GGUFOps, GatedDeltaNetRefGQA) {
    run_gdn_gqa(/*force_ref=*/true);
}

// GatedDeltaNet at T != D (T=3 tokens, D=4 head-size, H=2). The multi-head tests above use T==D==2,
// which makes the packed-attention rows (T of them) and the packed-state rows (D of them) the same
// width AND makes a token<->value-axis transpose in the output flattening invisible. Real qwen3-next
// dims are T=5, D=128 -- very much T != D -- and forcing the fused vs decomposed path on the real
// model produced DIFFERENT wrong tokens, which is the signature of exactly such a collapsed-axis bug.
// This case (T != D) distinguishes them. Oracle values from gdn_oracle2 (ggml CPU, same kernel).
// Packed output is [D*H, T + D]: T=3 attention rows (width D*H, head-outer/value-inner) then D=4
// final-state rows.
namespace {
// ggml-CPU oracle output for the T=3,H=2,D=4 inputs below (row-major, 7 rows x 8 cols).
const std::vector<float> kGdnTneqDExpected{
    0.037500f,  0.075000f,  0.112500f,  0.150000f,  0.885000f,  1.062000f,  1.239000f,  1.416000f,
    2.495686f,  2.863373f,  3.231059f,  3.598745f,  7.552056f,  8.348866f,  9.145678f,  9.942489f,
    10.676509f, 11.612682f, 12.548857f, 13.485029f, 18.103098f, 19.490808f, 20.878521f, 22.266232f,
    1.785152f,  2.487126f,  3.189099f,  3.891073f,  1.928201f,  2.700328f,  3.472456f,  4.244583f,
    2.071249f,  2.913531f,  3.755813f,  4.598094f,  2.214298f,  3.126734f,  4.039169f,  4.951605f,
    7.141428f,  4.162388f,  2.434744f,  2.688347f,  7.614682f,  4.536554f,  2.686254f,  2.846659f,
    8.087935f,  4.910720f,  2.937764f,  3.004971f,  8.561190f,  5.284886f,  3.189275f,  3.163283f};

// Inputs (row-major [T,H,D] for q/k/v, [T,H] for g/beta), matching gdn_oracle2.c.
const std::vector<float> kGdnTneqDq{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
                                    1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f};
const std::vector<float> kGdnTneqDk{0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f,
                                    0.30f, 0.35f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.05f, 0.10f, 0.15f};
const std::vector<float> kGdnTneqDv{1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
const std::vector<float> kGdnTneqDg{0.0f, -0.1f, -0.2f, 0.0f, -0.1f, -0.2f};
const std::vector<float> kGdnTneqDbeta{0.5f, 0.6f, 0.7f, 0.8f, 0.5f, 0.6f};

ov::Tensor run_gdn_tneqd(bool force_ref, int64_t valid_len = -1, bool dynamic_tokens = false) {
    const int64_t B = 1, T = 3, H = 2, D = 4;
    const auto tokens = dynamic_tokens ? ov::Dimension::dynamic() : ov::Dimension(T);
    auto qkv_shp = ov::PartialShape{B, tokens, H, D};
    auto gate_shp = ov::PartialShape{B, tokens, H, 1};
    auto state_shp = ov::PartialShape{B, H, D, D};
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_GATED_DELTA_NET")
                       .attr<bool>("force_ref", force_ref)
                       .input("q", ov::element::f32, qkv_shp)
                       .input("k", ov::element::f32, qkv_shp)
                       .input("v", ov::element::f32, qkv_shp)
                       .input("g", ov::element::f32, gate_shp)
                       .input("beta", ov::element::f32, gate_shp)
                       .input("state", ov::element::f32, state_shp)
                       .output("out", ov::element::f32, {1, 1, (T + D) * B, D * H});
    if (valid_len >= 0) {
        builder.extra_input("chunk_valid_len", ov::element::i64, {1});
    }
    auto model = builder.build();
    std::vector<float> state0((size_t)(H * D * D), 0.0f);
    std::map<std::string, ov::Tensor> inputs{
        {"q", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, kGdnTneqDq)},
        {"k", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, kGdnTneqDk)},
        {"v", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, (size_t)D}, kGdnTneqDv)},
        {"g", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, kGdnTneqDg)},
        {"beta", make_f32_tensor({(size_t)B, (size_t)T, (size_t)H, 1}, kGdnTneqDbeta)},
        {"state", make_f32_tensor({(size_t)B, (size_t)H, (size_t)D, (size_t)D}, state0)}};
    if (valid_len >= 0) {
        inputs.emplace("chunk_valid_len", make_i64_tensor({1}, {valid_len}));
    }
    return run_on_cpu(model, inputs);
}
}  // namespace

TEST(GGUFOps, GatedDeltaNetFusedTneqD) {
    auto out = run_gdn_tneqd(/*force_ref=*/false);
    expect_near(out, kGdnTneqDExpected, 1e-3f);
}

TEST(GGUFOps, GatedDeltaNetRefTneqD) {
    auto out = run_gdn_tneqd(/*force_ref=*/true);
    expect_near(out, kGdnTneqDExpected, 1e-3f);
}

TEST(GGUFOps, GatedDeltaNetRefDynamicTokens) {
    auto out = run_gdn_tneqd(/*force_ref=*/true, /*valid_len=*/-1, /*dynamic_tokens=*/true);
    expect_near(out, kGdnTneqDExpected, 1e-3f);
}

TEST(GGUFOps, GatedDeltaNetRefMasksStaticPadding) {
    for (const int64_t valid_len : {int64_t{0}, int64_t{1}, int64_t{3}}) {
        auto fused = run_gdn_tneqd(/*force_ref=*/false, valid_len);
        auto reference = run_gdn_tneqd(/*force_ref=*/true, valid_len);
        ASSERT_EQ(fused.get_size(), reference.get_size());
        expect_near(reference, std::vector<float>(fused.data<float>(), fused.data<float>() + fused.get_size()), 1e-3f);
    }
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

TEST(GGUFOps, MultimodalPoolingAndConvolution) {
    auto pool = SingleOpBuilder()
                    .op("GGML_OP_POOL_2D")
                    .input("x", ov::element::f32, {1, 1, 2, 4})
                    .output("out", ov::element::f32, {1, 1, 1, 2})
                    .attr<std::vector<int32_t>>("pool_params", {1, 2, 2, 2, 2, 0, 0})
                    .build();
    const auto x = make_f32_tensor({1, 1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    expect_near(run_on_cpu(pool, {{"x", x}}), {3.5f, 5.5f}, 1e-6f);
    auto conv = SingleOpBuilder()
                    .op("GGML_OP_CONV_2D")
                    .input("w", ov::element::f32, {1, 1, 2, 2})
                    .input("x", ov::element::f32, {1, 1, 2, 4})
                    .output("out", ov::element::f32, {1, 1, 1, 2})
                    .attr<std::vector<int64_t>>("conv_params", {2, 2, 0, 0, 1, 1})
                    .build();
    expect_near(run_on_cpu(conv, {{"x", x}, {"w", make_f32_tensor({1, 1, 2, 2}, {1, 2, 3, 4})}}), {44, 64}, 1e-6f);
}

TEST(GGUFOps, GeluErf) {
    auto model = SingleOpBuilder()
                     .op("GGML_UNARY_OP_GELU_ERF")
                     .input("x", ov::element::f32, {1, 1, 1, 5})
                     .output("out", ov::element::f32, {1, 1, 1, 5})
                     .build();
    std::vector<float> values{-2, -1, 0, 1, 2}, expected;
    for (auto x : values)
        expected.push_back(0.5f * x * (1.f + std::erf(x / std::sqrt(2.f))));
    expect_near(run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 5}, values)}}), expected, 1e-6f);
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

TEST(GGUFOps, CpyConvolutionStateUsesDestinationType) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CPY")
                     .input("window", ov::element::f32, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 1, 6})
                     .attr<ov::element::Type>("dst_type", ov::element::f32)
                     .attr<std::vector<int64_t>>("reshape_target", {1, 1, 1, 6})
                     .attr<ov::element::Type>("output_type", ov::element::dynamic)
                     .build();

    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    auto out = run_on_cpu(model, {{"window", make_f32_tensor({1, 1, 2, 3}, values)}});
    EXPECT_EQ(out.get_shape(), (ov::Shape{1, 1, 1, 6}));
    expect_near(out, values, 0.0f);
}

TEST(GGUFOps, CpyEmptyCompactionPreservesCacheProducerName) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CPY")
                     .input("empty_rows", ov::element::f32, {1, 1, 0, 4})
                     .input("cache", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 4})
                     .attr<int64_t>("axis", 2)
                     .attr<int64_t>("destination_offset", 2)
                     .build();

    const auto& cache = model->get_results()[0]->input_value(0).get_node_shared_ptr();
    EXPECT_STREQ(cache->get_type_name(), "Parameter");
    EXPECT_EQ(cache->get_friendly_name(), "cache");
}

// Cont: after a PERMUTE/TRANSPOSE the OV tensor is already logically contiguous, so
// CONT is an identity passthrough of its input.
TEST(GGUFOps, Cont) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CONT")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});
    expect_near(out, x, 0.0f);
}

TEST(GGUFOps, ContViewPassThroughPreservesProducerName) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CONT")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .build();

    EXPECT_EQ(model->get_parameters()[0]->get_friendly_name(), "x");
    EXPECT_EQ(model->get_results()[0]->input_value(0).get_node_shared_ptr(), model->get_parameters()[0]);
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

TEST(GGUFOps, GeGLUQuickOneInputSwapped) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_GEGLU_QUICK")
                     .input("x", ov::element::f32, {1, 8})
                     .output("out", ov::element::f32, {1, 4})
                     .attr<bool>("swapped", true)
                     .build();
    std::vector<float> x{1, 2, 3, 4, -1, -2, 0.5f, 1.5f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 8}, x)}});
    std::vector<float> expected(4);
    for (size_t i = 0; i < expected.size(); ++i) {
        const float gate = x[4 + i];
        expected[i] = gate / (1.0f + std::exp(-1.702f * gate)) * x[i];
    }
    expect_near(out, expected);
}

TEST(GGUFOps, GeGLUQuickTwoInputs) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_GEGLU_QUICK")
                     .input("gate", ov::element::f32, {1, 3})
                     .input("up", ov::element::f32, {1, 3})
                     .output("out", ov::element::f32, {1, 3})
                     .attr<bool>("swapped", false)
                     .build();
    std::vector<float> gate{-2.0f, 0.5f, 3.0f};
    std::vector<float> up{4.0f, -1.0f, 2.0f};
    auto out = run_on_cpu(model, {{"gate", make_f32_tensor({1, 3}, gate)}, {"up", make_f32_tensor({1, 3}, up)}});
    std::vector<float> expected(3);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = gate[i] / (1.0f + std::exp(-1.702f * gate[i])) * up[i];
    }
    expect_near(out, expected);
}

TEST(GGUFOps, Pool2DMaxAndAverage) {
    const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<int32_t> params{2, 2, 2, 1, 1, 0};  // {k0, k1, s0, s1, p0, p1}
    const auto with_op = [&](int32_t op) {
        std::vector<int32_t> values{op};
        values.insert(values.end(), params.begin(), params.end());
        return values;
    };

    auto max_model = SingleOpBuilder()
                         .op("GGML_OP_POOL_2D")
                         .input("x", ov::element::f32, {1, 1, 3, 4})
                         .output("out", ov::element::f32, {1, 1, 2, 3})
                         .attr<std::vector<int32_t>>("pool_params", with_op(0))
                         .build();
    auto avg_model = SingleOpBuilder()
                         .op("GGML_OP_POOL_2D")
                         .input("x", ov::element::f32, {1, 1, 3, 4})
                         .output("out", ov::element::f32, {1, 1, 2, 3})
                         .attr<std::vector<int32_t>>("pool_params", with_op(1))
                         .build();

    auto tensor = make_f32_tensor({1, 1, 3, 4}, input);
    expect_near(run_on_cpu(max_model, {{"x", tensor}}), {5, 7, 8, 9, 11, 12});
    expect_near(run_on_cpu(avg_model, {{"x", tensor}}), {1.5f, 4.5f, 3.0f, 3.5f, 8.5f, 5.0f});
}

TEST(GGUFOps, Pool2DF16InputProducesF32) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_POOL_2D")
                     .input("x", ov::element::f16, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 1, 2})
                     .attr<std::vector<int32_t>>("pool_params", {0, 2, 2, 1, 1, 0, 0})
                     .build();

    auto out = run_on_cpu(model, {{"x", make_f16_tensor({1, 1, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})}});
    expect_near(out, {5.0f, 6.0f});
}

TEST(GGUFOps, Pool2DRejectsInvalidParams) {
    const std::vector<std::vector<int32_t>> invalid_params{
        {1, -1, 2, 1, 1, 0, 0},
        {1, 2, 0, 1, 1, 0, 0},
        {1, 2, 2, -1, 1, 0, 0},
        {1, 2, 2, 1, 0, 0, 0},
        {1, 2, 2, 1, 1, -1, 0},
        {1, 2, 2, 1, 1, 0, -1},
        {1, 2, 2, 1, 1, 0},     // missing a parameter
        {2, 2, 2, 1, 1, 0, 0},  // unknown pooling op
    };
    for (const auto& params : invalid_params) {
        EXPECT_THROW(SingleOpBuilder()
                         .op("GGML_OP_POOL_2D")
                         .input("x", ov::element::f32, {1, 1, 2, 2})
                         .output("out", ov::element::f32, {1, 1, 1, 1})
                         .attr<std::vector<int32_t>>("pool_params", params)
                         .build(),
                     ov::Exception);
    }
}

TEST(GGUFOps, RollFourAxes) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROLL")
                     .input("x", ov::element::f32, {2, 2, 2, 3})
                     .output("out", ov::element::f32, {2, 2, 2, 3})
                     // ggml op_params order (ggml dim 0 first): OV axes {0, 1, 2, 3} shift by {1, -1, 1, -1}.
                     .attr<std::vector<int32_t>>("roll_shifts", {-1, 1, -1, 1})
                     .build();
    std::vector<float> input(24);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
    }
    std::vector<float> expected(24);
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 3; ++d) {
                    const int src_a = (a - 1 + 2) % 2;
                    const int src_b = (b + 1) % 2;
                    const int src_c = (c - 1 + 2) % 2;
                    const int src_d = (d + 1) % 3;
                    expected[((a * 2 + b) * 2 + c) * 3 + d] = input[((src_a * 2 + src_b) * 2 + src_c) * 3 + src_d];
                }
            }
        }
    }
    expect_near(run_on_cpu(model, {{"x", make_f32_tensor({2, 2, 2, 3}, input)}}), expected, 0.0f);
}

TEST(GGUFOps, SolveTriBatched) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_SOLVE_TRI")
                       .input("a", ov::element::f32, {2, 1, 3, 3})
                       .input("b", ov::element::f32, {2, 1, 3, 2})
                       .output("out", ov::element::f32, {2, 1, 3, 2})
                       .attr<std::vector<int32_t>>("solve_tri_params", {1, 1, 0});
    auto model = builder.build();
    std::vector<float> a{2, 0, 0, 3, 1, 0, 1, -1, 2, 1, 0, 0, 2, 2, 0, -1, 1, 1};
    std::vector<float> b{2, 4, 5, 7, 3, 1, 1, 3, 4, 8, 2, 5};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 1, 3, 3}, a)}, {"b", make_f32_tensor({2, 1, 3, 2}, b)}});
    expect_near(out, {1, 2, 2, 1, 2, 0, 1, 3, 1, 1, 2, 7});

    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_SOLVE_TRI")
                     .input("a", ov::element::f32, {1, 1, 2, 2})
                     .input("b", ov::element::f32, {1, 1, 2, 1})
                     .output("out", ov::element::f32, {1, 1, 2, 1})
                     .attr<std::vector<int32_t>>("solve_tri_params", {0, 1, 0})
                     .build(),
                 ov::Exception);
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

TEST(GGUFOps, DiagKeepsInputPort) {
    using namespace ov::frontend::gguf;
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 4});
    auto k = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto topk = std::make_shared<ov::op::v11::TopK>(data,
                                                    k,
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i32);
    auto decoder = SingleOpBuilder()
                       .op("GGML_OP_DIAG")
                       .input("x", ov::element::i32, {1, 1, 1, 2})
                       .output("out", ov::element::i32, {1, 1, 2, 2})
                       .decoder();
    auto tensors = std::make_shared<TensorMap>();
    (*tensors)["x"] = topk->output(1);
    NodeContext context(decoder, tensors);
    auto diag = op::translate_diag(context)[0];
    auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(diag.get_node_shared_ptr());
    ASSERT_NE(multiply, nullptr);
    auto eye = ov::as_type_ptr<ov::op::v9::Eye>(multiply->input_value(1).get_node_shared_ptr());
    ASSERT_NE(eye, nullptr);
    auto shape_of = eye->input_value(0).get_node_shared_ptr()->input_value(0).get_node_shared_ptr();
    ASSERT_NE(shape_of, nullptr);
    EXPECT_EQ(shape_of->input_value(0).get_node_shared_ptr(), topk);
    EXPECT_EQ(shape_of->input_value(0).get_index(), 1u);
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

// Div: plain element-wise divide when both inputs already share a shape.
TEST(GGUFOps, Div) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIV")
                     .input("a", ov::element::f32, {2, 3})
                     .input("b", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();

    std::vector<float> a{1, 2, 3, 4, 5, 6};
    std::vector<float> b{2, 4, 3, 8, 5, 12};
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({2, 3}, a)}, {"b", make_f32_tensor({2, 3}, b)}});

    std::vector<float> expected(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        expected[i] = a[i] / b[i];
    expect_near(out, expected);
}

// Div with ggml-style integer-repeat broadcast: the denominator [1,2] repeats to match [1,4].
TEST(GGUFOps, DivBroadcast) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIV")
                     .input("a", ov::element::f32, {1, 4})
                     .input("b", ov::element::f32, {1, 2})
                     .output("out", ov::element::f32, {1, 4})
                     .build();

    std::vector<float> a{2, 4, 6, 8};
    std::vector<float> b{2, 4};  // repeats to {2, 4, 2, 4}
    auto out = run_on_cpu(model, {{"a", make_f32_tensor({1, 4}, a)}, {"b", make_f32_tensor({1, 2}, b)}});

    expect_near(out, {1, 1, 3, 2});
}

// Only the Divide stays in f32 under f16 inference: the MoE weight renormalization (Gemma 4 26B)
// is multiplied by a constant per-expert scale, which must not end up in a different precision.
TEST(GGUFOps, DivKeepsOnlyItselfInF32) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIV")
                     .input("a", ov::element::f32, {2, 3})
                     .input("b", ov::element::f32, {2, 3})
                     .output("out", ov::element::f32, {2, 3})
                     .build();
    auto result = model->get_results()[0];
    auto div = result->get_input_node_shared_ptr(0);
    ASSERT_TRUE(ov::is_type<ov::op::v1::Divide>(div));
    auto scale = ov::op::v0::Constant::create(ov::element::f32, {2, 3}, {1, 2, 3, 4, 5, 6});
    result->input(0).replace_source_output(std::make_shared<ov::op::v1::Multiply>(div, scale));
    model->validate_nodes_and_infer_types();

    // The configuration the GPU plugin uses for f16 inference.
    ov::pass::ConvertPrecision({{ov::element::f32, ov::element::f16}}, {}, true, false, true).run_on_model(model);

    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_TRUE(ov::is_conversion_disabled(div, ov::element::f16));
    EXPECT_EQ(div->get_output_element_type(0), ov::element::f32);
    for (const auto& param : model->get_parameters()) {
        EXPECT_FALSE(ov::is_conversion_disabled(param, ov::element::f16)) << param->get_friendly_name();
    }
}

// Div over an empty token axis, as produced by a non-final chunked-prefill chunk (see
// MulMatIdEmptyTokens). Both operands are empty, so the ggml-style repeat has nothing to repeat and
// must be skipped rather than computing a 0/0 repeat count.
TEST(GGUFOps, DivEmptyTokens) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIV")
                     .input("a", ov::element::f32, {ov::Dimension::dynamic(), 4})
                     .input("b", ov::element::f32, {ov::Dimension::dynamic(), 4})
                     .output("out", ov::element::f32, {ov::Dimension::dynamic(), 4})
                     .build();

    auto out = run_on_cpu(
        model,
        {{"a", ov::Tensor(ov::element::f32, ov::Shape{0, 4})}, {"b", ov::Tensor(ov::element::f32, ov::Shape{0, 4})}});

    EXPECT_EQ(out.get_shape(), (ov::Shape{0, 4}));
}

// Set: write src into a contiguous region of dst (flattened) starting at set_offset_elems.
TEST(GGUFOps, Set) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SET")
                     .input("dst", ov::element::f32, {2, 4})
                     .input("src", ov::element::f32, {1, 2})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<int64_t>("set_offset_elems", 2)
                     .build();

    std::vector<float> dst(8, 1.0f);
    std::vector<float> src{10, 20};
    auto out = run_on_cpu(model, {{"dst", make_f32_tensor({2, 4}, dst)}, {"src", make_f32_tensor({1, 2}, src)}});

    // Flattened dst with src written at [2,3], reshaped back to [2,4].
    expect_near(out, {1, 1, 10, 20, 1, 1, 1, 1});
}

TEST(GGUFOps, AddMixedPrecisionPreservesLeftType) {
    for (auto type : {ov::element::f32, ov::element::f16}) {
        const auto rhs_type = type == ov::element::f32 ? ov::element::f16 : ov::element::f32;
        auto model = SingleOpBuilder()
                         .op("GGML_OP_ADD")
                         .input("a", type, {1, 3})
                         .input("b", rhs_type, {1, 3})
                         .output("out", type, {1, 3})
                         .build();
        auto a = type == ov::element::f32 ? make_f32_tensor({1, 3}, {1, 2, 3}) : make_f16_tensor({1, 3}, {1, 2, 3});
        auto b = rhs_type == ov::element::f32 ? make_f32_tensor({1, 3}, {0.5f, -1, 2})
                                              : make_f16_tensor({1, 3}, {0.5f, -1, 2});
        auto out = run_on_cpu(model, {{"a", a}, {"b", b}});
        ASSERT_EQ(out.get_element_type(), type);
        const std::vector<float> expected{1.5f, 1, 5};
        for (size_t i = 0; i < expected.size(); ++i) {
            const float value = type == ov::element::f32 ? out.data<float>()[i] : float(out.data<ov::float16>()[i]);
            EXPECT_EQ(value, expected[i]);
        }
    }
}

TEST(GGUFOps, SwigluClampF16RoundsOnce) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU_CLAMP")
                     .input("gate", ov::element::f16, {1, 6})
                     .input("up", ov::element::f16, {1, 6})
                     .output("out", ov::element::f16, {1, 6})
                     .attr<bool>("swapped", false)
                     .attr<float>("glu_limit", 3.0f)
                     .build();
    auto gate = make_f16_tensor({1, 6}, {-2.3f, -0.7f, 0.3f, 1.7f, 3.7f, -5.1f});
    auto up = make_f16_tensor({1, 6}, {2.7f, -1.3f, 0.7f, -4.1f, 1.3f, 3.4f});
    ov::TensorVector outputs{ov::Tensor(ov::element::f16, {1, 6})};
    ASSERT_TRUE(model->evaluate(outputs, {gate, up}));
    for (size_t i = 0; i < 6; ++i) {
        const float g = std::min(float(gate.data<ov::float16>()[i]), 3.0f);
        const float u = std::clamp(float(up.data<ov::float16>()[i]), -3.0f, 3.0f);
        const ov::float16 expected(g / (1.0f + std::exp(-g)) * u);
        EXPECT_EQ(outputs[0].data<ov::float16>()[i], expected);
    }
}

TEST(GGUFOps, PermuteCase1RankThree) {
    for (const auto& perm : {std::vector<int64_t>{0, 2, 1, 3}, std::vector<int64_t>{1, 0, 2}}) {
        auto model = SingleOpBuilder()
                         .op("GGML_OP_PERMUTE")
                         .input("x", ov::element::f32, {-1, 3, 2})
                         .output("out", ov::element::f32, {3, -1, 2})
                         .attr<std::vector<int64_t>>("perm", perm)
                         .build();
        auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})}});
        ASSERT_EQ(out.get_shape(), (ov::Shape{3, 2, 2}));
        expect_near(out, {0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11});
    }
}

TEST(GGUFOps, DiagDynamicWidth) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DIAG")
                     .input("x", ov::element::f32, {1, 2, 1, -1})
                     .output("out", ov::element::f32, {1, 2, -1, -1})
                     .build();
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 1, 2}, {2, 3, 5, 7})}});
    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 2, 2, 2}));
    expect_near(out, {2, 0, 0, 3, 5, 0, 0, 7});
}

TEST(GGUFOps, DivSiluAtZero) {
    using namespace ov::frontend::gguf;
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3});
    x->output(0).set_names({"x"});
    auto silu_decoder = SingleOpBuilder()
                            .op("GGML_UNARY_OP_SILU")
                            .input("x", ov::element::f32, {1, 3})
                            .output("silu", ov::element::f32, {1, 3})
                            .decoder();
    auto tensors = std::make_shared<TensorMap>();
    (*tensors)["x"] = x;
    NodeContext silu_context(silu_decoder, tensors);
    (*tensors)["silu"] = op::translate_unary_silu(silu_context)[0];
    auto decoder = SingleOpBuilder()
                       .op("GGML_OP_DIV")
                       .input("silu", ov::element::f32, {1, 3})
                       .input("x", ov::element::f32, {1, 3})
                       .output("out", ov::element::f32, {1, 3})
                       .decoder();
    NodeContext context(decoder, tensors);
    auto model = std::make_shared<ov::Model>(op::translate_div(context), ov::ParameterVector{x});
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 3}, {0, -1, 1})}});
    expect_near(out, {0.5f, 1.0f / (1.0f + std::exp(1.0f)), 1.0f / (1.0f + std::exp(-1.0f))});
}

TEST(GGUFOps, RopeOffsetPreservesPrefixAndTail) {
    // ggml CPU oracle: rope_ext [8,2,2,1], dims=4, offset=2, positions={2,5}; normal and NEOX.
    const std::vector<std::vector<float>> expected{
        {-2, -1.75f, 1.76084197f,   -0.843762517f, -0.984801054f, -0.769848704f, -0.5f, -0.25f,
         0,  0.25f,  -0.890046477f, 0.142538577f,  0.974801719f,  1.26974869f,   1.5f,  1.75f,
         2,  2.25f,  3.34619737f,   -1.61723971f,  2.83381844f,   3.39587569f,   3.5f,  3.75f,
         4,  4.25f,  5.83137035f,   -2.9677639f,   4.73136091f,   5.49333477f,   5.5f,  5.75f},
        {-2, -1.75f, 1.53351772f,  -1.23475099f, -0.947799265f, -0.774848342f, -0.5f, -0.25f,
         0,  0.25f,  -1.11737084f, 0.724851668f, 0.0385018587f, 1.26474905f,   1.5f,  1.75f,
         2,  2.25f,  3.58592844f,  2.584131f,    -1.54632413f,  3.38338089f,   3.5f,  3.75f,
         4,  4.25f,  6.07110119f,  4.48167324f,  -2.8968482f,   5.48083973f,   5.5f,  5.75f}};
    for (int mode = 0; mode < 2; ++mode) {
        RopeConfig cfg;
        cfg.n_dims = 4;
        cfg.n_ctx_orig = 4096;
        cfg.freq_base = 10000;
        cfg.freq_scale = 1;
        cfg.attn_factor = 1;
        auto model = SingleOpBuilder()
                         .op("GGML_OP_ROPE")
                         .input("data", ov::element::f32, {1, 2, 2, 8})
                         .input("pos", ov::element::i32, {1, 1, 1, 2})
                         .output("out", ov::element::f32, {1, 2, 2, 8})
                         .attr<std::string>("rope_mode", mode == 0 ? "normal" : "neox")
                         .attr<RopeConfig>("rope_config", cfg)
                         .attr<int64_t>("rope_offset", 2)
                         .build();
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = (float(i) - 8) * 0.25f;
        ov::Tensor pos(ov::element::i32, {1, 1, 1, 2});
        pos.data<int32_t>()[0] = 2;
        pos.data<int32_t>()[1] = 5;
        auto out = run_on_cpu(model, {{"data", make_f32_tensor({1, 2, 2, 8}, data)}, {"pos", pos}});
        expect_near(out, expected[mode]);
    }
}

TEST(GGUFOps, RopeInvalidOffsetThrows) {
    for (int64_t offset : {-2, 1, 6}) {
        RopeConfig cfg;
        cfg.n_dims = 4;
        cfg.freq_base = 10000;
        cfg.freq_scale = 1;
        cfg.attn_factor = 1;
        EXPECT_ANY_THROW(SingleOpBuilder()
                             .op("GGML_OP_ROPE")
                             .input("data", ov::element::f32, {1, 2, 2, 8})
                             .input("pos", ov::element::i32, {1, 1, 1, 2})
                             .output("out", ov::element::f32, {1, 2, 2, 8})
                             .attr<RopeConfig>("rope_config", cfg)
                             .attr<int64_t>("rope_offset", offset)
                             .build());
    }
}

TEST(GGUFOps, ContDynamicLayoutRequiresReshape) {
    {
        auto builder = SingleOpBuilder()
                           .op("GGML_OP_CONT")
                           .input("x", ov::element::f32, {1, -1, 3, 2})
                           .output("out", ov::element::f32, {1, 3, -1, 2});
        EXPECT_THROW(builder.build(), ov::Exception);
        auto model = builder.attr<std::vector<int64_t>>("cont_reshape", {1, 3, -1, 2}).build();
        std::vector<float> values{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 3, 2}, values)}});
        EXPECT_EQ(out.get_shape(), (ov::Shape{1, 3, 2, 2}));
        expect_near(out, values);

        auto without_metadata = builder.output("out", ov::element::f32, ov::PartialShape::dynamic()).build();
        auto reshaped = run_on_cpu(without_metadata, {{"x", make_f32_tensor({1, 2, 3, 2}, values)}});
        EXPECT_EQ(reshaped.get_shape(), (ov::Shape{1, 3, 2, 2}));
        expect_near(reshaped, values);
    }
}

TEST(GGUFOps, ContDynamicPassThroughPreservesProducer) {
    {
        auto model = SingleOpBuilder()
                         .op("GGML_OP_CONT")
                         .input("x", ov::element::f32, {1, -1, 3, 2})
                         .output("out", ov::element::f32, {1, -1, 3, 2})
                         .build();
        EXPECT_EQ(model->get_results()[0]->input_value(0).get_node_shared_ptr(), model->get_parameters()[0]);
        EXPECT_EQ(model->get_parameters()[0]->get_friendly_name(), "x");
        auto without_metadata = SingleOpBuilder()
                                    .op("GGML_OP_CONT")
                                    .input("x", ov::element::f32, {1, -1, 3, 2})
                                    .output("out", ov::element::f32, ov::PartialShape::dynamic())
                                    .build();
        EXPECT_EQ(without_metadata->get_results()[0]->input_value(0).get_node_shared_ptr(),
                  without_metadata->get_parameters()[0]);
    }
}

TEST(GGUFOps, Pool2DF16AverageWithoutOutputType) {
    std::shared_ptr<ov::frontend::gguf::GgufDecoder> decoder = std::make_shared<SingleOpDecoder>(
        "GGML_OP_POOL_2D",
        std::vector<TensorDesc>{{"x", ov::element::f16, {1, 1, 2, 2}}},
        std::vector<TensorDesc>{},
        TensorDesc{"out", ov::element::f32, {1, 1, 1, 1}},
        std::map<std::string, ov::Any>{{"pool_params", std::vector<int32_t>{1, 2, 2, 1, 1, 0, 0}}});
    ov::frontend::gguf::FrontEnd frontend;
    auto model = frontend.convert(frontend.load(decoder));
    auto input = make_f16_tensor({1, 1, 2, 2}, {2048, 1, 0, 0});
    // ggml CPU oracle: F16 2x2 AVG pool of {2048,1,0,0}; F16 accumulation/output loses 0.25.
    auto out = run_on_cpu(model, {{"x", input}});
    ASSERT_EQ(out.get_element_type(), ov::element::f32);
    EXPECT_FLOAT_EQ(out.data<float>()[0], 512.25f);
}

TEST(GGUFOps, SolveTriDynamicBatchDimensions) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOLVE_TRI")
                     .input("a", ov::element::f32, {-1, -1, 3, 3})
                     .input("b", ov::element::f32, {-1, -1, 3, 2})
                     .output("out", ov::element::f32, {-1, -1, 3, 2})
                     .attr<std::vector<int32_t>>("solve_tri_params", {1, 1, 0})
                     .build();
    const std::vector<float> a{2, 0, 0, 3, 1, 0, 1, -1, 2, 1, 0, 0, 2, 2, 0, -1, 1, 1};
    const std::vector<float> b{2, 4, 5, 7, 3, 1, 1, 3, 4, 8, 2, 5};
    for (size_t batch : {1, 2}) {
        auto out = run_on_cpu(
            model,
            {{"a", make_f32_tensor({batch, 2 / batch, 3, 3}, a)}, {"b", make_f32_tensor({batch, 2 / batch, 3, 2}, b)}});
        EXPECT_EQ(out.get_shape(), (ov::Shape{batch, 2 / batch, 3, 2}));
        expect_near(out, {1, 2, 2, 1, 2, 0, 1, 3, 1, 1, 2, 7});
    }
}

}  // namespace

TEST(GGUFOps, MultimodalRopeMatchesIndependentSections) {
    // Reference: mmproj_ops_oracle.cpp, pinned ggml CPU. Three heads, differing
    // coordinates on all four axes, and unequal IMROPE sections exercise layout and routing.
    for (bool vision : {false, true}) {
        RopeConfig config;
        config.n_dims = vision ? 32 : 64;
        config.n_ctx_orig = 32768;
        config.freq_base = 10000.f;
        config.freq_scale = config.attn_factor = 1.f;
        config.beta_fast = 32.f;
        config.beta_slow = 1.f;
        config.sections = vision ? std::array<int32_t, 4>{16, 16, 16, 16} : std::array<int32_t, 4>{8, 7, 9, 8};
        auto model = SingleOpBuilder()
                         .op("GGML_OP_ROPE")
                         .input("data", ov::element::f32, {1, -1, 3, 64})
                         .input("pos", ov::element::i32, {1, 1, 1, -1})
                         .output("out", ov::element::f32, {1, -1, 3, 64})
                         .attr<std::string>("rope_mode", vision ? "vision" : "imrope")
                         .attr<RopeConfig>("rope_config", config)
                         .build();
        std::vector<float> data(384);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::sin(float(i) * 0.13f);
        ov::Tensor positions(ov::element::i32, {1, 1, 1, 8});
        const std::vector<int32_t> values{3, 7, 11, 2, 5, 13, 17, 19};
        std::copy(values.begin(), values.end(), positions.data<int32_t>());
        auto actual = run_on_cpu(model, {{"data", make_f32_tensor({1, 2, 3, 64}, data)}, {"pos", positions}});
        expect_near(actual, load_npy<float>(vision ? "vision_rope_expected" : "multimodal_imrope_expected"));
    }
}

TEST(GGUFOps, InterpolateBilinearAntialiasDynamicSize) {
    for (bool corners : {false, true}) {
        // Reference: mmproj_ops_oracle.cpp resize modes, ggml CPU.
        auto model = SingleOpBuilder()
                         .op("GGML_OP_UPSCALE")
                         .input("data", ov::element::f32, {1, 2, -1, -1})
                         .input("sizes", ov::element::i64, {2})
                         .output("out", ov::element::f32, {1, 2, -1, -1})
                         .attr<int>("interpolation_mode", 1 | 0x200 | (corners ? 0x100 : 0))
                         .build();
        std::vector<float> data(12);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::sin(float(i) * 0.13f);
        ov::Tensor sizes(ov::element::i64, {2});
        sizes.data<int64_t>()[0] = 4;
        sizes.data<int64_t>()[1] = 5;
        auto actual = run_on_cpu(model, {{"data", make_f32_tensor({1, 2, 2, 3}, data)}, {"sizes", sizes}});
        expect_near(actual,
                    load_npy<float>(corners ? "mmproj_interpolate_corners_expected" : "mmproj_interpolate_expected"));
        // Downsampling exercises clipped filter support at the border of a learned position grid.
        data.resize(2 * 8 * 12);
        for (size_t i = 0; i < data.size(); ++i)
            data[i] = std::sin(float(i) * 0.13f);
        actual = run_on_cpu(model, {{"data", make_f32_tensor({1, 2, 8, 12}, data)}, {"sizes", sizes}});
        expect_near(
            actual,
            load_npy<float>(corners ? "mmproj_interpolate_down_corners_expected" : "mmproj_interpolate_down_expected"));
    }
}

TEST(GGUFOps, Im2colDynamicRectangularGridsMatchCPU) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_IM2COL")
                     .input("kernel", ov::element::f32, {1, 2, 2, 3})
                     .input("image", ov::element::f32, {1, 2, -1, -1})
                     .output("out", ov::element::f32, {1, -1, -1, 12})
                     .attr<std::vector<int32_t>>("im2col_params", {2, 1, 1, 0, 1, 1, 1})
                     .build();
    ov::Core core;
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    for (size_t width : {7, 9}) {
        const size_t height = width == 7 ? 5 : 4;
        std::vector<float> values(2 * height * width);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = std::sin(float(i) * 0.13f);
        request.set_tensor("image", make_f32_tensor({1, 2, height, width}, values));
        request.infer();
        EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, height - 1, (width + 1) / 2, 12}));
        expect_near(request.get_output_tensor(), load_npy<float>("mmproj_im2col" + std::to_string(width)));
    }
}

// Non-overlapping patches (stride == kernel, Gemma 4 unified vision) avoid ExtractImagePatches,
// which the GPU plugin can't compile for dynamic image sizes, and crop partial patches the same way.
TEST(GGUFOps, Im2colNonOverlappingPatches) {
    const size_t IC = 2, KH = 2, KW = 3;
    auto model = SingleOpBuilder()
                     .op("GGML_OP_IM2COL")
                     .input("kernel", ov::element::f32, {1, IC, KH, KW})
                     .input("image", ov::element::f32, {1, IC, -1, -1})
                     .output("out", ov::element::f32, {1, -1, -1, IC * KH * KW})
                     .attr<std::vector<int32_t>>("im2col_params", {KW, KH, 0, 0, 1, 1, 1})
                     .build();
    for (const auto& op : model->get_ops()) {
        EXPECT_FALSE(ov::is_type<ov::op::v3::ExtractImagePatches>(op));
    }
    ov::Core core;
    auto request =
        core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    for (const auto& [height, width] : {std::pair<size_t, size_t>{4, 6}, {5, 7}}) {
        std::vector<float> image(IC * height * width);
        for (size_t i = 0; i < image.size(); ++i)
            image[i] = static_cast<float>(i);
        request.set_tensor("image", make_f32_tensor({1, IC, height, width}, image));
        request.infer();
        const size_t OH = height / KH, OW = width / KW;
        std::vector<float> expected;
        for (size_t oh = 0; oh < OH; ++oh)
            for (size_t ow = 0; ow < OW; ++ow)
                for (size_t c = 0; c < IC; ++c)
                    for (size_t kh = 0; kh < KH; ++kh)
                        for (size_t kw = 0; kw < KW; ++kw)
                            expected.push_back(image[(c * height + oh * KH + kh) * width + ow * KW + kw]);
        EXPECT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, OH, OW, IC * KH * KW}));
        expect_near(request.get_output_tensor(), expected);
    }
}

// Geometry expectations generated by mmproj_ops_oracle.cpp against pinned ggml CPU.
TEST(GGUFOps, WindowPartitionPadsRectangularGrid) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_WIN_PART")
                     .input("x", ov::element::f32, {1, -1, -1, 3})
                     .output("out", ov::element::f32, {-1, 2, 2, 3})
                     .attr<int64_t>("window", 2)
                     .build();
    auto result = run_on_cpu(model, {{"x", make_f32_tensor({1, 3, 5, 3}, load_npy<float>("mmproj_window_input"))}});
    EXPECT_EQ(result.get_shape(), (ov::Shape{6, 2, 2, 3}));
    expect_near(result, load_npy<float>("mmproj_windows"));
}

TEST(GGUFOps, WindowUnpartitionRemovesPadding) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_WIN_UNPART")
                     .input("x", ov::element::f32, {-1, 2, 2, 3})
                     .input("reference", ov::element::f32, {1, -1, -1, 3})
                     .output("out", ov::element::f32, {1, -1, -1, 3})
                     .attr<int64_t>("window", 2)
                     .build();
    auto result = run_on_cpu(model,
                             {{"x", make_f32_tensor({6, 2, 2, 3}, load_npy<float>("mmproj_windows"))},
                              {"reference", make_f32_tensor({1, 3, 5, 3}, load_npy<float>("mmproj_window_input"))}});
    expect_near(result, load_npy<float>("mmproj_restored"));
}

TEST(GGUFOps, IndexedRelativePositionsMatchCPU) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_GET_REL_POS")
                     .input("table", ov::element::f32, {3, 4})
                     .input("indices", ov::element::i32, {1, 3, 3})
                     .output("out", ov::element::f32, {3, 3, 4})
                     .build();
    std::vector<int32_t> indices;
    for (int32_t q = 0; q < 3; ++q)
        for (int32_t k = 0; k < 3; ++k)
            indices.push_back(q - k + 2);
    ov::Tensor ids(ov::element::i32, {1, 3, 3});
    std::copy(indices.begin(), indices.end(), ids.data<int32_t>());
    auto result =
        run_on_cpu(model,
                   {{"table", make_f32_tensor({3, 4}, load_npy<float>("mmproj_relative_table"))}, {"indices", ids}});
    expect_near(result, load_npy<float>("mmproj_relative"));
}

TEST(GGUFOps, ReferenceReshapeTracksSpatialDimensions) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_RESHAPE")
                     .input("x", ov::element::f32, {1, 1, 3, -1})
                     .input("reference", ov::element::f32, {1, 3, -1, -1})
                     .output("out", ov::element::f32, {1, 3, -1, -1})
                     .attr<std::vector<int64_t>>("reshape_target", {1, 3, 0, 0})
                     .attr<std::vector<int64_t>>("shape_axes", {-1, -1, 2, 3})
                     .build();
    const auto values = load_npy<float>("mmproj_window_input");
    auto result = run_on_cpu(
        model,
        {{"x", make_f32_tensor({1, 1, 3, 15}, values)}, {"reference", make_f32_tensor({1, 3, 3, 5}, values)}});
    EXPECT_EQ(result.get_shape(), (ov::Shape{1, 3, 3, 5}));
    expect_near(result, values);
}

TEST(GGUFOps, ViewRawStridesDenseCropAndDynamicReuse) {
    // First exercise a dense byte-offset slice whose output uses a different row width.
    auto dense = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {1, 1, 3, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 3})
                     .attr("view_input_strides", std::vector<int64_t>{4, 16, 48, 48})
                     .attr("view_output_strides", std::vector<int64_t>{4, 12, 24, 24})
                     .attr("view_offset_bytes", int64_t{8})
                     .build();
    auto dense_out =
        run_on_cpu(dense, {{"x", make_f32_tensor({1, 1, 3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})}});
    ASSERT_EQ(dense_out.get_shape(), (ov::Shape{1, 1, 2, 3}));
    expect_near(dense_out, {2, 3, 4, 5, 6, 7}, 0.0f, 0.0f);

    // Dense re-layout, rectangular crop, fixed capture-size crop, and stepped columns.
    for (int variant : {0, 1, 2, 3}) {
        SCOPED_TRACE(variant);
        const bool fixed = variant == 2;
        const int64_t captured_tokens = fixed ? 1 : 2;
        const ov::PartialShape captured{1, 1, captured_tokens, 6};
        const ov::PartialShape output = variant == 0 ? ov::PartialShape{1, 2, 3, 2}
                                                    : ov::PartialShape{1, 1, captured_tokens, fixed ? 6 : 3};
        const std::vector<int64_t> input_strides{4, 24, 24 * captured_tokens, 24 * captured_tokens};
        const auto output_strides = variant == 0   ? std::vector<int64_t>{4, 8, 24, 48}
                                    : variant == 3 ? std::vector<int64_t>{8, 24, 48, 48}
                                                   : input_strides;
        auto builder = SingleOpBuilder()
                           .op("GGML_OP_VIEW")
                           .input("x", ov::element::f32, {1, 1, ov::Dimension(1, -1), 6})
                           .output("out", ov::element::f32, output)
                           .attr("view_input_strides", input_strides)
                           .attr("view_output_strides", output_strides)
                           .attr("view_offset_bytes", int64_t{variant == 1 ? 4 : 0})
                           .attr("output_dynamic_axis", fixed ? -1 : variant == 0 ? 1 : 2);
        auto model = build_captured_op(builder, {{"x", captured}});
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        for (size_t tokens : {2u, 5u, 1u}) {
            SCOPED_TRACE(tokens);
            std::vector<float> values(tokens * 6);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(100 * tokens + i);
            request.set_tensor("x", make_f32_tensor({1, 1, tokens, 6}, values));
            request.infer();
            const ov::Shape expected_shape = variant == 0 ? ov::Shape{1, tokens, 3, 2}
                                            : fixed      ? ov::Shape{1, 1, 1, 6}
                                                         : ov::Shape{1, 1, tokens, 3};
            ASSERT_EQ(request.get_output_tensor().get_shape(), expected_shape);
            std::vector<float> expected;
            if (variant == 0) {
                expected = values;
            } else if (fixed) {
                expected.assign(values.begin(), values.begin() + 6);
            } else {
                for (size_t t = 0; t < tokens; ++t)
                    for (size_t d = 0; d < 3; ++d)
                        expected.push_back(values[t * 6 + (variant == 1 ? d + 1 : 2 * d)]);
            }
            expect_near(request.get_output_tensor(), expected, 0.0f, 0.0f);
        }
    }
}

TEST(GGUFOps, ViewRawInvalidStridesAreRejected) {
    for (const auto& strides : {std::vector<int64_t>{6, 16, 32, 32},
                                std::vector<int64_t>{0, 16, 32, 32},
                                std::vector<int64_t>{-4, 16, 32, 32}}) {
        SCOPED_TRACE(::testing::PrintToString(strides));
        try {
            SingleOpBuilder()
                .op("GGML_OP_VIEW")
                .input("x", ov::element::f32, {1, 1, 2, 4})
                .output("out", ov::element::f32, {1, 1, 2, 2})
                .attr("view_input_strides", std::vector<int64_t>{4, 16, 32, 32})
                .attr("view_output_strides", strides)
                .attr("view_offset_bytes", int64_t{0})
                .build();
            FAIL() << "Invalid VIEW strides were accepted";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("Unsupported VIEW layout"), std::string::npos) << error.what();
        }
        try {
            SingleOpBuilder()
                .op("GGML_OP_VIEW")
                .input("x", ov::element::f32, {1, 1, 4, 4})
                .output("out", ov::element::f32, {1, 1, 2, 3})
                .attr("view_input_strides", std::vector<int64_t>{4, 16, 64, 64})
                .attr("view_output_strides", std::vector<int64_t>{4, 16, 64, 64})
                .attr("view_offset_bytes", int64_t{8})
                .build();
            FAIL() << "A row-crossing VIEW must not be silently flattened to contiguous elements 2..7";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("Unsupported VIEW layout"), std::string::npos) << error.what();
        }
    }
}

// A non-dense view of a dense source (columns 2..5 of every [6]-wide token row, as a merged QKV
// projection slice [head_size=2, n_head=2, tokens]) is regrouped into a stride grid and sliced.
TEST(GGUFOps, ViewNonDenseSliceOfDenseSource) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_VIEW")
                       .input("x", ov::element::f32, {1, 1, ov::Dimension(1, -1), 6})
                       .output("out", ov::element::f32, {1, 2, 2, 2})
                       .attr("view_input_strides", std::vector<int64_t>{4, 24, 48, 48})
                       .attr("view_output_strides", std::vector<int64_t>{4, 8, 24, 48})
                       .attr("view_offset_bytes", int64_t{8})
                       .attr("output_dynamic_axis", 1);
    auto model = build_captured_op(builder, {{"x", {1, 1, 2, 6}}});
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t tokens : {2u, 5u, 1u}) {
        SCOPED_TRACE(tokens);
        std::vector<float> values(tokens * 6), expected;
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(100 * tokens + i);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t c = 2; c < 6; ++c)
                expected.push_back(values[t * 6 + c]);
        request.set_tensor("x", make_f32_tensor({1, 1, tokens, 6}, values));
        request.infer();
        ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, tokens, 2, 2}));
        expect_near(request.get_output_tensor(), expected, 0.0f, 0.0f);
    }
}

// A KV-cache read with runtime bounds: the live length and the first stream are operands, so one
// compiled model serves every batch. Cache [streams=2, cells=3, features=4], heads 2 x head_size 2.
TEST(GGUFOps, ViewKvCacheRuntimeBounds) {
    const size_t streams = 2, cells = 3, width = 4, heads = 2, head_size = 2;
    std::vector<float> cache(streams * cells * width);
    for (size_t i = 0; i < cache.size(); ++i)
        cache[i] = static_cast<float>(i);
    for (bool transposed : {false, true}) {
        SCOPED_TRACE(transposed);
        // Transposed (`-fa off` V): each stream holds `cells` values per feature, cells innermost, and the
        // view's OV order is [streams, head_size, heads, cells].
        const std::vector<int64_t> order =
            transposed ? std::vector<int64_t>{0, 3, 2, 1} : std::vector<int64_t>{0, 1, 2, 3};
        auto model = SingleOpBuilder()
                         .op("GGML_OP_VIEW")
                         .input("cache", ov::element::f32, {1, streams, cells, width})
                         .input("length", ov::element::i64, {1})
                         .input("first", ov::element::i64, {1})
                         .output("out", ov::element::f32, {1, -1, heads, head_size})
                         .attr("kv_view",
                               std::vector<int64_t>{2, 4, 2, 2, 1, 0, transposed ? 1 : 0, order[0], order[1], order[2],
                                                    order[3], 3})
                         .build();
        ov::Core core;
        auto request =
            core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
        for (auto bounds : {std::pair<int64_t, int64_t>{2, 1}, {3, 0}, {1, 1}}) {
            const int64_t length = bounds.first, first = bounds.second;
            SCOPED_TRACE(length);
            ov::Tensor length_tensor(ov::element::i64, {1});
            ov::Tensor first_tensor(ov::element::i64, {1});
            length_tensor.data<int64_t>()[0] = length;
            first_tensor.data<int64_t>()[0] = first;
            request.set_tensor("cache", make_f32_tensor({1, streams, cells, width}, cache));
            request.set_tensor("length", length_tensor);
            request.set_tensor("first", first_tensor);
            request.infer();
            const auto at = [&](size_t c, size_t h, size_t d) {
                const size_t base = static_cast<size_t>(first) * cells * width;
                return transposed ? cache[base + (h * head_size + d) * cells + c]
                                  : cache[base + c * width + h * head_size + d];
            };
            std::vector<float> expected;
            if (transposed) {
                for (size_t d = 0; d < head_size; ++d)
                    for (size_t h = 0; h < heads; ++h)
                        for (size_t c = 0; c < static_cast<size_t>(length); ++c)
                            expected.push_back(at(c, h, d));
            } else {
                for (size_t c = 0; c < static_cast<size_t>(length); ++c)
                    for (size_t h = 0; h < heads; ++h)
                        for (size_t d = 0; d < head_size; ++d)
                            expected.push_back(at(c, h, d));
            }
            expect_near(request.get_output_tensor(0), expected, 0.0f, 0.0f);
        }
    }
}

// whisper's flat KV buffer: a static-length read starting at a whole row, laid out [1, heads, cells, dim].
TEST(GGUFOps, ViewKvCacheFlatBufferRowBase) {
    std::vector<float> buffer(4 * 4);
    for (size_t i = 0; i < buffer.size(); ++i)
        buffer[i] = static_cast<float>(i);
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("buffer", ov::element::f32, {1, 1, 1, 16})
                     .output("out", ov::element::f32, {1, 2, 2, 2})
                     .attr("kv_view", std::vector<int64_t>{1, 4, 2, 2, 1, 1, 0, 0, 2, 1, 3, 2})
                     .build();
    auto out = run_on_cpu(model, {{"buffer", make_f32_tensor({1, 1, 1, 16}, buffer)}});
    std::vector<float> expected;
    for (size_t h = 0; h < 2; ++h)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                expected.push_back(buffer[(1 + c) * 4 + h * 2 + d]);
    expect_near(out, expected, 0.0f, 0.0f);
}

TEST(GGUFOps, ScaleRangeComesFromOperands) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {5})
                     .input("begin", ov::element::i64, {1})
                     .input("length", ov::element::i64, {1})
                     .output("out", ov::element::f32, {5})
                     .attr("axis", int64_t{0})
                     .attr("scale", 2.0f)
                     .attr("bias", 1.0f)
                     .build();
    ov::Tensor begin(ov::element::i64, {1});
    ov::Tensor length(ov::element::i64, {1});
    begin.data<int64_t>()[0] = 1;
    length.data<int64_t>()[0] = 3;
    expect_near(run_on_cpu(model,
                           {{"x", make_f32_tensor({5}, {1, 2, 3, 4, 5})},
                            {"begin", begin},
                            {"length", length}}),
                {1, 5, 7, 9, 5});
}

// PERMUTE consumes the logical input described by its operand; head splitting is a separate
// RESHAPE operation rather than a producer-specific repair in this translator.
TEST(GGUFOps, PermuteDescriptorSwapsHeadAndTokenAxes) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_PERMUTE")
                     .input("x", ov::element::f32, {1, 2, 3, 2})
                     .output("out", ov::element::f32, {1, 3, 2, 2})
                     .attr<std::vector<int64_t>>("perm", {0, 2, 1, 3})
                     .build();

    std::vector<float> x(12);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(i);
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 3, 2}, x)}});

    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 3, 2, 2}));
    // PERMUTE is a plain transpose of its operand: out[0,h,t,d] == x[0,t,h,d] == x_flat[t*6 + h*2 + d].
    std::vector<float> expected;
    for (int64_t h = 0; h < 3; ++h)
        for (int64_t t = 0; t < 2; ++t)
            for (int64_t d = 0; d < 2; ++d)
                expected.push_back(static_cast<float>(t * 6 + h * 2 + d));
    expect_near(out, expected, 0.0f);
}

// The permutation is operation data and therefore mandatory; an op-case does not imply a layout.
TEST(GGUFOps, PermuteWithoutDescriptorThrows) {
    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_PERMUTE")
                     .input("x", ov::element::f32, {1, 2, 3, 2})
                     .output("out", ov::element::f32, {1, 3, 2, 2})
                     .build(),
                 ov::Exception);
}

// VIEW applies its descriptor to the operand it receives. Legacy producer-shape metadata must not
// reshape that operand behind the producer's back.
TEST(GGUFOps, ViewDescriptorUsesOperandLayout) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_VIEW")
                     .input("x", ov::element::f32, {1, 1, 2, 4})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .attr<ov::Shape>("input_ggml_shape", ov::Shape{2, 4})
                     .attr<std::vector<int64_t>>("view_slice", {3, 1, 2})
                     .attr<std::vector<int64_t>>("view_reshape", {1, 1, 2, 2})
                     .build();

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});

    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 1, 2, 2}));
    std::vector<float> expected{2, 3, 6, 7};
    expect_near(out, expected, 0.0f);
}

TEST(GGUFOps, RepeatRejectsCapturedOutputAsDescriptor) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_REPEAT")
                       .input("x", ov::element::f32, {1, 1, 1, 3})
                       .output("out", ov::element::f32, {1, 1, 2, 3});
    EXPECT_THROW(builder.build(), ov::Exception);
}

TEST(GGUFOps, RepeatRuntimeTargetComesFromOperand) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_REPEAT")
                     .input("x", ov::element::f32, {1, 1, 1, 3})
                     .input("target", ov::element::f32, {1, 1, -1, 3})
                     .output("out", ov::element::f32, ov::PartialShape::dynamic())
                     .build();

    std::vector<float> x{1, 2, 3};
    ov::Tensor target(ov::element::f32, {1, 1, 2, 3});
    auto out = run_on_cpu(model,
                          {{"x", make_f32_tensor({1, 1, 1, 3}, x)},
                           {"target", target}});
    expect_near(out, {1, 2, 3, 1, 2, 3});
}

TEST(GGUFOps, PadDoesNotTreatCapturedOutputShapeAsIdentity) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_PAD")
                     .input("x", ov::element::f32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 1, 1, 2})
                     .attr<std::vector<int32_t>>("pad_params", {0, 1, 0, 0, 0, 0, 0, 0})
                     .attr<bool>("pad_circular", false)
                     .build();
    EXPECT_EQ(model->output().get_partial_shape(), (ov::PartialShape{1, 1, 1, 3}));
}

TEST(GGUFOps, RopeExplicitModeDoesNotReapplyViewOffset) {
    class ViewDecoder : public SingleOpDecoder {
    public:
        using SingleOpDecoder::SingleOpDecoder;
        int64_t get_input_view_element_offset(const std::string&) const override {
            return 2;
        }
    };
    RopeConfig cfg;
    cfg.n_dims = 4;
    cfg.freq_base = 10000.0f;
    cfg.freq_scale = cfg.attn_factor = 1.0f;
    cfg.n_ctx_orig = 4096;
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<ViewDecoder>(
        "GGML_OP_ROPE",
        std::vector<TensorDesc>{{"data", ov::element::f32, {1, 1, 1, 4}},
                                {"pos", ov::element::i32, {1, 1, 1, 4}}},
        std::vector<TensorDesc>{},
        TensorDesc{"out", ov::element::f32, {1, 1, 1, 4}},
        std::map<std::string, ov::Any>{{"rope_mode", std::string("imrope")}, {"rope_config", cfg}});
    FrontEnd fe;
    auto model = fe.convert(fe.load(decoder));
    ov::Tensor positions(ov::element::i32, {1, 1, 1, 4});
    std::fill_n(positions.data<int32_t>(), 4, 3);
    const std::vector<float> data{1, 2, 3, 4};
    std::vector<float> c, s;
    neox_ref_cos_sin(4, cfg.freq_base, 1.0f, 1.0f, 3, c, s);
    expect_near(run_on_cpu(model, {{"data", make_f32_tensor({1, 1, 1, 4}, data)}, {"pos", positions}}),
                {data[0] * c[0] - data[2] * s[0],
                 data[1] * c[1] - data[3] * s[1],
                 data[0] * s[0] + data[2] * c[0],
                 data[1] * s[1] + data[3] * c[1]});
}

TEST(GGUFOps, RopeRejectsUnknownExplicitMode) {
    RopeConfig cfg;
    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, 1, 1, 4})
                     .input("pos", ov::element::i32, {1, 1, 1, 1})
                     .output("out", ov::element::f32, {1, 1, 1, 4})
                     .attr("rope_mode", std::string("unsupported"))
                     .attr("rope_config", cfg)
                     .build(),
                 std::exception);
}

TEST(GGUFOps, RopeDoesNotRepairNonCanonicalOperandRank) {
    RopeConfig cfg;
    cfg.n_dims = 4;
    cfg.freq_base = 10000.0f;
    cfg.freq_scale = cfg.attn_factor = 1.0f;
    cfg.n_ctx_orig = 4096;
    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_ROPE")
                     .input("data", ov::element::f32, {1, 2, 4})
                     .input("pos", ov::element::i32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 2, 1, 4})
                     .attr("rope_mode", std::string("neox"))
                     .attr("rope_config", cfg)
                     .build(),
                 std::exception);
}

TEST(GGUFOps, MulMatMatrixWeightsWithDynamicActivations) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_MUL_MAT")
                     .input("b", ov::element::f32, {2, -1})
                     .input("a", ov::element::f32, {1, -1, -1, -1})
                     .output("out", ov::element::f32, ov::PartialShape::dynamic())
                     .build();
    for (size_t width : {size_t{3}, size_t{5}}) {
        const size_t heads = width - 2;
        std::vector<float> b(2 * width, 1.f);
        std::fill(b.begin() + width, b.end(), 2.f);
        std::vector<float> a(heads * 2 * width, 1.f);
        auto out = run_on_cpu(model, {{"b", make_f32_tensor({2, width}, b)},
                                      {"a", make_f32_tensor({1, heads, 2, width}, a)}});
        EXPECT_EQ(out.get_shape(), (ov::Shape{1, heads, 2, 2}));
        std::vector<float> expected;
        for (size_t row = 0; row < heads * 2; ++row) {
            expected.push_back(static_cast<float>(width));
            expected.push_back(static_cast<float>(2 * width));
        }
        expect_near(out, expected, 0.f);
    }
}

TEST(GGUFOps, MulMatGroupedHeadsPreserveDynamicRowsAndReduction) {
    for (bool dynamic_heads : {false, true}) {
        auto model = SingleOpBuilder()
                         .op("GGML_OP_MUL_MAT")
                         .input("b", ov::element::f32, {1, dynamic_heads ? -1 : 2, 1, -1})
                         .input("a", ov::element::f32, {1, dynamic_heads ? -1 : 4, -1, -1})
                         .output("out", ov::element::f32, ov::PartialShape::dynamic())
                         .build();
        for (size_t tokens : {size_t{1}, size_t{3}}) {
            auto out = run_on_cpu(model, {{"b", make_f32_tensor({1, 2, 1, 2}, {1, 1, 2, 2})},
                                          {"a", make_f32_tensor({1, 4, tokens, 2},
                                                                std::vector<float>(4 * tokens * 2, 1.f))}});
            EXPECT_EQ(out.get_shape(), (ov::Shape{1, 4, tokens, 1}));
            std::vector<float> expected(4 * tokens, 2.f);
            std::fill(expected.begin() + 2 * tokens, expected.end(), 4.f);
            expect_near(out, expected, 0.f);
        }
    }
}

TEST(GGUFOps, FlashAttnExtUsesResolvedViewShape) {
    class FlatStorageDecoder : public SingleOpDecoder {
    public:
        using SingleOpDecoder::SingleOpDecoder;
        ov::PartialShape get_input_shape(const std::string& name) const override {
            return name == "k" || name == "v" ? ov::PartialShape{1, 1, 1, 32} : SingleOpDecoder::get_input_shape(name);
        }
    };
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<FlatStorageDecoder>(
        "GGML_OP_FLASH_ATTN_EXT",
        std::vector<TensorDesc>{{"q", ov::element::f32, {1, 2, 1, 2}},
                                {"k", ov::element::f32, {1, 2, 2, 2}},
                                {"v", ov::element::f32, {1, 2, 2, 2}}},
        std::vector<TensorDesc>{},
        TensorDesc{"out", ov::element::f32, {1, 1, 2, 2}},
        std::map<std::string, ov::Any>{{"scale", 1.0f}});
    FrontEnd fe;
    auto model = fe.convert(fe.load(decoder));
    expect_near(run_on_cpu(model,
                           {{"q", make_f32_tensor({1, 2, 1, 2}, {0, 0, 0, 0})},
                            {"k", make_f32_tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8})},
                            {"v", make_f32_tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8})}}),
                {2, 3, 6, 7});
}

TEST(GGUFOps, FlashAttnExtRejectsMaxBiasWithoutMask) {
    EXPECT_THROW(SingleOpBuilder()
                     .op("GGML_OP_FLASH_ATTN_EXT")
                     .input("q", ov::element::f32, {1, 1, 1, 2})
                     .input("k", ov::element::f32, {1, 1, 1, 2})
                     .input("v", ov::element::f32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 1, 1, 2})
                     .attr("scale", 1.0f)
                     .attr("max_bias", 8.0f)
                     .attr("has_mask", false)
                     .build(),
                 std::exception);
}

// ALiBi: ggml scales the additive mask by a per-head slope. Two heads, max_bias 8: n_head_log2 = 2,
// m0 = 2^(-8/2) = 1/16, slopes m0^1 and m0^2. With zero queries the scores are slope * mask, so the
// attention weights expose the slope directly.
TEST(GGUFOps, FlashAttnExtAlibiScalesMaskPerHead) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_FLASH_ATTN_EXT")
                     .input("q", ov::element::f32, {1, 2, 1, 2})
                     .input("k", ov::element::f32, {1, 2, 2, 2})
                     .input("v", ov::element::f32, {1, 2, 2, 2})
                     .input("mask", ov::element::f32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 1, 2, 2})
                     .attr("scale", 1.0f)
                     .attr("max_bias", 8.0f)
                     .attr("has_mask", true)
                     .build();
    const std::vector<float> v{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> mask{0.0f, -16.0f};
    auto out = run_on_cpu(model,
                          {{"q", make_f32_tensor({1, 2, 1, 2}, {0, 0, 0, 0})},
                           {"k", make_f32_tensor({1, 2, 2, 2}, {0, 0, 0, 0, 0, 0, 0, 0})},
                           {"v", make_f32_tensor({1, 2, 2, 2}, v)},
                           {"mask", make_f32_tensor({1, 1, 1, 2}, mask)}});
    std::vector<float> expected;  // [B, L, H, D]
    for (int h = 0; h < 2; ++h) {
        const float slope = std::pow(1.0f / 16.0f, static_cast<float>(h + 1));
        const float w0 = 1.0f / (1.0f + std::exp(slope * mask[1]));
        const float w1 = 1.0f - w0;
        for (int d = 0; d < 2; ++d)
            expected.push_back(w0 * v[h * 4 + d] + w1 * v[h * 4 + 2 + d]);
    }
    // SDPA runs in f16.
    expect_near(out, expected, 2e-3f, 2e-3f);
}

TEST(GGUFOps, CpyDoesNotReshapeToCapturedOutputMetadata) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CPY")
                     .input("x", ov::element::i32, {2, 4})
                     .output("out", ov::element::f32, {1, 8})
                     .build();
    ov::Tensor input(ov::element::i32, {2, 4});
    std::iota(input.data<int32_t>(), input.data<int32_t>() + input.get_size(), 1);
    auto out = run_on_cpu(model, {{"x", input}});
    EXPECT_EQ(out.get_shape(), (ov::Shape{2, 4}));
    expect_near(out, {1, 2, 3, 4, 5, 6, 7, 8}, 0.0f);
}

TEST(GGUFOps, ContCaseZeroIdentityAndReshapeReuse) {
    auto static_model = SingleOpBuilder()
                            .op("GGML_OP_CONT")
                            .input("x", ov::element::f32, {1, 1, 2, 6})
                            .output("out", ov::element::f32, {1, 2, 3, 2})
                            .attr("cont_reshape", std::vector<int64_t>{1, 2, 3, 2})
                            .build();
    std::vector<float> initial(12);
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = static_cast<float>(i);
    const auto static_out = run_on_cpu(static_model, {{"x", make_f32_tensor({1, 1, 2, 6}, initial)}});
    ASSERT_EQ(static_out.get_shape(), (ov::Shape{1, 2, 3, 2}));
    expect_near(static_out, initial, 0.0f, 0.0f);

    for (bool reshape : {false, true}) {
        SCOPED_TRACE(reshape);
        auto builder = SingleOpBuilder()
                           .op("GGML_OP_CONT")
                           .input("x", ov::element::f32, {1, 1, -1, 6})
                           .output("out",
                                   ov::element::f32,
                                   reshape ? ov::PartialShape{1, 2, 3, 2} : ov::PartialShape{1, 1, 2, 6});
        if (reshape) {
            builder.attr("cont_reshape", std::vector<int64_t>{1, -1, 3, 2});
        }
        auto model = build_captured_op(builder, {{"x", {1, 1, 2, 6}}});
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        for (size_t tokens : {2u, 5u, 1u}) {
            SCOPED_TRACE(tokens);
            std::vector<float> values(tokens * 6);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(100 * tokens + i);
            request.set_tensor("x", make_f32_tensor({1, 1, tokens, 6}, values));
            request.infer();
            const auto out = request.get_output_tensor();
            ASSERT_EQ(out.get_shape(), (reshape ? ov::Shape{1, tokens, 3, 2} : ov::Shape{1, 1, tokens, 6}));
            expect_near(out, values, 0.0f, 0.0f);
        }
    }
}

// A negative slice length anchors the end of the axis: [0, -1) keeps every token row but the last one,
// whatever the runtime token count is (a GATED_DELTA_NET attention view ahead of its state rows).
TEST(GGUFOps, ViewSliceEndAnchoredKeepsDynamicAxis) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_VIEW")
                       .input("x", ov::element::f32, {1, 1, ov::Dimension(1, -1), 3})
                       .output("out", ov::element::f32, {1, 1, 2, 3})
                       .attr<std::vector<int64_t>>("view_slice", {2, 0, -1});
    auto model = build_captured_op(builder, {{"x", {1, 1, 3, 3}}});
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t tokens : {3u, 6u, 2u}) {
        SCOPED_TRACE(tokens);
        std::vector<float> values(tokens * 3);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(10 * tokens + i);
        request.set_tensor("x", make_f32_tensor({1, 1, tokens, 3}, values));
        request.infer();
        ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 1, tokens - 1, 3}));
        expect_near(request.get_output_tensor(),
                    std::vector<float>(values.begin(), values.end() - 3),
                    0.0f,
                    0.0f);
    }
}

TEST(GGUFOps, PermuteResolvedCasesReuseLogicalInput) {
    {
        auto builder = SingleOpBuilder()
                           .op("GGML_OP_PERMUTE")
                           .input("x", ov::element::f32, {1, -1, 3, 2})
                           .output("out", ov::element::f32, {1, 3, 2, 2})
                           .attr("perm", std::vector<int64_t>{0, 2, 1, 3})
                           .attr("input_ggml_shape", ov::Shape{1, 2, 3, 2})
                           .attr("view_seq_offset", int64_t{7});
        auto model = build_captured_op(builder, {{"x", {1, 2, 3, 2}}}, 7);
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        for (size_t tokens : {2u, 5u, 1u}) {
            SCOPED_TRACE(tokens);
            std::vector<float> values(tokens * 6), expected;
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(100 * tokens + i);
            for (size_t h = 0; h < 3; ++h)
                for (size_t t = 0; t < tokens; ++t)
                    for (size_t d = 0; d < 2; ++d)
                        expected.push_back(values[(t * 3 + h) * 2 + d]);
            request.set_tensor("x", make_f32_tensor({1, tokens, 3, 2}, values));
            request.infer();
            ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 3, tokens, 2}));
            expect_near(request.get_output_tensor(), expected, 0.0f, 0.0f);
        }
    }
}

TEST(GGUFOps, PermuteCaseZeroUsesLiveTokenExtent) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_PERMUTE")
                     .input("x", ov::element::f32, {1, 1, 3, 2})
                     .output("out", ov::element::f32, {1, 3, 1, 2})
                     .attr("input_ggml_shape", ov::Shape{1, 2, 3, 2})
                     .attr("perm", std::vector<int64_t>{0, 2, 1, 3})
                     .build();
    const std::vector<float> values{1, 2, 3, 4, 5, 6};
    const auto result = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 3, 2}, values)}});
    ASSERT_EQ(result.get_shape(), (ov::Shape{1, 3, 1, 2}));
    expect_near(result, values, 0.0f, 0.0f);
}

TEST(GGUFOps, GetRowsResolvedIndicesIgnoreViewOffsetOnReuse) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_GET_ROWS")
                       .input("data", ov::element::f32, {1, 1, 7, 3})
                       .input("indices", ov::element::i32, {1, 1, 1, -1})
                       .output("out", ov::element::f32, {1, 1, 2, 3});
    auto model = build_captured_op(builder, {{"indices", {1, 1, 1, 2}}}, 3);
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    std::vector<float> table(21);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = static_cast<float>(i);
    request.set_tensor("data", make_f32_tensor({1, 1, 7, 3}, table));
    const std::vector<int32_t> indices{6, 1, 4, 0, 3};
    for (size_t tokens : {2u, 5u, 1u}) {
        SCOPED_TRACE(tokens);
        ov::Tensor ids(ov::element::i32, {1, 1, 1, tokens});
        std::copy_n(indices.begin(), tokens, ids.data<int32_t>());
        request.set_tensor("indices", ids);
        request.infer();
        ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, 1, tokens, 3}));
        std::vector<float> expected;
        for (size_t t = 0; t < tokens; ++t)
            for (size_t d = 0; d < 3; ++d)
                expected.push_back(table[indices[t] * 3 + d]);
        expect_near(request.get_output_tensor(), expected, 0.0f, 0.0f);
    }
}

TEST(GGUFOps, AddMixedPrecisionAccumulatesBeforeOutputRounding) {
    const std::vector<float> lhs{1.0f, 1024.0f, -1024.0f, -1.0f};
    const std::vector<float> rhs{-0.9999f, -1023.75f, 1023.75f, 0.9999f};
    for (auto output_type : {ov::element::f16, ov::element::f32}) {
        SCOPED_TRACE(output_type);
        auto model = SingleOpBuilder()
                         .op("GGML_OP_ADD")
                         .input("a", ov::element::f16, {1, 1, -1, 4})
                         .input("b", ov::element::f32, {1, 1, -1, 4})
                         .output("out", output_type, {1, 1, -1, 4})
                         .build();
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        for (size_t tokens : {2u, 5u, 1u}) {
            SCOPED_TRACE(tokens);
            ov::Tensor a(ov::element::f16, {1, 1, tokens, 4});
            std::vector<float> b(tokens * 4), expected(tokens * 4);
            for (size_t i = 0; i < b.size(); ++i) {
                a.data<ov::float16>()[i] = ov::float16(lhs[i % 4]);
                b[i] = rhs[i % 4];
                const float sum = static_cast<float>(a.data<ov::float16>()[i]) + b[i];
                expected[i] = output_type == ov::element::f16 ? static_cast<float>(ov::float16(sum)) : sum;
            }
            request.set_tensor("a", a);
            request.set_tensor("b", make_f32_tensor({1, 1, tokens, 4}, b));
            request.infer();
            const auto actual = request.get_output_tensor();
            ASSERT_EQ(actual.get_element_type(), output_type);
            ASSERT_EQ(actual.get_shape(), (ov::Shape{1, 1, tokens, 4}));
            for (size_t i = 0; i < expected.size(); ++i) {
                const float value = output_type == ov::element::f16
                                        ? static_cast<float>(actual.data<ov::float16>()[i])
                                        : actual.data<float>()[i];
                EXPECT_NEAR(value, expected[i], 1e-7f) << "index " << i;
            }
        }
    }
}

TEST(GGUFOps, SetRowsViewPermutePreservesFunctionalCacheAcrossCalls) {
    class CacheGraphDecoder : public SingleOpDecoder {
    public:
        CacheGraphDecoder(const std::shared_ptr<GgufDecoder>& inputs,
                          std::vector<std::shared_ptr<GgufDecoder>> nodes)
            : SingleOpDecoder(*std::dynamic_pointer_cast<SingleOpDecoder>(inputs)),
              m_nodes(std::move(nodes)) {}

        void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> visitor) const override {
            for (const auto& node : m_nodes)
                visitor(node);
        }

        std::vector<std::string> get_model_output_names() const override {
            return {"cache", "new_cache", "attention_cache", "saved_cache"};
        }

    private:
        std::vector<std::shared_ptr<GgufDecoder>> m_nodes;
    };

    constexpr size_t capacity = 8, heads = 2, width = 2, row = heads * width;
    for (bool reshaped_destination : {false, true}) {
        SCOPED_TRACE(reshaped_destination);
        const ov::Shape root_shape{1, 1, capacity, row};
        const ov::Shape destination_shape = reshaped_destination ? ov::Shape{1, capacity, heads, width} : root_shape;
        auto inputs = SingleOpBuilder()
                          .op("GGML_OP_SET_ROWS")
                          .input("updates", ov::element::f32, {1, 1, -1, row})
                          .input("indices", ov::element::i64, {1, 1, 1, -1})
                          .input("cache", ov::element::f32, root_shape)
                          .output("new_cache", ov::element::f32, root_shape)
                          .decoder();
        std::vector<std::shared_ptr<GgufDecoder>> nodes;
        nodes.push_back(SingleOpBuilder()
                            .op("GGML_OP_CONT")
                            .input("cache", ov::element::f32, root_shape)
                            .output("saved_cache", ov::element::f32, root_shape)
                            .decoder());
        if (reshaped_destination) {
            nodes.push_back(SingleOpBuilder()
                                .op("GGML_OP_RESHAPE")
                                .input("cache", ov::element::f32, root_shape)
                                .output("destination", ov::element::f32, destination_shape)
                                .attr("reshape_target", std::vector<int64_t>{1, capacity, heads, width})
                                .decoder());
        }
        nodes.push_back(SingleOpBuilder()
                            .op("GGML_OP_SET_ROWS")
                            .input("updates", ov::element::f32, {1, 1, -1, row})
                            .input("indices", ov::element::i64, {1, 1, 1, -1})
                            .input(reshaped_destination ? "destination" : "cache", ov::element::f32, destination_shape)
                            .output("new_cache", ov::element::f32, destination_shape)
                            .decoder());
        // Read the root's original name after writing either it or its full-storage reshape alias.
        nodes.push_back(SingleOpBuilder()
                            .op("GGML_OP_VIEW")
                            .input("cache", ov::element::f32, root_shape)
                            .output("logical_cache", ov::element::f32, {1, capacity, heads, width})
                            .attr("view_input_strides", std::vector<int64_t>{4, 16, 128, 128})
                            .attr("view_output_strides", std::vector<int64_t>{4, 8, 16, 128})
                            .attr("view_offset_bytes", int64_t{0})
                            .decoder());
        nodes.push_back(SingleOpBuilder()
                            .op("GGML_OP_PERMUTE")
                            .input("logical_cache", ov::element::f32, {1, capacity, heads, width})
                            .output("attention_cache", ov::element::f32, {1, heads, capacity, width})
                            .attr("perm", std::vector<int64_t>{0, 2, 1, 3})
                            .attr("input_ggml_shape", ov::Shape{1, capacity, heads, width})
                            .decoder());
        std::shared_ptr<GgufDecoder> decoder = std::make_shared<CacheGraphDecoder>(inputs, std::move(nodes));
        FrontEnd fe;
        auto model = fe.convert(fe.load(decoder));
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        std::vector<float> expected_cache(capacity * row, -1.0f);
        auto cache = make_f32_tensor(root_shape, expected_cache);
        size_t position = 0;
        for (size_t tokens : {2u, 5u, 1u}) {
            SCOPED_TRACE(tokens);
            const auto saved_cache = expected_cache;
            const size_t rows_per_token = reshaped_destination ? heads : 1;
            std::vector<float> updates(tokens * row);
            ov::Tensor indices(ov::element::i64, {1, 1, 1, tokens * rows_per_token});
            for (size_t t = 0; t < tokens; ++t) {
                for (size_t h = 0; h < rows_per_token; ++h)
                    indices.data<int64_t>()[t * rows_per_token + h] =
                        static_cast<int64_t>((position + t) * rows_per_token + h);
                for (size_t d = 0; d < row; ++d) {
                    updates[t * row + d] = static_cast<float>((position + t + 1) * 10 + d);
                    expected_cache[(position + t) * row + d] = updates[t * row + d];
                }
            }
            request.set_tensor("updates", make_f32_tensor({1, 1, tokens, row}, updates));
            request.set_tensor("indices", indices);
            request.set_tensor(compiled.input("cache"), cache);
            request.infer();
            ASSERT_EQ(request.get_output_tensor(0).get_shape(), root_shape);
            expect_near(request.get_output_tensor(0), expected_cache, 0.0f, 0.0f);
            ASSERT_EQ(request.get_output_tensor(1).get_shape(), destination_shape);
            expect_near(request.get_output_tensor(1), expected_cache, 0.0f, 0.0f);
            std::vector<float> expected_attention;
            for (size_t h = 0; h < heads; ++h)
                for (size_t t = 0; t < capacity; ++t)
                    for (size_t d = 0; d < width; ++d)
                        expected_attention.push_back(expected_cache[t * row + h * width + d]);
            ASSERT_EQ(request.get_output_tensor(2).get_shape(), (ov::Shape{1, heads, capacity, width}));
            expect_near(request.get_output_tensor(2), expected_attention, 0.0f, 0.0f);
            ASSERT_EQ(request.get_output_tensor(3).get_shape(), root_shape);
            expect_near(request.get_output_tensor(3), saved_cache, 0.0f, 0.0f);
            request.get_output_tensor(0).copy_to(cache);
            position += tokens;
        }
    }
}

TEST(GGUFOps, StatefulSetRowsRebindsPrewriteReshapeAndPreservesSnapshot) {
    class StatefulAliasDecoder : public SingleOpDecoder {
    public:
        StatefulAliasDecoder(const std::shared_ptr<GgufDecoder>& inputs,
                             std::vector<std::shared_ptr<GgufDecoder>> nodes)
            : SingleOpDecoder(*std::dynamic_pointer_cast<SingleOpDecoder>(inputs)),
              m_nodes(std::move(nodes)) {}

        void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> visitor) const override {
            for (const auto& node : m_nodes)
                visitor(node);
        }

        std::vector<std::string> get_model_output_names() const override {
            return {"alias_read", "saved_cache"};
        }

    private:
        std::vector<std::shared_ptr<GgufDecoder>> m_nodes;
    };

    const ov::PartialShape cache_shape{1, -1, 2, 2};
    const ov::PartialShape alias_shape{1, 1, -1, 4};
    auto write = SingleOpBuilder()
                     .op("GGML_OP_SET_ROWS")
                     .input("updates", ov::element::f32, cache_shape)
                     .input("indices", ov::element::i64, {1, 1, 1, -1})
                     .input("cache", ov::element::f32, cache_shape)
                     .output("new_cache", ov::element::f32, cache_shape)
                     .decoder();
    auto snapshot = SingleOpBuilder()
                        .op("GGML_OP_CONT")
                        .input("cache", ov::element::f32, cache_shape)
                        .output("saved_cache", ov::element::f32, cache_shape)
                        .decoder();
    auto alias = SingleOpBuilder()
                     .op("GGML_OP_RESHAPE")
                     .input("cache", ov::element::f32, cache_shape)
                     .output("cache_alias", ov::element::f32, alias_shape)
                     .attr("reshape_target", std::vector<int64_t>{1, 1, -1, 4})
                     .decoder();
    auto read = SingleOpBuilder()
                    .op("GGML_OP_CONT")
                    .input("cache_alias", ov::element::f32, alias_shape)
                    .output("alias_read", ov::element::f32, alias_shape)
                    .decoder();
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<StatefulAliasDecoder>(
        write, std::vector<std::shared_ptr<GgufDecoder>>{snapshot, alias, write, read});
    FrontEnd fe;
    fe.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
        ov::frontend::gguf::pass::GGUFMakeStateful()));
    auto model = fe.convert(fe.load(decoder));
    for (const auto& parameter : model->get_parameters())
        EXPECT_NE(parameter->get_friendly_name(), "cache");
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    ov::Tensor beam(ov::element::i32, {1});
    beam.data<int32_t>()[0] = 0;
    request.set_tensor("beam_idx", beam);
    std::vector<float> expected_cache;
    size_t past = 0;
    for (size_t tokens : {2u, 5u, 1u}) {
        SCOPED_TRACE(tokens);
        const auto expected_snapshot = expected_cache;
        std::vector<float> updates(tokens * 4);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t d = 0; d < 4; ++d)
                updates[t * 4 + d] = static_cast<float>((past + t + 1) * 10 + d);
        expected_cache.insert(expected_cache.end(), updates.begin(), updates.end());
        request.set_tensor("updates", make_f32_tensor({1, tokens, 2, 2}, updates));
        for (const auto& input : compiled.inputs()) {
            if (input.get_names().count("indices")) {
                ov::Tensor indices(ov::element::i64, {1, 1, 1, tokens * 2});
                for (size_t i = 0; i < tokens * 2; ++i)
                    indices.data<int64_t>()[i] = static_cast<int64_t>(past * 2 + i);
                request.set_tensor(input, indices);
            }
        }
        request.infer();
        ASSERT_EQ(request.get_output_tensor(0).get_shape(), (ov::Shape{1, 1, past + tokens, 4}));
        expect_near(request.get_output_tensor(0), expected_cache, 0.0f, 0.0f);
        ASSERT_EQ(request.get_output_tensor(1).get_shape(), (ov::Shape{1, past, 2, 2}));
        expect_near(request.get_output_tensor(1), expected_snapshot, 0.0f, 0.0f);
        const auto states = request.query_state();
        ASSERT_EQ(states.size(), 1u);
        EXPECT_EQ(states[0].get_name(), "cache");
        const auto state = states[0].get_state();
        ASSERT_EQ(state.get_shape(), (ov::Shape{1, past + tokens, 2, 2}));
        expect_near(state, expected_cache, 0.0f, 0.0f);
        past += tokens;
    }
}

// SwiGLU with clamped halves (ggml_compute_forward_swiglu_clamp): silu(min(g, limit)) * clamp(u, -limit, limit).
TEST(GGUFOps, SwigluClamp) {
    const float limit = 2.0f;
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_SWIGLU_CLAMP")
                     .input("x", ov::element::f32, {2, 8})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<bool>("swapped", false)
                     .attr<float>("glu_limit", limit)
                     .build();
    std::vector<float> x{-3, -1, 1, 3, -5, -1, 0.5f, 5, 4, 0.25f, -0.5f, 2.5f, 1, -3, 3, -1};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 8}, x)}});
    std::vector<float> expected;
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 4; ++c) {
            const float g = std::min(x[r * 8 + c], limit);
            const float u = std::clamp(x[r * 8 + 4 + c], -limit, limit);
            expected.push_back(g / (1.0f + std::exp(-g)) * u);
        }
    }
    expect_near(out, expected);
}

// GEGLU_QUICK: gelu_quick(g) * u with gelu_quick(x) = x * sigmoid(1.702 x) (ggml_gelu_quick_f32).
TEST(GGUFOps, GegluQuick) {
    auto model = SingleOpBuilder()
                     .op("GGML_GLU_OP_GEGLU_QUICK")
                     .input("x", ov::element::f32, {2, 6})
                     .output("out", ov::element::f32, {2, 3})
                     .attr<bool>("swapped", true)
                     .build();
    std::vector<float> x{1, -2, 0.5f, 3, -1, 2, -0.5f, 4, 1.5f, 2, 0.25f, -3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 6}, x)}});
    std::vector<float> expected;
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 3; ++c) {
            const float g = x[r * 6 + 3 + c];  // swapped: the gate is the second half
            const float u = x[r * 6 + c];
            expected.push_back(g / (1.0f + std::exp(-1.702f * g)) * u);
        }
    }
    expect_near(out, expected);
}

// POOL_2D over ggml [W, H, C, N] = OV NCHW, params {op, k0, k1, s0, s1, p0, p1}: a 2-wide, 1-high
// window stepping 2 along W.
TEST(GGUFOps, Pool2dMaxAndAvg) {
    const std::vector<float> x{1, 5, -2, 3, 4, 0, 7, -1};
    for (int32_t mode : {0, 1}) {
        SCOPED_TRACE(mode);
        auto model = SingleOpBuilder()
                         .op("GGML_OP_POOL_2D")
                         .input("x", ov::element::f32, {1, 1, 2, 4})
                         .output("out", ov::element::f32, {1, 1, 2, 2})
                         .attr("pool_params", std::vector<int32_t>{mode, 2, 1, 2, 1, 0, 0})
                         .build();
        auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 4}, x)}});
        if (mode == 0) {
            expect_near(out, {5, 3, 4, 7});
        } else {
            expect_near(out, {3, 0.5f, 2, 3});
        }
    }
}

// ROLL: dst[i] = src[(i - shift) mod n] per axis; shifts are in ggml order (dim 0 innermost).
TEST(GGUFOps, Roll) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_ROLL")
                     .input("x", ov::element::f32, {1, 1, 2, 3})
                     .output("out", ov::element::f32, {1, 1, 2, 3})
                     .attr("roll_shifts", std::vector<int32_t>{1, 1, 0, 0})
                     .build();
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 2, 3}, {0, 1, 2, 10, 11, 12})}});
    // Columns shift by one ([0,1,2] -> [2,0,1]), then the two rows swap.
    expect_near(out, {12, 10, 11, 2, 0, 1});
}

// SOLVE_TRI: A X = B with A lower triangular (the only variant ggml accepts), two batches.
TEST(GGUFOps, SolveTri) {
    const size_t n = 3, k = 2, batches = 2;
    std::vector<float> a{2, 0, 0, 1, 4, 0, -1, 3, 5,   // batch 0
                         1, 0, 0, 2, 1, 0, 3, -2, 4};  // batch 1
    std::vector<float> b{2, 4, 9, 6, 1, 18, 1, -1, 4, 0, 9, 5};
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SOLVE_TRI")
                     .input("a", ov::element::f32, {1, batches, n, n})
                     .input("b", ov::element::f32, {1, batches, n, k})
                     .output("out", ov::element::f32, {1, batches, n, k})
                     .build();
    auto out = run_on_cpu(model,
                          {{"a", make_f32_tensor({1, batches, n, n}, a)}, {"b", make_f32_tensor({1, batches, n, k}, b)}});
    std::vector<float> expected(batches * n * k);
    for (size_t bt = 0; bt < batches; ++bt) {
        for (size_t c = 0; c < k; ++c) {
            for (size_t i = 0; i < n; ++i) {
                float sum = b[(bt * n + i) * k + c];
                for (size_t j = 0; j < i; ++j)
                    sum -= a[(bt * n + i) * n + j] * expected[(bt * n + j) * k + c];
                expected[(bt * n + i) * k + c] = sum / a[(bt * n + i) * n + i];
            }
        }
    }
    expect_near(out, expected);
}

// A CPY that rewrites a range of its destination updates that storage in place: a later read of the
// destination, and the model output written back to it, observe the new value.
TEST(GGUFOps, CpyUpdateRebindsDestinationStorage) {
    class GraphDecoder : public SingleOpDecoder {
    public:
        GraphDecoder(const std::shared_ptr<GgufDecoder>& inputs, std::vector<std::shared_ptr<GgufDecoder>> nodes)
            : SingleOpDecoder(*std::dynamic_pointer_cast<SingleOpDecoder>(inputs)),
              m_nodes(std::move(nodes)) {}
        void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> visitor) const override {
            for (const auto& node : m_nodes)
                visitor(node);
        }
        std::vector<std::string> get_model_output_names() const override {
            return {"buffer", "read"};
        }

    private:
        std::vector<std::shared_ptr<GgufDecoder>> m_nodes;
    };

    const ov::Shape buffer_shape{1, 1, 1, 8};
    auto write = [&] {
        return SingleOpBuilder()
            .op("GGML_OP_CPY")
            .input("value", ov::element::f32, {1, 1, 1, -1})
            .input("buffer", ov::element::f32, buffer_shape)
            .input("begin", ov::element::i64, {1})
            .output("written", ov::element::f32, buffer_shape)
            .attr("dst_type", ov::element::f32)
            .attr("axis", int64_t{3})
            .attr("reshape_target", std::vector<int64_t>{1, 1, 1, -1});
    };
    std::vector<std::shared_ptr<GgufDecoder>> nodes{write().decoder(),
                                                    SingleOpBuilder()
                                                        .op("GGML_OP_SCALE")
                                                        .input("buffer", ov::element::f32, buffer_shape)
                                                        .output("read", ov::element::f32, buffer_shape)
                                                        .attr("scale", 2.0f)
                                                        .decoder()};
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<GraphDecoder>(write().decoder(), std::move(nodes));
    FrontEnd fe;
    auto model = fe.convert(fe.load(decoder));
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    ov::Tensor begin(ov::element::i64, {1});
    begin.data<int64_t>()[0] = 5;
    request.set_tensor("value", make_f32_tensor({1, 1, 1, 2}, {7, 9}));
    request.set_tensor("buffer", make_f32_tensor(buffer_shape, {0, 1, 2, 3, 4, 5, 6, 7}));
    request.set_tensor("begin", begin);
    request.infer();
    const std::vector<float> updated{0, 1, 2, 3, 4, 7, 9, 7};
    // Outputs follow get_model_output_names(): the written-back buffer, then the later read.
    expect_near(request.get_output_tensor(0), updated, 0.0f, 0.0f);
    std::vector<float> doubled;
    for (float value : updated)
        doubled.push_back(2 * value);
    expect_near(request.get_output_tensor(1), doubled, 0.0f, 0.0f);
}

class GGUFRopeDescriptor : public ::testing::TestWithParam<int> {};

TEST_P(GGUFRopeDescriptor, ReusesDynamicPrefillAndDecode) {
    const int mode = GetParam();
    const bool imrope = mode == 2;
    const int64_t heads = 2, head_dim = 8, n_rot = 6;
    RopeConfig cfg;
    cfg.n_dims = n_rot;
    cfg.freq_base = 10000.0f;
    cfg.freq_scale = 1.0f;
    cfg.ext_factor = 0.0f;
    cfg.attn_factor = 1.0f;
    cfg.n_ctx_orig = 4096;
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_ROPE")
                       .input("data", ov::element::f32, {1, -1, heads, head_dim})
                       .input("pos", ov::element::i32, {1, 1, 1, -1})
                       .output("out", ov::element::f32, {1, -1, heads, head_dim})
                       .attr("rope_config", cfg)
                       .attr("rope_mode", std::string(mode == 0 ? "normal" : mode == 1 ? "neox" : "imrope"));
    ov::Core core;
    auto compiled = core.compile_model(builder.build(), "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t tokens : {3u, 1u}) {
        SCOPED_TRACE(tokens);
        std::vector<float> data(tokens * heads * head_dim);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = 0.13f * static_cast<float>(i + 1);
        }
        ov::Tensor positions(ov::element::i32, {1, 1, 1, tokens * (imrope ? 4 : 1)});
        for (size_t plane = 0; plane < (imrope ? 4u : 1u); ++plane) {
            for (size_t t = 0; t < tokens; ++t) {
                positions.data<int32_t>()[plane * tokens + t] = static_cast<int32_t>(5 + t);
            }
        }
        request.set_tensor("data", make_f32_tensor({1, tokens, heads, head_dim}, data));
        request.set_tensor("pos", positions);
        request.infer();
        ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, tokens, heads, head_dim}));
        std::vector<float> expected = data;
        for (size_t t = 0; t < tokens; ++t) {
            std::vector<float> cos_v, sin_v;
            neox_ref_cos_sin(n_rot, cfg.freq_base, cfg.freq_scale, cfg.attn_factor, 5 + t, cos_v, sin_v);
            for (int64_t h = 0; h < heads; ++h) {
                const auto base = (t * heads + h) * head_dim;
                for (int64_t j = 0; j < n_rot / 2; ++j) {
                    const auto first = base + (mode == 0 ? 2 * j : j);
                    const auto second = mode == 0 ? first + 1 : first + n_rot / 2;
                    expected[first] = data[first] * cos_v[j] - data[second] * sin_v[j];
                    expected[second] = data[first] * sin_v[j] + data[second] * cos_v[j];
                }
            }
        }
        expect_near(request.get_output_tensor(), expected);
    }
}

INSTANTIATE_TEST_SUITE_P(GGUFOps,
                         GGUFRopeDescriptor,
                         ::testing::Values(0, 1, 2));

class GGUFFlashDescriptor : public ::testing::TestWithParam<std::tuple<bool, float, bool>> {};

TEST_P(GGUFFlashDescriptor, ReusesDynamicPrefillAndDecode) {
    const bool has_mask = std::get<0>(GetParam());
    const float softcap = std::get<1>(GetParam());
    const bool has_sinks = std::get<2>(GetParam());
    const size_t heads = 4, kv_heads = 2, width = 3, value_width = 2;
    const float scale = 0.71f;
    auto shape = [&](size_t h, size_t d) {
        return ov::PartialShape{1, h, -1, d};
    };
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_FLASH_ATTN_EXT")
                       .input("q", ov::element::f32, shape(heads, width))
                       .input("k", ov::element::f32, shape(kv_heads, width))
                       .input("v", ov::element::f32, shape(kv_heads, value_width));
    if (has_mask) {
        builder.input("mask", ov::element::f32, {1, 1, 8, -1});
    }
    if (has_sinks) {
        builder.input("sinks", ov::element::f32, {heads});
    }
    builder.output("out", ov::element::f32, {1, -1, heads, value_width})
        .attr("scale", scale)
        .attr("kq_soft_cap", softcap)
        .attr("has_mask", has_mask);
    ov::Core core;
    auto compiled = core.compile_model(builder.build(), "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t tokens : {3u, 1u}) {
        SCOPED_TRACE(tokens);
        const size_t keys = tokens == 3 ? 5 : 7;
        auto index = [&](size_t h, size_t t, size_t d, size_t H, size_t T, size_t D) {
            return (h * T + t) * D + d;
        };
        auto tensor_shape = [&](size_t H, size_t T, size_t D) {
            return ov::Shape{1, H, T, D};
        };
        std::vector<float> q(heads * tokens * width), k(kv_heads * keys * width), v(kv_heads * keys * value_width);
        for (size_t i = 0; i < q.size(); ++i)
            q[i] = 0.137f * static_cast<float>(i % 17) - 0.81f;
        for (size_t i = 0; i < k.size(); ++i)
            k[i] = 0.173f * static_cast<float>(i % 13) - 0.92f;
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = 0.227f * static_cast<float>(i % 11) - 0.37f;
        std::vector<float> mask(8 * keys);
        for (size_t t = 0; t < 8; ++t)
            for (size_t s = 0; s < keys; ++s)
                mask[t * keys + s] = s > t + 2 ? -10000.0f : -0.17f * static_cast<float>(s);
        const std::vector<float> sinks{-0.3f, 0.4f, 0.2f, -0.1f};
        request.set_tensor("q", make_f32_tensor(tensor_shape(heads, tokens, width), q));
        request.set_tensor("k", make_f32_tensor(tensor_shape(kv_heads, keys, width), k));
        request.set_tensor("v", make_f32_tensor(tensor_shape(kv_heads, keys, value_width), v));
        if (has_mask)
            request.set_tensor("mask", make_f32_tensor({1, 1, 8, keys}, mask));
        if (has_sinks)
            request.set_tensor("sinks", make_f32_tensor({heads}, sinks));
        request.infer();
        ASSERT_EQ(request.get_output_tensor().get_shape(), (ov::Shape{1, tokens, heads, value_width}));
        std::vector<float> expected(tokens * heads * value_width);
        for (size_t t = 0; t < tokens; ++t) {
            for (size_t h = 0; h < heads; ++h) {
                const auto kh = h / (heads / kv_heads);
                std::vector<float> logits(keys + (has_sinks ? 1 : 0));
                for (size_t s = 0; s < keys; ++s) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < width; ++d)
                        dot += q[index(h, t, d, heads, tokens, width)] * k[index(kh, s, d, kv_heads, keys, width)];
                    const float scaled = scale * dot;
                    logits[s] = (softcap == 0.0f ? scaled : softcap * std::tanh(scaled / softcap)) +
                                (has_mask ? mask[t * keys + s] : 0.0f);
                }
                if (has_sinks)
                    logits[keys] = sinks[h];
                const auto maximum = *std::max_element(logits.begin(), logits.end());
                float sum = 0.0f;
                for (auto& logit : logits) {
                    logit = std::exp(logit - maximum);
                    sum += logit;
                }
                for (size_t d = 0; d < value_width; ++d) {
                    for (size_t s = 0; s < keys; ++s) {
                        expected[(t * heads + h) * value_width + d] +=
                            logits[s] / sum * v[index(kh, s, d, kv_heads, keys, value_width)];
                    }
                }
            }
        }
        expect_near(request.get_output_tensor(), expected, 1e-4f, OV_GGUF_TEST_DEFAULT_RTOL);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GGUFOps,
    GGUFFlashDescriptor,
    ::testing::Values(std::make_tuple(true, 0.0f, false),
                      std::make_tuple(false, 0.0f, false),
                      std::make_tuple(true, 0.8f, false),
                      std::make_tuple(false, 0.8f, false),
                      std::make_tuple(true, 0.8f, true),
                      std::make_tuple(false, 0.8f, true),
                      std::make_tuple(false, 0.0f, true)));

// Ops ported from the llama.cpp OpenVINO backend: described by ggml op parameters only.
TEST(GGUFOps, MeanAndSumReduce) {
    const auto x = make_f32_tensor({1, 1, 2, 3}, {1, 2, 3, 4, 5, 6});
    auto mean = SingleOpBuilder().op("GGML_OP_MEAN").input("x", ov::element::f32, {1, 1, 2, 3}).output("out", ov::element::f32, {1, 1, 2, 1}).build();
    expect_near(run_on_cpu(mean, {{"x", x}}), {2, 5});
    auto sum = SingleOpBuilder().op("GGML_OP_SUM").input("x", ov::element::f32, {1, 1, 2, 3}).output("out", ov::element::f32, {1, 1, 1, 1}).build();
    expect_near(run_on_cpu(sum, {{"x", x}}), {21});
}

TEST(GGUFOps, DupCopiesItsOperand) {
    auto model = SingleOpBuilder().op("GGML_OP_DUP").input("x", ov::element::f32, {1, 1, 1, 3}).output("out", ov::element::f32, {1, 1, 1, 3}).build();
    expect_near(run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 3}, {1, -2, 3})}}), {1, -2, 3});
}

TEST(GGUFOps, DupConvertsToTheOutputType) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_DUP")
                     .input("x", ov::element::f32, {1, 1, 1, 3})
                     .output("out", ov::element::f16, {1, 1, 1, 3})
                     .attr<ov::element::Type>("dst_type", ov::element::f16)
                     .build();
    // 0.1f is not representable in f16, so the values show the conversion happened.
    const std::vector<float> values{0.1f, -2.5f, 3.0f};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 3}, values)}});
    ASSERT_EQ(out.get_element_type(), ov::element::f16);
    ASSERT_EQ(out.get_size(), values.size());
    const auto* data = out.data<ov::float16>();
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ(static_cast<float>(data[i]), static_cast<float>(ov::float16(values[i]))) << "index " << i;
    }
}

TEST(GGUFOps, UnaryHardsigmoidExpm1StepRound) {
    const std::vector<float> v{-4.0f, -0.5f, 0.0f, 0.5f, 2.5f};
    const auto run = [&](const char* op) {
        auto model = SingleOpBuilder().op(op).input("x", ov::element::f32, {1, 1, 1, 5}).output("out", ov::element::f32, {1, 1, 1, 5}).build();
        return run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 5}, v)}});
    };
    std::vector<float> hs, em, st, rd;
    for (float x : v) {
        hs.push_back(std::min(1.0f, std::max(0.0f, x / 6.0f + 0.5f)));
        em.push_back(std::expm1(x));
        st.push_back(x > 0.0f ? 1.0f : 0.0f);
        rd.push_back(std::round(x));
    }
    expect_near(run("GGML_UNARY_OP_HARDSIGMOID"), hs);
    expect_near(run("GGML_UNARY_OP_EXPM1"), em);
    expect_near(run("GGML_UNARY_OP_STEP"), st);
    expect_near(run("GGML_UNARY_OP_ROUND"), rd);
}

TEST(GGUFOps, ConvDepthwiseAndTransposed) {
    // Depthwise 2x2 kernel per channel, stride 1, no padding: one output per channel.
    auto dw = SingleOpBuilder()
                  .op("GGML_OP_CONV_2D_DW")
                  .input("w", ov::element::f32, {2, 1, 2, 2})
                  .input("x", ov::element::f32, {1, 2, 2, 2})
                  .output("out", ov::element::f32, {1, 2, 1, 1})
                  .attr<std::vector<int64_t>>("conv_params", {1, 1, 0, 0, 1, 1})
                  .build();
    expect_near(run_on_cpu(dw, {{"w", make_f32_tensor({2, 1, 2, 2}, {1, 1, 1, 1, 1, 0, 0, 1})},
                                {"x", make_f32_tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8})}}),
                {10, 13});
    // 1D transposed conv, Cin = Cout = 1, K = 2, stride 2: each input sample emits the kernel scaled.
    auto t1 = SingleOpBuilder()
                  .op("GGML_OP_CONV_TRANSPOSE_1D")
                  .input("w", ov::element::f32, {1, 1, 1, 2})
                  .input("x", ov::element::f32, {1, 1, 1, 2})
                  .output("out", ov::element::f32, {1, 1, 1, 4})
                  .attr<std::vector<int64_t>>("conv_params", {2, 0, 1})
                  .build();
    expect_near(run_on_cpu(t1, {{"w", make_f32_tensor({1, 1, 1, 2}, {1, 10})}, {"x", make_f32_tensor({1, 1, 1, 2}, {1, 2})}}),
                {1, 10, 2, 20});
    // 2D transposed conv, 1x1 kernel, stride 2: inputs land on the even grid, zeros elsewhere.
    auto t2 = SingleOpBuilder()
                  .op("GGML_OP_CONV_TRANSPOSE_2D")
                  .input("w", ov::element::f32, {1, 1, 1, 1})
                  .input("x", ov::element::f32, {1, 1, 1, 2})
                  .output("out", ov::element::f32, {1, 1, 1, 3})
                  .attr<std::vector<int64_t>>("conv_params", {2})
                  .build();
    expect_near(run_on_cpu(t2, {{"w", make_f32_tensor({1, 1, 1, 1}, {3})}, {"x", make_f32_tensor({1, 1, 1, 2}, {1, 2})}}),
                {3, 0, 6});
}

TEST(GGUFOps, Conv3dSumsTheKernelWindow) {
    // c = n = oc = 1, a 2x2x2 all-ones kernel over a 2x2x2 input: the single output is the input sum.
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CONV_3D")
                     .input("w", ov::element::f32, {1, 2, 2, 2})
                     .input("x", ov::element::f32, {1, 2, 2, 2})
                     .output("out", ov::element::f32, {1, 1, 1, 1})
                     .attr<std::vector<int64_t>>("conv_params", {1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1})
                     .build();
    expect_near(run_on_cpu(model, {{"w", make_f32_tensor({1, 2, 2, 2}, std::vector<float>(8, 1.0f))},
                                   {"x", make_f32_tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8})}}),
                {36});
}

TEST(GGUFOps, Im2col3dSinglePatch) {
    // A 2x2x2 kernel over a 2x2x2 single-channel volume yields one patch holding every element.
    auto model = SingleOpBuilder()
                     .op("GGML_OP_IM2COL_3D")
                     .input("w", ov::element::f32, {1, 2, 2, 2})
                     .input("x", ov::element::f32, {1, 2, 2, 2})
                     .output("out", ov::element::f32, {1, 1, 1, 8})
                     .attr<std::vector<int32_t>>("im2col_params", {1, 1, 1, 0, 0, 0, 1, 1, 1, 1})
                     .build();
    const std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    // The kernel only supplies its shape, so its Parameter is not part of the converted model.
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({1, 2, 2, 2}, x)}});
    ASSERT_EQ(out.get_shape(), (ov::Shape{1, 1, 1, 8}));
    expect_near(out, x);
}

TEST(GGUFOps, UpscaleToFixedSizes) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_UPSCALE")
                     .input("x", ov::element::f32, {1, 1, 1, 2})
                     .output("out", ov::element::f32, {1, 1, 2, 4})
                     .attr<int>("interpolation_mode", 0)
                     .attr<std::vector<int64_t>>("upscale_sizes", {2, 4})
                     .build();
    expect_near(run_on_cpu(model, {{"x", make_f32_tensor({1, 1, 1, 2}, {1, 2})}}), {1, 1, 2, 2, 1, 1, 2, 2});
}
