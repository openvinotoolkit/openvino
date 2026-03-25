// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for DynamicQuantize decomposition accuracy.
//
// All models use fp16 input/output — this is the native precision of the
// KV-cache compression subgraphs we want to test.  The roundtrip graph is:
//   Parameter(f16) → Convert(f32) → DynamicQuantize → dequant chain → Convert(f16) → Result(f16)
//
// Input data is generated in f32 then rounded to fp16, so the source
// distribution is exactly fp16-representable.  This ensures that quantization
// error measurements are not polluted by f32→f16 conversion artifacts.
//
// Two test cases share the same parameterized fixture:
//   RoundtripAccuracy       — uses model->evaluate() (reference interpreter,
//                             always available, no device dependency)
//   DeviceRoundtripAccuracy — compiles on a real device via ov::Core and runs
//                             inference.  Enabled by setting the environment
//                             variable OV_DQ_TEST_DEVICE (e.g. "CPU", "NPU").
//                             Skips gracefully when the device is unavailable.
//
// The dequantize chain matches the actual product code in create_dequant_nodes():
//   deq = (convert(q, f32) - convert(zp, f32)) * scale_f32

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kv_cache_compressed.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/dynamic_quantize.hpp"

namespace {

using namespace ov;

// ============================================================================
// Accuracy metrics
// ============================================================================
struct AccuracyMetrics {
    double l2_norm;
    double rel_l2_norm;
    double max_abs_error;
    double mean_abs_error;
};

AccuracyMetrics compute_metrics(const float* original, const float* reconstructed, size_t count) {
    double sum_sq_err = 0.0;
    double sum_abs_err = 0.0;
    double max_abs = 0.0;
    double sum_sq_orig = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double diff = static_cast<double>(original[i]) - static_cast<double>(reconstructed[i]);
        sum_sq_err += diff * diff;
        double abs_diff = std::abs(diff);
        sum_abs_err += abs_diff;
        if (abs_diff > max_abs) max_abs = abs_diff;
        sum_sq_orig += static_cast<double>(original[i]) * static_cast<double>(original[i]);
    }

    AccuracyMetrics m;
    m.l2_norm = std::sqrt(sum_sq_err);
    m.rel_l2_norm = (sum_sq_orig > 0.0) ? m.l2_norm / std::sqrt(sum_sq_orig) : 0.0;
    m.max_abs_error = max_abs;
    m.mean_abs_error = (count > 0) ? sum_abs_err / static_cast<double>(count) : 0.0;
    return m;
}

// ============================================================================
// Build a roundtrip model using the actual production decomposition passes.
// The model always uses fp16 I/O to match the real KV-cache pipeline:
//   Parameter(f16) → Convert(f32) → [DQ + dequant in f32] → Convert(f16) → Result(f16)
//
// The DynamicQuantize node is named with "key" so the decomposition pass
// selects reduction_axis=3 (the key-cache embedding dimension).
// ============================================================================
std::shared_ptr<Model> build_roundtrip_model(const Shape& shape,
                                             int decompose_version) {
    // ── Step 1: build the complete quantize → dequant graph ──────────────

    auto input = std::make_shared<op::v0::Parameter>(element::f16, shape);
    input->set_friendly_name("input");

    // Convert fp16 input to f32 for the quantize core
    auto dq_input = std::make_shared<op::v0::Convert>(input, element::f32);
    dq_input->set_friendly_name("input_cvt_f32");

    op::internal::DynamicQuantize::Attributes config;
    config.quantization_type = op::internal::DynamicQuantize::QuantizationType::Asymmetric;
    config.scale_dt = element::f32;
    config.group_sizes = std::vector<uint64_t>(shape.size(), 1);

    // V2 uses u8, V1/V3 use i8
    if (decompose_version == 2) {
        config.quantization_dt = element::u8;
        config.zp_dt = element::u8;
    } else {
        config.quantization_dt = element::i8;
        config.zp_dt = element::i8;
    }

    auto dq = std::make_shared<op::internal::DynamicQuantize>(dq_input, config);
    // Name must contain "key" so the pass picks reduction_axis=3
    dq->set_friendly_name("DynamicQuantize/0/key");

    // Dequant chain: deq = (convert(q, f32) - convert(zp, f32)) * scale
    // This mirrors create_dequant_nodes() in production code.
    auto q_f32 = std::make_shared<op::v0::Convert>(dq->output(0), element::f32);
    auto zp_f32 = std::make_shared<op::v0::Convert>(dq->output(2), element::f32);
    auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32);
    auto reconstructed = std::make_shared<op::v1::Multiply>(sub_zp, dq->output(1));

    // Convert result back to f16
    auto output_cvt = std::make_shared<op::v0::Convert>(reconstructed, element::f16);
    output_cvt->set_friendly_name("output_cvt_f16");

    auto result = std::make_shared<op::v0::Result>(output_cvt);
    result->set_friendly_name("result_reconstructed");

    auto model = std::make_shared<Model>(
        ResultVector{result},
        ParameterVector{input},
        "roundtrip_fp16_v" + std::to_string(decompose_version));

    model->validate_nodes_and_infer_types();

    // ── Step 2: run the decompose pass ───────────────────────────────────
    // The pass pattern-matches DynamicQuantize and replaces it in-place.
    // The dequant chain (already connected to DQ outputs) transparently
    // picks up the decomposed quantize subgraph.

    pass::Manager manager("decompose_dq");
    switch (decompose_version) {
    case 1:
        manager.register_pass<ov::npuw::DecomposeDynamicQuantize>();
        break;
    case 2:
        manager.register_pass<ov::npuw::DecomposeDynamicQuantize2>();
        break;
    case 3:
        manager.register_pass<ov::npuw::DecomposeDynamicQuantize3>();
        break;
    default:
        OPENVINO_THROW("Unknown decompose version: ", decompose_version);
    }
    manager.run_passes(model);
    model->validate_nodes_and_infer_types();

    return model;
}

// ============================================================================
// Data generation helpers
// Generated values are rounded to fp16 so the source distribution is exactly
// representable — no f32→f16 conversion artifacts in error measurements.
// ============================================================================

// Round a float value through fp16 and back
static float to_fp16(float v) {
    return static_cast<float>(ov::float16(v));
}

std::vector<float> generate_uniform_data(size_t count, float min_val, float max_val, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> data(count);
    for (auto& v : data) {
        v = to_fp16(dist(gen));
    }
    return data;
}

std::vector<float> generate_gen_gaussian_data(size_t count, float mu, float alpha, float beta, unsigned seed) {
    std::mt19937 gen(seed);
    std::gamma_distribution<double> gamma_dist(1.0 / static_cast<double>(beta), 1.0);
    std::bernoulli_distribution sign_dist(0.5);

    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        double g = gamma_dist(gen);
        double abs_x = static_cast<double>(alpha) * std::pow(g, 1.0 / static_cast<double>(beta));
        double sign = sign_dist(gen) ? 1.0 : -1.0;
        data[i] = to_fp16(static_cast<float>(static_cast<double>(mu) + sign * abs_x));
    }
    return data;
}

// ============================================================================
// Helper: evaluate a roundtrip model via the reference interpreter.
// The model has fp16 I/O.  Input data is provided as f32 (already fp16-rounded),
// converted to fp16 for the model, output converted back to f32 for comparison.
// ============================================================================
AccuracyMetrics evaluate_roundtrip(const std::shared_ptr<Model>& model,
                                   const float* input_data,
                                   size_t element_count) {
    auto input_tensor = Tensor(element::f16, model->get_parameters()[0]->get_shape());
    auto* dst = input_tensor.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        dst[i] = ov::float16(input_data[i]);
    }

    auto output_tensor = Tensor(element::f16, model->get_results()[0]->get_shape());

    TensorVector inputs{input_tensor};
    TensorVector outputs{output_tensor};

    bool ok = model->evaluate(outputs, inputs);
    EXPECT_TRUE(ok) << "Model evaluation failed for " << model->get_friendly_name();

    // Convert fp16 output to f32 for metric computation
    std::vector<float> output_f32(element_count);
    const auto* src = outputs[0].data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        output_f32[i] = static_cast<float>(src[i]);
    }

    return compute_metrics(input_data, output_f32.data(), element_count);
}

// ============================================================================
// Helper: evaluate a roundtrip model on a real device via ov::Core.
// The model always has fp16 I/O.  Input data is provided as f32
// (already fp16-rounded), output is compared as f32.
// ============================================================================
AccuracyMetrics evaluate_roundtrip_on_device(const std::shared_ptr<Model>& model,
                                             const std::string& device_name,
                                             const float* input_data,
                                             size_t element_count) {
    Core core;
    ov::AnyMap device_config;

    auto compiled = core.compile_model(model, device_name, device_config);
    auto infer_request = compiled.create_infer_request();

    auto input_tensor = Tensor(element::f16, model->get_parameters()[0]->get_shape());
    auto* in_dst = input_tensor.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        in_dst[i] = ov::float16(input_data[i]);
    }

    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();

    std::vector<float> output_f32(element_count);
    const auto* src = output_tensor.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        output_f32[i] = static_cast<float>(src[i]);
    }

    return compute_metrics(input_data, output_f32.data(), element_count);
}

// ============================================================================
// Helper: get the target device from the environment, empty if not set
// ============================================================================
std::string get_test_device() {
    const char* env = std::getenv("OV_DQ_TEST_DEVICE");
    return env ? std::string(env) : std::string{};
}

// ============================================================================
// Test parameter definition
// ============================================================================
enum class DistributionKind { Uniform, GenGaussian };

struct DQTestParams {
    std::string name;
    int decompose_version;  // 1, 2, or 3 — selects the production decomposition pass
    Shape shape;
    unsigned seed;
    DistributionKind dist_kind;
    // Uniform distribution params
    float uniform_min;
    float uniform_max;
    // Generalized Gaussian params
    float ggd_mu;
    float ggd_alpha;
    float ggd_beta;
};

std::ostream& operator<<(std::ostream& os, const DQTestParams& p) {
    return os << p.name;
}

// ============================================================================
// Parameterized test fixture
// ============================================================================
class DynamicQuantizeAccuracyTest : public ::testing::TestWithParam<DQTestParams> {
protected:
    static constexpr double kRelL2Threshold = 0.02;

    static void print_metrics(const std::string& label, const AccuracyMetrics& m) {
        std::cout << "  " << label
                  << ": L2=" << m.l2_norm
                  << ", relL2=" << m.rel_l2_norm
                  << ", maxAbs=" << m.max_abs_error
                  << ", meanAbs=" << m.mean_abs_error
                  << std::endl;
    }

    std::vector<float> generate_data(const DQTestParams& p, size_t count) {
        switch (p.dist_kind) {
        case DistributionKind::Uniform:
            return generate_uniform_data(count, p.uniform_min, p.uniform_max, p.seed);
        case DistributionKind::GenGaussian:
            return generate_gen_gaussian_data(count, p.ggd_mu, p.ggd_alpha, p.ggd_beta, p.seed);
        default:
            return {};
        }
    }

    // Shared setup: build fp16 model + generate fp16-rounded data
    struct TestContext {
        std::shared_ptr<Model> model;
        std::vector<float> data;
        size_t element_count;
    };

    TestContext setup_test() {
        const auto& p = GetParam();
        size_t element_count = 1;
        for (auto d : p.shape) {
            element_count *= d;
        }
        auto data = generate_data(p, element_count);
        EXPECT_EQ(data.size(), element_count);

        auto model = build_roundtrip_model(p.shape, p.decompose_version);
        model->validate_nodes_and_infer_types();

        return {model, std::move(data), element_count};
    }
};

// ── Reference interpreter test (always runs) ─────────────────────────────────

TEST_P(DynamicQuantizeAccuracyTest, RoundtripAccuracy) {
    auto [model, data, element_count] = setup_test();
    const auto& p = GetParam();

    auto metrics = evaluate_roundtrip(model, data.data(), element_count);

    std::cout << p.name << " (V" << p.decompose_version << ", fp16) [evaluate]:" << std::endl;
    print_metrics("  roundtrip", metrics);

    EXPECT_LT(metrics.rel_l2_norm, kRelL2Threshold)
        << "V" << p.decompose_version << " relative L2 too high";
}

// ── Device-based inference test (opt-in via OV_DQ_TEST_DEVICE env var) ───────
// Run with:  OV_DQ_TEST_DEVICE=CPU  ./ov_npu_unit_tests --gtest_filter=*DeviceRoundtripAccuracy*
//            OV_DQ_TEST_DEVICE=NPU  ./ov_npu_unit_tests --gtest_filter=*DeviceRoundtripAccuracy*

TEST_P(DynamicQuantizeAccuracyTest, DeviceRoundtripAccuracy) {
    const std::string device = get_test_device();
    if (device.empty()) {
        GTEST_SKIP() << "OV_DQ_TEST_DEVICE not set — skipping device inference test";
    }

    auto [model, data, element_count] = setup_test();
    const auto& p = GetParam();

    // Reference: interpreter on the same fp16 model
    auto ref_metrics = evaluate_roundtrip(model, data.data(), element_count);

    AccuracyMetrics dev_metrics;
    try {
        dev_metrics = evaluate_roundtrip_on_device(model, device, data.data(), element_count);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Device '" << device << "' is not available: " << e.what();
    }

    std::cout << p.name << " (V" << p.decompose_version << ", fp16) [" << device << "]:" << std::endl;
    print_metrics("  reference (evaluate)", ref_metrics);
    print_metrics("  device    (" + device + ")", dev_metrics);

    // Device result must meet the same threshold
    EXPECT_LT(dev_metrics.rel_l2_norm, kRelL2Threshold)
        << "V" << p.decompose_version << " on " << device << ": relative L2 too high";

    // Device result should not be dramatically worse than the interpreter
    if (ref_metrics.rel_l2_norm > 1e-9) {
        double ratio = dev_metrics.rel_l2_norm / ref_metrics.rel_l2_norm;
        EXPECT_LT(ratio, 4.0)
            << "V" << p.decompose_version << " on " << device
            << ": device accuracy is " << ratio << "x worse than interpreter";
    }
}

// ============================================================================
// Helper functions for concise param construction
// ============================================================================
// clang-format off
static DQTestParams make_uniform(const std::string& name, int version, Shape shape,
                                 float lo, float hi, unsigned seed) {
    return {name, version, shape, seed, DistributionKind::Uniform,
            lo, hi, 0.0f, 0.0f, 0.0f};
}

static DQTestParams make_ggd(const std::string& name, int version, Shape shape,
                             float mu, float alpha, float beta, unsigned seed) {
    return {name, version, shape, seed, DistributionKind::GenGaussian,
            0.0f, 0.0f, mu, alpha, beta};
}
// clang-format on

// ============================================================================
// Test instantiation — V1: handcrafted symmetric i8 [-127, 127]
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    V1_Uniform,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_uniform("V1_SmallRange",        1, {1, 8, 64, 128},   -1.0f,  1.0f,  42),
        make_uniform("V1_WiderRange",        1, {1, 8, 64, 128},   -10.0f, 10.0f, 77),
        make_uniform("V1_AsymPositive",      1, {1, 8, 64, 128},   0.0f,   5.0f,  999),
        make_uniform("V1_LargeSeq",          1, {1, 8, 1024, 128}, -3.0f,  3.0f,  314)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

INSTANTIATE_TEST_SUITE_P(
    V1_GenGaussian,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_ggd("V1_HeavyTails",   1, {1, 8, 1024, 128}, 0.018f, 1.121f, 0.923f, 42),
        make_ggd("V1_NearLaplace",  1, {1, 8, 1024, 128}, 0.032f, 1.388f, 1.016f, 77),
        make_ggd("V1_NearNormal",   1, {1, 8, 1024, 128}, -0.027f, 1.830f, 1.271f, 123)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

// ============================================================================
// Test instantiation — V2: ONNX DynamicQuantizeLinear u8 [0, 255]
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    V2_Uniform,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_uniform("V2_SmallRange",        2, {1, 8, 64, 128},   -1.0f,  1.0f,  42),
        make_uniform("V2_WiderRange",        2, {1, 8, 64, 128},   -10.0f, 10.0f, 77),
        make_uniform("V2_AsymPositive",      2, {1, 8, 64, 128},   0.0f,   5.0f,  999),
        make_uniform("V2_LargeSeq",          2, {1, 8, 1024, 128}, -3.0f,  3.0f,  314)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

INSTANTIATE_TEST_SUITE_P(
    V2_GenGaussian,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_ggd("V2_HeavyTails",   2, {1, 8, 1024, 128}, 0.018f, 1.121f, 0.923f, 42),
        make_ggd("V2_NearLaplace",  2, {1, 8, 1024, 128}, 0.032f, 1.388f, 1.016f, 77),
        make_ggd("V2_NearNormal",   2, {1, 8, 1024, 128}, -0.027f, 1.830f, 1.271f, 123)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

// ============================================================================
// Test instantiation — V3: compiler pattern style i8 [-128, 127]
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    V3_Uniform,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_uniform("V3_SmallRange",        3, {1, 8, 64, 128},   -1.0f,  1.0f,  42),
        make_uniform("V3_WiderRange",        3, {1, 8, 64, 128},   -10.0f, 10.0f, 77),
        make_uniform("V3_AsymPositive",      3, {1, 8, 64, 128},   0.0f,   5.0f,  999),
        make_uniform("V3_LargeSeq",          3, {1, 8, 1024, 128}, -3.0f,  3.0f,  314)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

INSTANTIATE_TEST_SUITE_P(
    V3_GenGaussian,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_ggd("V3_HeavyTails",   3, {1, 8, 1024, 128}, 0.018f, 1.121f, 0.923f, 42),
        make_ggd("V3_NearLaplace",  3, {1, 8, 1024, 128}, 0.032f, 1.388f, 1.016f, 77),
        make_ggd("V3_NearNormal",   3, {1, 8, 1024, 128}, -0.027f, 1.830f, 1.271f, 123)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

}  // namespace