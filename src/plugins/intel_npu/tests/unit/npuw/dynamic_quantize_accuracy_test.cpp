// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for DynamicQuantize decomposition accuracy.
//
// All models use fp16 input/output — this is the native precision of the
// KV-cache compression subgraphs we want to test.
//
// The test builds models that mirror the real KV-cache compression pipeline,
// which in production consists of TWO separate compiled models:
//
//   1. Quantize model (applied to each new token being stored):
//        Parameter(f16) -> Convert(f32) -> DynamicQuantize -> [q, scale, zp] Results
//
//   2. Dequantize model (applied when reading cached tokens for attention):
//        Parameters(q_i8, scale_f32, zp_i8) -> dequant chain -> Result(f16)
//
// Test cases:
//   RoundtripAccuracy           - single fused roundtrip model via evaluate()
//   SplitModelRoundtripAccuracy - separate quantize + dequantize models via
//                                 evaluate(), validates that each model works
//                                 independently and that the roundtrip through
//                                 both produces acceptable accuracy
//   DeviceRoundtripAccuracy     - opt-in device test (OV_DQ_TEST_DEVICE env var)
//                                 using the split model approach; catches
//                                 compilation failures for quantize-only models
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

#include "npuw_transformations/kv_cache_compressed.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
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
// Test parameter definition
// ============================================================================
enum class DistributionKind { Uniform, GenGaussian };

struct DQTestParams {
    std::string name;
    int decompose_version;
    Shape shape;
    unsigned seed;
    DistributionKind dist_kind;
    float uniform_min;
    float uniform_max;
    float ggd_mu;
    float ggd_alpha;
    float ggd_beta;
    // Quantization config overrides.
    // quant_dt == element::dynamic → infer from decompose_version (u8 for v2, i8 otherwise)
    element::Type quant_dt  = element::dynamic;
    bool          is_symmetric = false;  // true = symmetric (no ZP, 2 DQ outputs)
    double        threshold    = 0.02;   // rel-L2 pass threshold
};

std::ostream& operator<<(std::ostream& os, const DQTestParams& p) {
    return os << p.name;
}


// ============================================================================
// Shared helpers for DQ config and decompose pass
// ============================================================================
op::internal::DynamicQuantize::Attributes make_dq_config(const Shape& shape,
                                                          const DQTestParams& p) {
    using QT = op::internal::DynamicQuantize::QuantizationType;

    // Resolve effective quantization dtype: explicit override beats version default.
    const element::Type quant_dt = (p.quant_dt != element::dynamic)
                                       ? p.quant_dt
                                       : (p.decompose_version == 2 ? element::u8 : element::i8);

    op::internal::DynamicQuantize::Attributes config;
    config.scale_dt    = element::f32;
    config.quantization_dt = quant_dt;
    config.group_sizes = std::vector<uint64_t>(shape.size(), 1);

    if (p.is_symmetric) {
        config.quantization_type = QT::Symmetric;
        config.zp_dt             = element::dynamic;  // no zero-point
    } else {
        config.quantization_type = QT::Asymmetric;
        config.zp_dt             = quant_dt;
    }
    return config;
}

void run_decompose_pass(const std::shared_ptr<Model>& model, int decompose_version) {
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
}

// ============================================================================
// Build a single fused roundtrip model:
//   Parameter(f16) -> Convert(f32) -> [DQ + dequant in f32] -> Convert(f16) -> Result(f16)
//
// The DynamicQuantize node is named with "key" so the decomposition pass
// selects reduction_axis=3 (the key-cache embedding dimension).
// ============================================================================
std::shared_ptr<Model> build_roundtrip_model(const Shape& shape,
                                             const DQTestParams& p) {
    auto input = std::make_shared<op::v0::Parameter>(element::f16, shape);
    input->set_friendly_name("input");

    auto dq_input = std::make_shared<op::v0::Convert>(input, element::f32);
    dq_input->set_friendly_name("input_cvt_f32");

    auto config = make_dq_config(shape, p);

    auto dq = std::make_shared<op::internal::DynamicQuantize>(dq_input, config);
    dq->set_friendly_name("DynamicQuantize/0/key");

    // Dequant chain depends on symmetry:
    //   Symmetric:  q * scale
    //   Asymmetric: (q - zp) * scale
    auto q_f32 = std::make_shared<op::v0::Convert>(dq->output(0), element::f32);
    std::shared_ptr<Node> dequant_input;
    if (p.is_symmetric) {
        dequant_input = q_f32;
    } else {
        auto zp_f32 = std::make_shared<op::v0::Convert>(dq->output(2), element::f32);
        dequant_input = std::make_shared<op::v1::Subtract>(q_f32, zp_f32);
    }
    auto reconstructed = std::make_shared<op::v1::Multiply>(dequant_input, dq->output(1));

    auto output_cvt = std::make_shared<op::v0::Convert>(reconstructed, element::f16);
    output_cvt->set_friendly_name("output_cvt_f16");

    auto result = std::make_shared<op::v0::Result>(output_cvt);
    result->set_friendly_name("result_reconstructed");

    auto model = std::make_shared<Model>(
        ResultVector{result},
        ParameterVector{input},
        "roundtrip_fp16_v" + std::to_string(p.decompose_version));

    model->validate_nodes_and_infer_types();
    run_decompose_pass(model, p.decompose_version);
    return model;
}

// ============================================================================
// Build a quantize-only model (matches the real "store new token" path):
//   Parameter(f16) -> Convert(f32) -> DynamicQuantize -> Results(q, scale, zp)
//
// In production this is compiled as a separate subgraph.  A compilation
// failure here (which has been observed) would go unnoticed with a single
// fused roundtrip model.
// ============================================================================
std::shared_ptr<Model> build_quantize_model(const Shape& shape,
                                            const DQTestParams& p) {
    auto input = std::make_shared<op::v0::Parameter>(element::f16, shape);
    input->set_friendly_name("input");

    auto cvt_f32 = std::make_shared<op::v0::Convert>(input, element::f32);
    cvt_f32->set_friendly_name("input_cvt_f32");

    auto config = make_dq_config(shape, p);

    auto dq = std::make_shared<op::internal::DynamicQuantize>(cvt_f32, config);
    dq->set_friendly_name("DynamicQuantize/0/key");

    auto result_q = std::make_shared<op::v0::Result>(dq->output(0));
    result_q->set_friendly_name("result_quantized");

    auto result_scale = std::make_shared<op::v0::Result>(dq->output(1));
    result_scale->set_friendly_name("result_scale");

    ResultVector results{result_q, result_scale};
    if (!p.is_symmetric) {
        auto result_zp = std::make_shared<op::v0::Result>(dq->output(2));
        result_zp->set_friendly_name("result_zp");
        results.push_back(result_zp);
    }

    auto model = std::make_shared<Model>(
        results,
        ParameterVector{input},
        "quantize_fp16_v" + std::to_string(p.decompose_version));

    model->validate_nodes_and_infer_types();
    run_decompose_pass(model, p.decompose_version);
    return model;
}

// ============================================================================
// Build a dequantize-only model (matches the real "read cached tokens" path):
//   Parameters(q, scale, zp) -> (convert(q,f32) - convert(zp,f32)) * scale
//                             -> Convert(f16) -> Result
//
// This mirrors create_dequant_nodes() in the production code.
// No decompose pass needed -- the dequant chain is plain arithmetic.
// ============================================================================
std::shared_ptr<Model> build_dequantize_model(const Shape& shape,
                                              const DQTestParams& p) {
    auto config = make_dq_config(shape, p);

    auto param_q = std::make_shared<op::v0::Parameter>(config.quantization_dt, shape);
    param_q->set_friendly_name("quantized_data");

    // Scale/ZP shape: same as data but with embedding dim (last) collapsed to 1
    Shape scale_shape = shape;
    scale_shape.back() = 1;

    auto param_scale = std::make_shared<op::v0::Parameter>(element::f32, scale_shape);
    param_scale->set_friendly_name("scale");

    // Dequant chain depends on symmetry:
    //   Symmetric:  convert(q, f32) * scale
    //   Asymmetric: (convert(q, f32) - convert(zp, f32)) * scale
    auto q_f32 = std::make_shared<op::v0::Convert>(param_q, element::f32);
    q_f32->set_friendly_name("q_cvt_f32");

    std::shared_ptr<Node> dequant_input;
    ParameterVector params{param_q, param_scale};

    if (p.is_symmetric) {
        dequant_input = q_f32;
    } else {
        auto param_zp = std::make_shared<op::v0::Parameter>(config.zp_dt, scale_shape);
        param_zp->set_friendly_name("zero_point");
        params.push_back(param_zp);

        auto zp_f32 = std::make_shared<op::v0::Convert>(param_zp, element::f32);
        zp_f32->set_friendly_name("zp_cvt_f32");
        auto sub_zp = std::make_shared<op::v1::Subtract>(q_f32, zp_f32);
        sub_zp->set_friendly_name("sub_zp");
        dequant_input = sub_zp;
    }

    auto dequantized = std::make_shared<op::v1::Multiply>(dequant_input, param_scale);
    dequantized->set_friendly_name("dequantized");

    auto output_cvt = std::make_shared<op::v0::Convert>(dequantized, element::f16);
    output_cvt->set_friendly_name("output_cvt_f16");

    auto result = std::make_shared<op::v0::Result>(output_cvt);
    result->set_friendly_name("result_dequantized");

    auto model = std::make_shared<Model>(
        ResultVector{result},
        params,
        "dequantize_fp16_v" + std::to_string(p.decompose_version));

    model->validate_nodes_and_infer_types();
    return model;
}

// ============================================================================
// Data generation helpers
// Generated values are rounded to fp16 so the source distribution is exactly
// representable -- no f32->f16 conversion artifacts in error measurements.
// ============================================================================

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

    std::vector<float> output_f32(element_count);
    const auto* src = outputs[0].data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        output_f32[i] = static_cast<float>(src[i]);
    }

    return compute_metrics(input_data, output_f32.data(), element_count);
}

// ============================================================================
// Helper: run split quantize -> dequantize pipeline via the reference
// interpreter.  Executes two separate model->evaluate() calls, passing
// intermediate tensors (q, scale, zp) from the quantize model to the
// dequantize model -- exactly as the real NPU pipeline does.
// ============================================================================
AccuracyMetrics evaluate_split_roundtrip(const std::shared_ptr<Model>& quant_model,
                                         const std::shared_ptr<Model>& dequant_model,
                                         const float* input_data,
                                         size_t element_count) {
    // Step 1: run quantize model
    auto quant_input = Tensor(element::f16, quant_model->get_parameters()[0]->get_shape());
    auto* q_dst = quant_input.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        q_dst[i] = ov::float16(input_data[i]);
    }

    TensorVector quant_outputs;
    for (size_t i = 0; i < quant_model->get_results().size(); ++i) {
        const auto& res = quant_model->get_results()[i];
        quant_outputs.emplace_back(res->get_output_element_type(0), res->get_output_shape(0));
    }

    TensorVector quant_inputs{quant_input};
    bool ok = quant_model->evaluate(quant_outputs, quant_inputs);
    EXPECT_TRUE(ok) << "Quantize model evaluation failed";

    // Step 2: run dequantize model
    // Pass all quantize outputs as dequantize inputs (2 for symmetric, 3 for asymmetric).
    TensorVector dequant_inputs(quant_outputs.begin(), quant_outputs.end());

    auto dequant_output = Tensor(element::f16, dequant_model->get_results()[0]->get_output_shape(0));
    TensorVector dequant_outputs{dequant_output};

    ok = dequant_model->evaluate(dequant_outputs, dequant_inputs);
    EXPECT_TRUE(ok) << "Dequantize model evaluation failed";

    std::vector<float> output_f32(element_count);
    const auto* src = dequant_outputs[0].data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        output_f32[i] = static_cast<float>(src[i]);
    }

    return compute_metrics(input_data, output_f32.data(), element_count);
}

// ============================================================================
// Helper: run split quantize -> dequantize pipeline on a real device.
// Compiles and infers each model separately, passing intermediate tensors.
// ============================================================================
AccuracyMetrics evaluate_split_roundtrip_on_device(const std::shared_ptr<Model>& quant_model,
                                                   const std::shared_ptr<Model>& dequant_model,
                                                   const std::string& device_name,
                                                   const float* input_data,
                                                   size_t element_count) {
    Core core;
    ov::AnyMap device_config;
    if (device_name == "NPU") {
        device_config[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
    }

    // Step 1: compile and run quantize model
    auto compiled_quant = core.compile_model(quant_model, device_name, device_config);
    auto quant_request = compiled_quant.create_infer_request();

    auto quant_input = Tensor(element::f16, quant_model->get_parameters()[0]->get_shape());
    auto* q_dst = quant_input.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        q_dst[i] = ov::float16(input_data[i]);
    }
    quant_request.set_input_tensor(quant_input);
    quant_request.infer();

    auto tensor_q     = quant_request.get_output_tensor(0);
    auto tensor_scale = quant_request.get_output_tensor(1);

    // Step 2: compile and run dequantize model
    auto compiled_dequant = core.compile_model(dequant_model, device_name, device_config);
    auto dequant_request = compiled_dequant.create_infer_request();

    dequant_request.set_input_tensor(0, tensor_q);
    dequant_request.set_input_tensor(1, tensor_scale);
    // For asymmetric quantization a ZP tensor exists as the 3rd output/input.
    const bool has_zp = (quant_model->get_results().size() == 3);
    if (has_zp) {
        auto tensor_zp = quant_request.get_output_tensor(2);
        dequant_request.set_input_tensor(2, tensor_zp);
    }
    dequant_request.infer();

    auto output_tensor = dequant_request.get_output_tensor();

    std::vector<float> output_f32(element_count);
    const auto* src = output_tensor.data<ov::float16>();
    for (size_t i = 0; i < element_count; ++i) {
        output_f32[i] = static_cast<float>(src[i]);
    }

    return compute_metrics(input_data, output_f32.data(), element_count);
}

// ============================================================================
// Helper: evaluate a roundtrip model on a real device via ov::Core.
// ============================================================================
AccuracyMetrics evaluate_roundtrip_on_device(const std::shared_ptr<Model>& model,
                                             const std::string& device_name,
                                             const float* input_data,
                                             size_t element_count) {
    Core core;
    ov::AnyMap device_config;
    if (device_name == "NPU") {
        device_config[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;
    }

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
// Parameterized test fixture
// ============================================================================
class DynamicQuantizeAccuracyTest : public ::testing::TestWithParam<DQTestParams> {
protected:
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

    size_t compute_element_count(const Shape& shape) {
        size_t n = 1;
        for (auto d : shape) {
            n *= d;
        }
        return n;
    }

    // Setup for the single fused roundtrip model
    struct TestContext {
        std::shared_ptr<Model> model;
        std::vector<float> data;
        size_t element_count;
    };

    TestContext setup_test() {
        const auto& p = GetParam();
        size_t element_count = compute_element_count(p.shape);
        auto data = generate_data(p, element_count);
        EXPECT_EQ(data.size(), element_count);

        auto model = build_roundtrip_model(p.shape, p);
        model->validate_nodes_and_infer_types();

        return {model, std::move(data), element_count};
    }

    // Setup for the split quantize + dequantize models
    struct SplitTestContext {
        std::shared_ptr<Model> quant_model;
        std::shared_ptr<Model> dequant_model;
        std::vector<float> data;
        size_t element_count;
    };

    SplitTestContext setup_split_test() {
        const auto& p = GetParam();
        size_t element_count = compute_element_count(p.shape);
        auto data = generate_data(p, element_count);
        EXPECT_EQ(data.size(), element_count);

        auto quant_model = build_quantize_model(p.shape, p);
        auto dequant_model = build_dequantize_model(p.shape, p);

        return {quant_model, dequant_model, std::move(data), element_count};
    }
};

// -- Reference interpreter test -- single fused roundtrip (always runs) -------

TEST_P(DynamicQuantizeAccuracyTest, RoundtripAccuracy) {
    auto [model, data, element_count] = setup_test();
    const auto& p = GetParam();

    auto metrics = evaluate_roundtrip(model, data.data(), element_count);

    std::cout << p.name << " (V" << p.decompose_version << ", fp16) [evaluate]:" << std::endl;
    print_metrics("  roundtrip", metrics);

    EXPECT_LT(metrics.rel_l2_norm, p.threshold)
        << "V" << p.decompose_version << " relative L2 too high";
}

// -- Reference interpreter test -- split quantize + dequantize models ---------
// This matches the real pipeline: two separate models with intermediate
// tensors passed between them.

TEST_P(DynamicQuantizeAccuracyTest, SplitModelRoundtripAccuracy) {
    auto [quant_model, dequant_model, data, element_count] = setup_split_test();
    const auto& p = GetParam();

    auto metrics = evaluate_split_roundtrip(quant_model, dequant_model,
                                            data.data(), element_count);

    std::cout << p.name << " (V" << p.decompose_version << ", fp16) [split evaluate]:"
              << std::endl;
    print_metrics("  split roundtrip", metrics);

    EXPECT_LT(metrics.rel_l2_norm, p.threshold)
        << "V" << p.decompose_version << " split model relative L2 too high";
}

// -- Device-based inference test -- split models (opt-in via OV_DQ_TEST_DEVICE)
// Compiles quantize and dequantize models SEPARATELY on the device.
// This catches compilation failures for quantize-only models (which have
// been observed in practice) and measures real device error accumulation.
//
// Run with:  OV_DQ_TEST_DEVICE=CPU  ./ov_npu_unit_tests --gtest_filter=*DeviceRoundtripAccuracy*
//            OV_DQ_TEST_DEVICE=NPU  ./ov_npu_unit_tests --gtest_filter=*DeviceRoundtripAccuracy*

TEST_P(DynamicQuantizeAccuracyTest, DeviceRoundtripAccuracy) {
    const std::string device = get_test_device();
    if (device.empty()) {
        GTEST_SKIP() << "OV_DQ_TEST_DEVICE not set -- skipping device inference test";
    }

    auto [quant_model, dequant_model, data, element_count] = setup_split_test();
    const auto& p = GetParam();

    // Reference: interpreter on the split models
    auto ref_metrics = evaluate_split_roundtrip(quant_model, dequant_model,
                                                data.data(), element_count);

    AccuracyMetrics dev_metrics;
    try {
        dev_metrics = evaluate_split_roundtrip_on_device(quant_model, dequant_model,
                                                         device, data.data(), element_count);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Device '" << device << "' is not available: " << e.what();
    }

    std::cout << p.name << " (V" << p.decompose_version << ", fp16) [" << device
              << ", split]:" << std::endl;
    print_metrics("  reference (evaluate)", ref_metrics);
    print_metrics("  device    (" + device + ")", dev_metrics);

    EXPECT_LT(dev_metrics.rel_l2_norm, p.threshold)
        << "V" << p.decompose_version << " on " << device << ": relative L2 too high";

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

// Symmetric i4 factories: decompose_version=1 (V1 now handles i4 via dtype dispatch),
// is_symmetric=true (2 DQ outputs, no ZP), higher threshold (coarser 4-bit grid).
static DQTestParams make_sym_i4_uniform(const std::string& name, Shape shape,
                                        float lo, float hi, unsigned seed) {
    DQTestParams p = make_uniform(name, 1, shape, lo, hi, seed);
    p.quant_dt     = element::i4;
    p.is_symmetric = true;
    p.threshold    = 0.15;  // i4: ±7 levels → higher quantisation error
    return p;
}

static DQTestParams make_sym_i4_ggd(const std::string& name, Shape shape,
                                    float mu, float alpha, float beta, unsigned seed) {
    DQTestParams p = make_ggd(name, 1, shape, mu, alpha, beta, seed);
    p.quant_dt     = element::i4;
    p.is_symmetric = true;
    // GGD heavy-tail distributions produce outliers that compress the i4 scale,
    // causing higher quantisation error than uniform data (~0.25 observed).
    p.threshold    = 0.40;
    return p;
}
// clang-format on

// ============================================================================
// Test instantiation -- V1: handcrafted symmetric i8 [-127, 127]
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
// Test instantiation -- V2: ONNX DynamicQuantizeLinear u8 [0, 255]
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
// Test instantiation -- V3: compiler pattern style i8 [-128, 127]
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

// ============================================================================
// Test instantiation -- V1 symmetric i4: handcrafted symmetric i4 [-7, 7]
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    V1_Symmetric_i4_Uniform,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_sym_i4_uniform("V1_i4_SmallRange",   {1, 8, 64, 128},   -1.0f,  1.0f,  42),
        make_sym_i4_uniform("V1_i4_WiderRange",   {1, 8, 64, 128},   -10.0f, 10.0f, 77),
        make_sym_i4_uniform("V1_i4_AsymPositive", {1, 8, 64, 128},   0.0f,   5.0f,  999),
        make_sym_i4_uniform("V1_i4_LargeSeq",     {1, 8, 1024, 128}, -3.0f,  3.0f,  314)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

INSTANTIATE_TEST_SUITE_P(
    V1_Symmetric_i4_GenGaussian,
    DynamicQuantizeAccuracyTest,
    ::testing::Values(
        make_sym_i4_ggd("V1_i4_HeavyTails",  {1, 8, 1024, 128}, 0.018f, 1.121f, 0.923f, 42),
        make_sym_i4_ggd("V1_i4_NearLaplace", {1, 8, 1024, 128}, 0.032f, 1.388f, 1.016f, 77),
        make_sym_i4_ggd("V1_i4_NearNormal",  {1, 8, 1024, 128}, -0.027f, 1.830f, 1.271f, 123)
    ),
    [](const ::testing::TestParamInfo<DQTestParams>& info) { return info.param.name; }
);

}  // namespace
