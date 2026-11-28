// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct softmax_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    int64_t dimension;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

class SoftmaxPrimitiveFusingTest : public ::BaseFusingTest<softmax_test_params> {
public:

    void execute(softmax_test_params& p, std::map<std::string, std::vector<std::string>> expected_fused_primitives_ids = {}) {
        auto input_prim = get_mem(get_input_layout(p));

        if (!p.kernel_name.empty()) {
            ov::intel_gpu::ImplementationDesc softmax_target_impl = { format::bfyx, p.kernel_name };
            cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"softmax", softmax_target_impl} }));
        }

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        check_fusions_correctness(network_fused, expected_fused_primitives_ids);
    }

    layout get_input_layout(softmax_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    tensor get_reshape_shape(softmax_test_params& p) {
        auto output_shape = p.in_shape;
        output_shape.feature[0] *= output_shape.spatial[0];
        output_shape.spatial[0] = 1;
        return output_shape;
    }

    layout get_output_layout(softmax_test_params& p) {
        return { data_types::f32, p.default_format, get_reshape_shape(p) };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- SoftMax cases ---------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_SOFTMAX_FP32_1 {1, 15, 4, 5}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_2 {1, 15, 4, 5}, data_types::f32, format::bfyx, 3, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_3 {1, 15, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_SOFTMAX_FP32_4 {1, 15, 1, 2}, data_types::f32, format::bfyx, 2, data_types::f32, format::bfyx

#define CASE_SOFTMAX_FP16_1 {1, 15, 4, 5}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_2 {1, 15, 4, 5}, data_types::f16, format::bfyx, 3, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_3 {1, 15, 1, 1}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx
#define CASE_SOFTMAX_FP16_4 {1, 15, 1, 2}, data_types::f16, format::bfyx, 2, data_types::f16, format::bfyx

#define CASE_SOFTMAX_BF16_1 {1, 15, 4, 5}, data_types::bf16, format::bfyx, 1, data_types::bf16, format::bfyx
#define CASE_SOFTMAX_BF16_2 {1, 15, 4, 5}, data_types::bf16, format::bfyx, 3, data_types::bf16, format::bfyx
#define CASE_SOFTMAX_BF16_3 {1, 15, 1, 1}, data_types::bf16, format::bfyx, 1, data_types::bf16, format::bfyx
#define CASE_SOFTMAX_BF16_4 {1, 15, 1, 2}, data_types::bf16, format::bfyx, 2, data_types::bf16, format::bfyx

// Keep batch > 1 and use larger odd feature count to force both packed and tail
// fused-output write loops in softmax_gpu_bf (the pre-fix bug used wrong index in tail writes).
#define CASE_SOFTMAX_FP32_BF_FUSED_INDEXING {2, 8193, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
// Runtime reproducer for stale tail write index in softmax_gpu_bf:
// feature=7000 is large enough to trigger scalar tail handling in common dispatch configurations.
#define CASE_SOFTMAX_FP32_BF_BUG_REPRO {1, 7000, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx

class softmax_quantize : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize, basic) {
    auto p = GetParam();
    auto quant_layout = get_single_element_layout(p);
    if (p.default_type == data_types::bf16) quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, min_random, 0)),
        data("in_hi", get_mem(quant_layout, 1, max_random)),
        data("out_lo", get_mem(quant_layout, -127)),
        data("out_hi", get_mem(quant_layout, 127)),
        softmax("softmax", input_info("input"), p.dimension),
        quantize("quantize", input_info("softmax"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_quantize,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 2, 3 },

                        softmax_test_params{ CASE_SOFTMAX_FP16_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 2, 3 },

                        softmax_test_params{ CASE_SOFTMAX_BF16_1, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_BF16_2, 3, 3 },
                        softmax_test_params{ CASE_SOFTMAX_BF16_3, 2, 3 },
}));

class softmax_activation : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        softmax("softmax", input_info("input"), p.dimension),
        activation("log", input_info("softmax"), activation_func::log),
        reorder("reorder_bfyx", input_info("log"), get_output_layout(p))
    );

    tolerance = default_tolerance(p.data_type);
    if (p.default_type == data_types::f16 || p.default_type == data_types::bf16)
        tolerance *= 2; // Issue: 185375

    execute(p);
}

class softmax_eltwise_activation : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_eltwise_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_single_element_layout(p), 1.25f)),
        softmax("softmax", input_info("input"), p.dimension),
        eltwise("scale", { input_info("softmax"), input_info("eltwise_data") }, eltwise_mode::prod, p.default_type),
        activation("log", input_info("scale"), activation_func::log),
        reorder("reorder_bfyx", input_info("log"), get_output_layout(p))
    );

    tolerance = default_tolerance(p.data_type);
    if (p.default_type == data_types::bf16)
        tolerance = 1.6e-2f; // Default 1e-2 is too small for this test, but error is still within 1ULP, so it's OK
    execute(p);
}

class softmax_eltwise_per_element : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_eltwise_per_element, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_input_layout(p), min_random, max_random)),
        softmax("softmax", input_info("input"), p.dimension),
        eltwise("sum", { input_info("softmax"), input_info("eltwise_data") }, eltwise_mode::sum, p.default_type),
        reorder("reorder_bfyx", input_info("sum"), get_output_layout(p))
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

class softmax_activation_bug_repro : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_activation_bug_repro, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        softmax("softmax", input_info("input"), p.dimension),
        activation("log", input_info("softmax"), activation_func::log),
        reorder("reorder_bfyx", input_info("log"), get_output_layout(p))
    );

    tolerance = default_tolerance(p.data_type);
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_activation,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP32_4, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP32_BF_FUSED_INDEXING, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_4, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_BF16_3, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_BF16_4, 2, 3, "softmax_gpu_bf" }
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_eltwise_activation,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_BF_FUSED_INDEXING, 3, 4, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 3, 4, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_BF16_3, 3, 4, "softmax_gpu_bf" }
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_eltwise_per_element,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_BF_FUSED_INDEXING, 3, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 3, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_SOFTMAX_BF16_3, 3, 3, "softmax_gpu_bf" }
}));

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_activation_bug_repro,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_BF_BUG_REPRO, 2, 3, "softmax_gpu_bf" }
}));

// LogSoftmax (softmax with log_softmax = true) with a fused per-tensor quantize.
//
// Two properties are needed for these cases to gate the LogSoftmax behaviour:
//   * the quantization window must straddle LogSoftmax and Softmax. Inputs span [-18, 2], so
//     LogSoftmax results reach about [-20, -1.5] (in f16 the low end is exactly where
//     exp(x - max) underflows and log(softmax(x)) would return -inf) while softmax values live in
//     (0, 1]. With in_hi = 0 and 256 levels over [-25, 0] the step is ~0.1, so both a dropped
//     log(sum) term (~2 => ~20 levels) and plain softmax (clamped to out_hi) are far outside the
//     tolerance. The input spread must stay moderate: with a very wide range only the maximum
//     element survives exp(x - max), log(sum) collapses to ~0 and formula errors become invisible.
//   * the fused output must be compared against a host reference and not only against the
//     non-fused network: both networks run the same kernel, so they keep agreeing with each other
//     even if that kernel computes log(softmax(x)) rather than the specified formula.
// Spatial sizes stay at 1 so that the logical (batch, feature) order equals the linear order used
// by the host reference.
class log_softmax_fusing_test : public SoftmaxPrimitiveFusingTest {
public:
    static constexpr float in_lo = -25.f;
    static constexpr float in_hi = 0.f;
    static constexpr float out_lo = -127.f;
    static constexpr float out_hi = 127.f;
    static constexpr int levels = 256;

    void execute(softmax_test_params& p) {
        ASSERT_EQ(p.dimension, 1) << "the host reference normalizes over the feature axis";
        ASSERT_EQ(p.in_shape.spatial[0], 1) << "the host reference assumes logical == linear order";
        ASSERT_EQ(p.in_shape.spatial[1], 1);

        const auto in_layout = get_input_layout(p);
        const size_t count = in_layout.get_tensor().count();
        auto input_prim = engine.allocate_memory(in_layout);
        std::vector<float> host_values;
        if (p.data_type == data_types::f16) {
            auto values = rg.generate_random_1d<ov::float16>(count, -18.f, 2.f);
            set_values(input_prim, values);
            host_values.assign(values.begin(), values.end());
        } else if (p.data_type == data_types::bf16) {
            auto values = rg.generate_random_1d<ov::bfloat16>(count, -18.f, 2.f);
            set_values(input_prim, values);
            host_values.assign(values.begin(), values.end());
        } else {
            host_values = rg.generate_random_1d<float>(count, -18.f, 2.f);
            set_values(input_prim, host_values);
        }

        if (!p.kernel_name.empty()) {
            ov::intel_gpu::ImplementationDesc target_impl = { p.input_format, p.kernel_name };
            cfg_fused.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"softmax", target_impl}}));
        }

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        // Fused and non-fused networks must agree, and the executed primitive counts must match.
        compare(network_not_fused, network_fused, p);
        // The quantize must really have been absorbed by the softmax primitive.
        check_fusions_correctness(network_fused, {{"softmax", {"quantize"}}});

        // Absolute anchor: the fused result must be LogSoftmax, not log(softmax(x)).
        auto outputs_fused = network_fused.execute();
        ASSERT_EQ(outputs_fused.size(), size_t(1));
        const auto got = get_output_values_to_float(network_fused, outputs_fused.begin()->second, count);
        const auto expected = log_softmax_quantized(p, host_values);
        ASSERT_EQ(got.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(got[i], expected[i], tolerance)
                << "tolerance = " << tolerance << "\ni = " << i
                << "\nexpected[i] = " << expected[i]
                << "\ngot[i] = " << got[i];
        }
    }

private:
    // f64 LogSoftmax over the feature axis followed by the same per-tensor quantization.
    std::vector<float> log_softmax_quantized(const softmax_test_params& p, const std::vector<float>& in) const {
        const size_t batch = static_cast<size_t>(p.in_shape.batch[0]);
        const size_t feature = static_cast<size_t>(p.in_shape.feature[0]);
        const double step_in = (in_hi - in_lo) / (levels - 1);
        const double step_out = (out_hi - out_lo) / (levels - 1);
        std::vector<float> out(in.size());
        for (size_t b = 0; b < batch; ++b) {
            const float* row = in.data() + b * feature;
            double max_value = -std::numeric_limits<double>::max();
            for (size_t f = 0; f < feature; ++f)
                max_value = std::max(max_value, static_cast<double>(row[f]));
            double sum = 0;
            for (size_t f = 0; f < feature; ++f)
                sum += std::exp(static_cast<double>(row[f]) - max_value);
            const double log_sum = std::log(sum);
            for (size_t f = 0; f < feature; ++f) {
                const double log_prob = static_cast<double>(row[f]) - max_value - log_sum;
                const double clamped = std::min(std::max(log_prob, static_cast<double>(in_lo)),
                                                static_cast<double>(in_hi));
                out[b * feature + f] = static_cast<float>(std::round((clamped - in_lo) / step_in) * step_out + out_lo);
            }
        }
        return out;
    }

};

class log_softmax_quantize : public log_softmax_fusing_test {};
TEST_P(log_softmax_quantize, basic) {
    auto p = GetParam();
    auto quant_layout = get_single_element_layout(p);
    if (p.default_type == data_types::bf16) quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, in_lo)),
        data("in_hi", get_mem(quant_layout, in_hi)),
        data("out_lo", get_mem(quant_layout, out_lo)),
        data("out_hi", get_mem(quant_layout, out_hi)),
        softmax("softmax", input_info("input"), p.dimension, true),
        quantize("quantize", input_info("softmax"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), levels, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.6f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, log_softmax_quantize,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_3, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_FP16_3, 2, 3 },
                        softmax_test_params{ CASE_SOFTMAX_BF16_3, 2, 3 },
}));

// The same anchor for the three remaining softmax kernels: their HAS_FUSED_OPS epilogues are never
// executed by the cases above, because an i8 quantize output is only supported by softmax_gpu_ref.
#define CASE_LOGSOFTMAX_BF_F32 {2, 15, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_LOGSOFTMAX_BF_F16 {2, 15, 1, 1}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx
#define CASE_LOGSOFTMAX_FB_F16 {1, 15, 1, 1}, data_types::f16, format::yxfb, 1, data_types::f16, format::bfyx
#define CASE_LOGSOFTMAX_ITEMS_CLASS_F32 {2, 256, 1, 1}, data_types::f32, format::bfyx, 1, data_types::f32, format::bfyx
#define CASE_LOGSOFTMAX_ITEMS_CLASS_F16 {2, 256, 1, 1}, data_types::f16, format::bfyx, 1, data_types::f16, format::bfyx

class log_softmax_quantize_kernels : public log_softmax_fusing_test {};
TEST_P(log_softmax_quantize_kernels, basic) {
    auto p = GetParam();
    auto quant_layout = get_single_element_layout(p);
    quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, in_lo)),
        data("in_hi", get_mem(quant_layout, in_hi)),
        data("out_lo", get_mem(quant_layout, out_lo)),
        data("out_hi", get_mem(quant_layout, out_hi)),
        softmax("softmax", input_info("input"), p.dimension, true),
        quantize("quantize", input_info("softmax"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), levels, data_types::f16),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.6f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, log_softmax_quantize_kernels,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_LOGSOFTMAX_BF_F32, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_LOGSOFTMAX_BF_F16, 2, 3, "softmax_gpu_bf" },
                        softmax_test_params{ CASE_LOGSOFTMAX_FB_F16, 2, 3, "softmax_gpu_fb" },
                        softmax_test_params{ CASE_LOGSOFTMAX_ITEMS_CLASS_F32, 2, 3, "softmax_gpu_items_class_optimized" },
                        softmax_test_params{ CASE_LOGSOFTMAX_ITEMS_CLASS_F16, 2, 3, "softmax_gpu_items_class_optimized" },
}));

class softmax_quantize_fusing_through : public SoftmaxPrimitiveFusingTest {};
TEST_P(softmax_quantize_fusing_through, reshape) {
    auto p = GetParam();
    auto quant_layout = get_single_element_layout(p);
    if (p.default_type == data_types::bf16) quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, min_random, 0)),
        data("in_hi", get_mem(quant_layout, 1, max_random)),
        data("out_lo", get_mem(quant_layout, -127)),
        data("out_hi", get_mem(quant_layout, 127)),
        softmax("softmax", input_info("input"), p.dimension),
        reshape("reshape", input_info("softmax"), get_reshape_shape(p)),
        quantize("quantize", input_info("reshape"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

TEST_P(softmax_quantize_fusing_through, reorder) {
    auto p = GetParam();
    auto reorder_layout = layout{ p.data_type, p.input_format, get_reshape_shape(p), padding{} };
    auto quant_layout = get_single_element_layout(p);
    if (p.default_type == data_types::bf16) quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, min_random, 0)),
        data("in_hi", get_mem(quant_layout, 1, max_random)),
        data("out_lo", get_mem(quant_layout, -127)),
        data("out_hi", get_mem(quant_layout, 127)),
        softmax("softmax", input_info("input"), p.dimension),
        reorder("reorder", input_info("softmax"), reorder_layout),
        quantize("quantize", input_info("reorder"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

TEST_P(softmax_quantize_fusing_through, chain) {
    auto p = GetParam();
    auto reorder_layout = layout{ p.data_type, p.input_format, get_reshape_shape(p), padding{} };
    auto quant_layout = get_single_element_layout(p);
    if (p.default_type == data_types::bf16) quant_layout.data_type = data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(quant_layout, min_random, 0)),
        data("in_hi", get_mem(quant_layout, 1, max_random)),
        data("out_lo", get_mem(quant_layout, -127)),
        data("out_hi", get_mem(quant_layout, 127)),
        softmax("softmax", input_info("input"), p.dimension),
        reshape("reshape_first", input_info("softmax"), get_reshape_shape(p)),
        reorder("reorder", input_info("reshape_first"), reorder_layout),
        reshape("reshape_second", input_info("reorder"), get_reshape_shape(p)),
        quantize("quantize", input_info("reshape_second"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), get_output_layout(p))
    );

    tolerance = 1.f;
    execute(p, {{"softmax", {"quantize"}}});
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, softmax_quantize_fusing_through,
    ::testing::ValuesIn(std::vector<softmax_test_params>{
                        softmax_test_params{ CASE_SOFTMAX_FP32_1, 2, 3 },
                        // Such fusions not allowed yet
                        // softmax_test_params{ CASE_SOFTMAX_FP32_2, 3, 3 },
                        // softmax_test_params{ CASE_SOFTMAX_FP32_3, 3, 3 },

                        softmax_test_params{ CASE_SOFTMAX_FP16_1, 2, 3 },
                        // Such fusions not allowed yet
                        // softmax_test_params{ CASE_SOFTMAX_FP16_2, 3, 3 },
                        // softmax_test_params{ CASE_SOFTMAX_FP16_3, 3, 3 },

                        softmax_test_params{ CASE_SOFTMAX_BF16_1, 2, 3 },
                        // Such fusions not allowed yet
                        // softmax_test_params{ CASE_SOFTMAX_BF16_2, 3, 3 },
                        // softmax_test_params{ CASE_SOFTMAX_BF16_3, 3, 3 },
}));
