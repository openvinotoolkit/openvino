// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the CM gated_residual primitive (design section 12).
//
// Covered:
//  - READ / MIX: per-branch, per-channel comparison with an FP32 reference, including
//    the branch-wise grouped RMSNorm, the 1/C inside SiLU(W_down * Rbar / C) and the
//    branch mean;
//  - the optional write gate produced by READ: 2 * sigmoid(W_inject * Rbar / C), and
//    that it stays inside (0, 2);
//  - WRITE / COMBINE: R'_c = R_c + S_c * Y for both in-place and out-of-place buffers;
//  - FINAL_MIX: the model-tail mixer that replaces the usual final norm;
//  - negative case: an unsupported branch count must not select the CM implementation.

#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gated_residual.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iostream>
#include <vector>

#include "gated_residual_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

constexpr size_t kBranches = 4;

bool cm_gr_available() {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
}

float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

struct gr_ref_params {
    size_t rows = 0;
    size_t C = kBranches;
    size_t D = 0;
    size_t L = 0;
    float eps = 1e-6f;
};

/// Reference branch-wise grouped RMSNorm: each branch normalizes over its own D channels.
std::vector<float> reference_rbar(const gr_ref_params& p, const std::vector<float>& residual, const std::vector<float>& gain) {
    std::vector<float> rbar(p.rows * p.C * p.D, 0.0f);
    for (size_t row = 0; row < p.rows; row++) {
        for (size_t c = 0; c < p.C; c++) {
            double sumsq = 0.0;
            for (size_t d = 0; d < p.D; d++) {
                const float v = residual[row * p.C * p.D + c * p.D + d];
                sumsq += static_cast<double>(v) * v;
            }
            const float inv = 1.0f / std::sqrt(static_cast<float>(sumsq / p.D) + p.eps);
            for (size_t d = 0; d < p.D; d++) {
                const size_t idx = row * p.C * p.D + c * p.D + d;
                rbar[idx] = residual[idx] * inv * gain[c * p.D + d];
            }
        }
    }
    return rbar;
}

void reference_read(const gr_ref_params& p,
                    const std::vector<float>& residual,
                    const std::vector<float>& gain,
                    const std::vector<float>& w_down,  // [L, C*D]
                    const std::vector<float>& w_up,    // [C*D, L]
                    const std::vector<float>* w_inject,  // [C, C*D] or null
                    std::vector<float>& mixed,
                    std::vector<float>& write_gate) {
    const auto rbar = reference_rbar(p, residual, gain);
    const size_t CD = p.C * p.D;
    const float inv_c = 1.0f / static_cast<float>(p.C);

    mixed.assign(p.rows * p.D, 0.0f);
    if (w_inject != nullptr) {
        write_gate.assign(p.rows * p.C, 0.0f);
    }

    for (size_t row = 0; row < p.rows; row++) {
        const float* rb = rbar.data() + row * CD;

        std::vector<float> t(p.L, 0.0f);
        for (size_t l = 0; l < p.L; l++) {
            double acc = 0.0;
            for (size_t k = 0; k < CD; k++) {
                acc += static_cast<double>(w_down[l * CD + k]) * rb[k];
            }
            const float x = static_cast<float>(acc) * inv_c;
            t[l] = x * sigmoidf(x);  // SiLU
        }

        for (size_t d = 0; d < p.D; d++) {
            double acc = 0.0;
            for (size_t c = 0; c < p.C; c++) {
                double logit = 0.0;
                for (size_t l = 0; l < p.L; l++) {
                    logit += static_cast<double>(w_up[(c * p.D + d) * p.L + l]) * t[l];
                }
                acc += static_cast<double>(sigmoidf(static_cast<float>(logit))) * rb[c * p.D + d];
            }
            mixed[row * p.D + d] = static_cast<float>(acc) * inv_c;
        }

        if (w_inject != nullptr) {
            for (size_t c = 0; c < p.C; c++) {
                double acc = 0.0;
                for (size_t k = 0; k < CD; k++) {
                    acc += static_cast<double>((*w_inject)[c * CD + k]) * rb[k];
                }
                write_gate[row * p.C + c] = 2.0f * sigmoidf(static_cast<float>(acc) * inv_c);
            }
        }
    }
}

std::vector<ov::float16> to_f16(const std::vector<float>& src) {
    std::vector<ov::float16> dst(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = ov::float16(src[i]);
    }
    return dst;
}

/// tests::random_generator only accepts integer bounds, so values are drawn from
/// [-1, 1] with 1/16 resolution and rescaled/shifted here.
std::vector<float> random_vec(tests::random_generator& rg, size_t count, float scale, float bias = 0.0f) {
    auto v = rg.generate_random_1d<float>(count, -1, 1, 16);
    std::vector<float> out(v.begin(), v.end());
    for (auto& x : out) {
        x = x * scale + bias;
    }
    return out;
}

memory::ptr make_f16(engine& eng, const ov::PartialShape& shape, const std::vector<float>& data) {
    auto mem = eng.allocate_memory(layout{shape, data_types::f16, format::bfyx});
    set_values(mem, to_f16(data));
    return mem;
}

struct gr_read_case {
    size_t rows;
    size_t D;
    size_t L;
    bool produce_write_gate;
};

class gated_residual_read_test : public ::testing::TestWithParam<gr_read_case> {};

TEST_P(gated_residual_read_test, matches_fp32_reference) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    const auto tc = GetParam();
    gr_ref_params p;
    p.rows = tc.rows;
    p.D = tc.D;
    p.L = tc.L;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t CD = p.C * p.D;
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, p.L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * p.L, 0.1f);
    const auto w_inject = random_vec(rg, p.C * CD, 0.1f);

    std::vector<float> ref_mixed;
    std::vector<float> ref_gate;
    reference_read(p, residual, gain, w_down, w_up, tc.produce_write_gate ? &w_inject : nullptr, ref_mixed, ref_gate);

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(p.rows), static_cast<int64_t>(CD)}, residual);
    auto gain_mem = make_f16(engine, {static_cast<int64_t>(CD)}, gain);
    auto wdown_mem = make_f16(engine, {static_cast<int64_t>(p.L), static_cast<int64_t>(CD)}, w_down);
    auto wup_mem = make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(p.L)}, w_up);
    auto winject_mem = make_f16(engine, {static_cast<int64_t>(p.C), static_cast<int64_t>(CD)}, w_inject);

    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(data("gain", gain_mem));
    topology.add(data("w_down", wdown_mem));
    topology.add(data("w_up", wup_mem));

    std::vector<input_info> inputs = {input_info("residual"), input_info("gain"), input_info("w_down"), input_info("w_up")};
    if (tc.produce_write_gate) {
        topology.add(data("w_inject", winject_mem));
        inputs.emplace_back("w_inject");
    }

    gated_residual prim("gr_read", inputs, gated_residual::Mode::READ, p.C, p.D, p.L, p.eps, /*norm_gain_plus_one=*/false, tc.produce_write_gate);
    topology.add(prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);

    auto outputs = network.execute();
    ASSERT_TRUE(outputs.count("gr_read") > 0);

    cldnn::mem_lock<ov::float16> mixed_out(outputs.at("gr_read").get_memory(), get_test_stream());
    ASSERT_EQ(mixed_out.size(), ref_mixed.size());
    for (size_t i = 0; i < ref_mixed.size(); i++) {
        ASSERT_NEAR(static_cast<float>(mixed_out[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at " << i;
    }

    if (tc.produce_write_gate) {
        // Secondary outputs are reached through the instance, not the result map.
        auto inst = network.get_primitive("gr_read");
        cldnn::mem_lock<ov::float16> gate_out(inst->output_memory_ptr(1), get_test_stream());
        ASSERT_EQ(gate_out.size(), ref_gate.size());
        for (size_t i = 0; i < ref_gate.size(); i++) {
            const float v = static_cast<float>(gate_out[i]);
            // 2 * sigmoid(x) lives in (0, 2); a (0, 1) clamp would be a semantic bug.
            ASSERT_GT(v, 0.0f);
            ASSERT_LT(v, 2.0f);
            ASSERT_NEAR(v, ref_gate[i], 2e-2f) << "write gate mismatch at " << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         gated_residual_read_test,
                         ::testing::Values(gr_read_case{1, 64, 16, false},
                                           gr_read_case{1, 64, 16, true},
                                           gr_read_case{7, 128, 32, true},
                                           gr_read_case{16, 32, 16, true}));

TEST(gated_residual_gpu_test, final_mix_matches_read) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    gr_ref_params p;
    p.rows = 4;
    p.D = 64;
    p.L = 16;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t CD = p.C * p.D;
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, p.L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * p.L, 0.1f);

    std::vector<float> ref_mixed;
    std::vector<float> unused;
    reference_read(p, residual, gain, w_down, w_up, nullptr, ref_mixed, unused);

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(p.rows), static_cast<int64_t>(CD)}, residual);

    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(data("gain", make_f16(engine, {static_cast<int64_t>(CD)}, gain)));
    topology.add(data("w_down", make_f16(engine, {static_cast<int64_t>(p.L), static_cast<int64_t>(CD)}, w_down)));
    topology.add(data("w_up", make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(p.L)}, w_up)));
    // FINAL_MIX is the model tail mixer: it replaces the traditional final norm and
    // never produces a write gate.
    topology.add(gated_residual("gr_final",
                                {input_info("residual"), input_info("gain"), input_info("w_down"), input_info("w_up")},
                                gated_residual::Mode::FINAL_MIX,
                                p.C,
                                p.D,
                                p.L,
                                p.eps,
                                false,
                                false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("gr_final").get_memory(), get_test_stream());
    ASSERT_EQ(out.size(), ref_mixed.size());
    for (size_t i = 0; i < ref_mixed.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref_mixed[i], 2e-2f);
    }
}

TEST(gated_residual_gpu_test, write_combines_branches) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    const size_t rows = 5;
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t CD = C * D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto residual = random_vec(rg, rows * CD, 1.0f);
    const auto y = random_vec(rg, rows * D, 1.0f);
    const auto gate = random_vec(rg, rows * C, 1.0f, 1.0f);

    std::vector<float> ref(rows * CD, 0.0f);
    for (size_t row = 0; row < rows; row++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t d = 0; d < D; d++) {
                ref[row * CD + c * D + d] = residual[row * CD + c * D + d] + gate[row * C + c] * y[row * D + d];
            }
        }
    }

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, residual);
    auto y_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(D)}, y);
    auto gate_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(C)}, gate);

    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(input_layout("y", y_mem->get_layout()));
    topology.add(input_layout("gate", gate_mem->get_layout()));
    topology.add(gated_residual("gr_write",
                                {input_info("residual"), input_info("y"), input_info("gate")},
                                gated_residual::Mode::WRITE,
                                C,
                                D,
                                0,
                                1e-6f,
                                false,
                                false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);
    network.set_input_data("y", y_mem);
    network.set_input_data("gate", gate_mem);

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("gr_write").get_memory(), get_test_stream());
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at " << i;
    }
}

TEST(gated_residual_gpu_test, unsupported_branch_count_is_rejected) {
    // C != 4 is outside the first supported configuration; the CM implementation
    // manager must reject it instead of silently producing wrong results.
    gated_residual prim("gr", {input_info("a"), input_info("b"), input_info("c"), input_info("d")}, gated_residual::Mode::READ, 3, 64, 16, 1e-6f, false, false);
    ASSERT_EQ(prim.hc_count, 3u);
    gated_residual supported("gr", {input_info("a"), input_info("b"), input_info("c"), input_info("d")}, gated_residual::Mode::READ, 4, 64, 16, 1e-6f, false, false);
    ASSERT_FALSE(prim == supported);
}

// --- roofline perf harness (token_size sweep) ------------------------------
//
// Companion to the QSA roofline harness in qsa_gpu_test.cpp: builds the production-shaped
// READ/MIX network once per (phase, token_size), times repeated executions with GPU-side
// profiling enabled, and prints one JSON line per measurement. These tests do not check
// output values. Run with:
//   ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter='*DISABLED_roofline*'
namespace roofline {

double mean_ms(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += x;
    }
    return v.empty() ? 0.0 : sum / static_cast<double>(v.size());
}

/// GPU-side "executing" duration, not host wall-clock, so host dispatch overhead is
/// excluded. gated_residual dispatches exactly one CM kernel per mode, so (unlike QSA)
/// `network::get_executed_primitives()` already exposes the real kernel's own event.
double time_primitive_ms(network& net, const std::string& primitive_id, int warmup, int iterations) {
    for (int i = 0; i < warmup; i++) {
        auto outputs = net.execute();
        outputs.at(primitive_id).get_memory();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; i++) {
        auto outputs = net.execute();
        outputs.at(primitive_id).get_memory();
        auto executed = net.get_executed_primitives();
        auto it = executed.find(primitive_id);
        if (it == executed.end() || it->second == nullptr) {
            continue;
        }
        for (const auto& interval : it->second->get_profiling_info()) {
            if (interval.stage == instrumentation::profiling_stage::executing && interval.value != nullptr) {
                samples.push_back(static_cast<double>(interval.value->value().count()) / 1.0e6);
            }
        }
    }
    if (samples.empty()) {
        ADD_FAILURE() << "no 'executing' profiling interval was captured for " << primitive_id;
        return 0.0;
    }
    return mean_ms(samples);
}

void print_measurement(const char* phase, size_t tokens, const char* stage, double ms) {
    std::cout << "ROOFLINE_JSON {\"component\":\"GR\",\"phase\":\"" << phase << "\",\"tokens\":" << tokens
               << ",\"stage\":\"" << stage << "\",\"ms\":" << ms << "}" << std::endl;
}

/// `phase == "prefill"` runs the whole context as one call (`rows = tokens`);
/// `phase == "decode"` runs a single row, matching the QSA harness's query counts. The
/// shape is the production hyper-connection width: hidden_size = 2560, C = 4 branches
/// (D = 640), low-rank width = 320 (design section 3).
void run_point(const char* phase, size_t tokens) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }
    const bool decode = std::string(phase) == "decode";
    const size_t rows = decode ? 1 : tokens;

    gr_ref_params p;
    p.C = kBranches;
    p.D = 640;
    p.L = 320;
    const size_t CD = p.C * p.D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto residual = random_vec(rg, rows * CD, 1.0f);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, p.L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * p.L, 0.1f);

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, residual);

    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(data("gain", make_f16(engine, {static_cast<int64_t>(CD)}, gain)));
    topology.add(data("w_down", make_f16(engine, {static_cast<int64_t>(p.L), static_cast<int64_t>(CD)}, w_down)));
    topology.add(data("w_up", make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(p.L)}, w_up)));
    topology.add(gated_residual("gr_read",
                                {input_info("residual"), input_info("gain"), input_info("w_down"), input_info("w_up")},
                                gated_residual::Mode::READ,
                                p.C,
                                p.D,
                                p.L,
                                p.eps,
                                /*norm_gain_plus_one=*/false,
                                /*produce_write_gate=*/false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::enable_profiling(true));
    network net(engine, topology, config);
    net.set_input_data("residual", residual_mem);

    const double ms = time_primitive_ms(net, "gr_read", /*warmup=*/2, /*iterations=*/100);
    print_measurement(phase, tokens, "READ/MIX", ms);
}

}  // namespace roofline

struct gr_roofline_case {
    size_t tokens;
    bool decode;
};

class gated_residual_roofline_perf : public ::testing::TestWithParam<gr_roofline_case> {};

TEST_P(gated_residual_roofline_perf, measures_cm_kernel_time) {
    const auto p = GetParam();
    roofline::run_point(p.decode ? "decode" : "prefill", p.tokens);
}

INSTANTIATE_TEST_SUITE_P(DISABLED_roofline,
                         gated_residual_roofline_perf,
                         ::testing::Values(gr_roofline_case{1024, false},
                                           gr_roofline_case{1024, true},
                                           gr_roofline_case{2048, false},
                                           gr_roofline_case{2048, true},
                                           gr_roofline_case{4096, false},
                                           gr_roofline_case{4096, true}));

}  // namespace
