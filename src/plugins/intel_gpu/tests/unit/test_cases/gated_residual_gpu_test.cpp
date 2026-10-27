// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the CM gated_residual primitive.
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gated_residual.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "gated_residual_inst.h"
#include "impls/cm/gated_residual/gr_stage_events.hpp"
#include "intel_gpu/op/gated_residual.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/runtime/core.hpp"
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
    bool gain_plus_one = false;
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
                const float g = gain[c * p.D + d] + (p.gain_plus_one ? 1.0f : 0.0f);
                rbar[idx] = residual[idx] * inv * g;
            }
        }
    }
    return rbar;
}

void reference_read(const gr_ref_params& p,
                    const std::vector<float>& residual,
                    const std::vector<float>& gain,
                    const std::vector<float>& w_down,    // [L, C*D]
                    const std::vector<float>& w_up,      // [C*D, L]
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

/// Stage names recorded by the last GatedResidualCMImpl::execute() on this thread.
std::vector<std::string> executed_stages() {
    std::vector<std::string> names;
    for (const auto& stage_event : ov::intel_gpu::cm::gr_debug::take_stage_events()) {
        names.push_back(stage_event.first);
    }
    return names;
}

/// With C = 4, C*D is always a multiple of 64, so only L decides the DPAS READ path. Above
/// 4 rows per EU (and D % 32 == 0) the single-kernel gr_read_dpas.cm replaces the 3-kernel
/// pipeline, whose DOWN_PROJ carries the write gate and UP_PROJ the branch mix.
std::vector<std::string> expected_read_stages(size_t rows, size_t D, size_t L) {
    if (L % 32 != 0) {
        return {"READ"};
    }
    const size_t eu = get_test_engine().get_device_info().execution_units_count;
    if (D % 32 == 0 && rows > 4 * eu) {
        return {"READ_DPAS"};
    }
    return {"RBAR", "DOWN_PROJ", "UP_PROJ"};
}

struct gr_read_case {
    size_t rows;
    size_t D;
    size_t L;
    bool produce_write_gate;
    bool gain_plus_one = false;
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
    p.gain_plus_one = tc.gain_plus_one;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t CD = p.C * p.D;
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    // Effective gain in [0.5, 1.5] either way; stored as w with the (1 + w) convention.
    const auto gain = random_vec(rg, CD, 0.5f, tc.gain_plus_one ? 0.0f : 1.0f);
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

    gated_residual prim("gr_read", inputs, gated_residual::Mode::READ, p.C, p.D, p.L, p.eps, tc.gain_plus_one, tc.produce_write_gate);
    topology.add(prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);

    auto outputs = network.execute();
    ASSERT_TRUE(outputs.count("gr_read") > 0);

    cldnn::mem_lock<ov::float16> mixed_out(outputs.at("gr_read").get_memory(), get_test_stream());
    EXPECT_EQ(executed_stages(), expected_read_stages(p.rows, p.D, p.L));
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
                                           gr_read_case{16, 32, 16, true},
                                           // Below: C*D and L satisfy can_use_dpas_gr_read(), so these route
                                           // through gr_rbar_dpas -> gr_down_proj_dpas[+write gate] ->
                                           // gr_up_proj_dpas[+branch mix] instead of gr_read.cm. Up to 32
                                           // rows the decode tiling is used, up to 4 rows per EU the mid
                                           // (split-K wide) tiling, above it the wide one.
                                           gr_read_case{32, 64, 32, true},           // exact token-tile boundary
                                           gr_read_case{37, 96, 32, false},          // partial last token tile, no write gate
                                           gr_read_case{41, 48, 64, true},           // partial tile; D=48, split-K over 6 K steps
                                           gr_read_case{16, 64, 32, true},           // exactly one N-tile
                                           gr_read_case{48, 64, 32, false},          // three full N-tiles
                                           gr_read_case{3, 32, 32, true},            // narrowest D
                                           gr_read_case{5, 64, 16, true, true},      // scalar path, gain stored as w (1 + w)
                                           gr_read_case{19, 64, 32, true, true},     // DPAS path, gain stored as w (1 + w)
                                           gr_read_case{1, 640, 320, true},          // production width, decode
                                           gr_read_case{17, 640, 320, true},         // production width, prefill with a partial tile
                                           gr_read_case{193, 32, 32, true, true},    // mid tiling
                                           gr_read_case{200, 64, 32, true},          // mid tiling, partial 32-token tile
                                           gr_read_case{300, 48, 64, false},         // mid tiling, D=48 column tiles
                                           gr_read_case{257, 640, 320, true},        // production-like width, mid tiling
                                           gr_read_case{17, 2560, 320, true},        // real Qwen4-Exp width, decode tiling
                                           gr_read_case{100, 2560, 320, true},       // real Qwen4-Exp width, mid tiling
                                           gr_read_case{1000, 64, 32, true},         // wide tiling / single kernel (> 4 rows per EU)
                                           gr_read_case{777, 48, 64, false},         // wide tiling, partial tiles (D % 32 != 0)
                                           gr_read_case{1111, 128, 64, true, true},  // single kernel, partial 32-row tile
                                           gr_read_case{700, 640, 320, false},       // single kernel, production-like width
                                           // L=16 keeps these on gr_read.cm; above 32 rows one work-group owns
                                           // several rows, the last group only partially filled.
                                           gr_read_case{37, 64, 16, true},
                                           gr_read_case{100, 48, 16, false},
                                           gr_read_case{50, 64, 16, true, true},
                                           gr_read_case{65, 2560, 16, true}));  // real width: 4 rows per work-group

class gated_residual_final_mix_test : public ::testing::TestWithParam<gr_read_case> {};

TEST_P(gated_residual_final_mix_test, matches_read) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    gr_ref_params p;
    p.rows = GetParam().rows;
    p.D = GetParam().D;
    p.L = GetParam().L;

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
    EXPECT_EQ(executed_stages(), expected_read_stages(p.rows, p.D, p.L));
    ASSERT_EQ(out.size(), ref_mixed.size());
    for (size_t i = 0; i < ref_mixed.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref_mixed[i], 2e-2f);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         gated_residual_final_mix_test,
                         ::testing::Values(gr_read_case{4, 64, 16, false},       // scalar gr_read
                                           gr_read_case{17, 64, 32, false},      // DPAS pipeline, decode tiling
                                           gr_read_case{250, 64, 32, false},     // DPAS pipeline, mid tiling
                                           gr_read_case{1100, 64, 32, false}));  // DPAS pipeline, wide tiling

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
    topology.add(
        gated_residual("gr_write", {input_info("residual"), input_info("y"), input_info("gate")}, gated_residual::Mode::WRITE, C, D, 0, 1e-6f, false, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);
    network.set_input_data("y", y_mem);
    network.set_input_data("gate", gate_mem);

    auto outputs = network.execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("gr_write").get_memory(), get_test_stream());
    EXPECT_EQ(executed_stages(), std::vector<std::string>{"WRITE"});
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at " << i;
    }
}

// D=48 is a multiple of 16 but not of 32 or 64, so the generator's gr_best_tile() falls
// back to GR_WRITE_TILE=16 - the same per-thread width as the original gr_write.cm. This
// exercises that fallback path (the smoke test above only ever hits GR_WRITE_TILE=64).
TEST(gated_residual_gpu_test, write_combines_branches_narrow_tile) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    const size_t rows = 9;
    const size_t C = kBranches;
    const size_t D = 48;
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
    topology.add(gated_residual("gr_write_narrow",
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
    cldnn::mem_lock<ov::float16> out(outputs.at("gr_write_narrow").get_memory(), get_test_stream());
    EXPECT_EQ(executed_stages(), std::vector<std::string>{"WRITE"});
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
    gated_residual
        supported("gr", {input_info("a"), input_info("b"), input_info("c"), input_info("d")}, gated_residual::Mode::READ, 4, 64, 16, 1e-6f, false, false);
    ASSERT_FALSE(prim == supported);
}

// --- ov::Model / ov::Core level reproduction (direct-construction WRITE mode) ----------
//
// Every other test above goes through the raw cldnn::topology/network API directly, i.e.
// it bypasses ov::Model -> ProgramBuilder -> CreateGatedResidualOp entirely. This test
// instead builds a tiny ov::Model containing a directly-constructed
// ov::intel_gpu::op::GatedResidual(Mode::WRITE) node (no pattern-fusion involved) and
// runs it through ov::Core::compile_model("GPU") + ov::InferRequest::infer(), mirroring
// how a real model reaches this op after GatedResidualWriteFusion, to check whether the
// WRITE-mode CM kernel segfaults when reached via the full plugin pipeline instead of the
// raw topology API above.
TEST(gated_residual_gpu_test, ov_model_write_mode_direct_construction) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    ov::Core core;
    std::vector<std::string> available_devices;
    try {
        available_devices = core.get_available_devices();
    } catch (...) {
    }
    const bool has_gpu = std::any_of(available_devices.begin(), available_devices.end(), [](const std::string& device) {
        return device.rfind("GPU", 0) == 0;
    });
    if (!has_gpu) {
        GTEST_SKIP() << "GPU plugin is not available in ov::Core (unit test binary may lack plugin discovery).";
    }

    const size_t rows = 37;
    const size_t C = kBranches;
    const size_t D = 256;  // matches the real Qwen4-Exp model's hidden_size/C (hc_norm.weight = 1024 = 4*256)
    const size_t CD = C * D;

    auto residual_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{rows, CD});
    auto y_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{rows, D});
    auto gate_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{rows, C});

    auto gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(ov::OutputVector{residual_p, y_p, gate_p},
                                                                 ov::intel_gpu::op::GatedResidual::Mode::WRITE,
                                                                 C,
                                                                 D,
                                                                 /*lowrank=*/0,
                                                                 /*epsilon=*/0.0f,
                                                                 /*norm_gain_plus_one=*/false,
                                                                 /*produce_write_gate=*/false,
                                                                 ov::element::f16);
    auto result = std::make_shared<ov::op::v0::Result>(gr);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{residual_p, y_p, gate_p});

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, "GPU"));

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

    auto infer_request = compiled.create_infer_request();

    auto fill_tensor = [&](size_t idx, const std::vector<float>& data) {
        auto tensor = infer_request.get_input_tensor(idx);
        auto* dst = tensor.data<ov::float16>();
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    fill_tensor(0, residual);
    fill_tensor(1, y);
    fill_tensor(2, gate);

    OV_ASSERT_NO_THROW(infer_request.infer());

    auto out_tensor = infer_request.get_output_tensor(0);
    const auto* out = out_tensor.data<ov::float16>();
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at " << i;
    }
}

TEST(gated_residual_gpu_test, ov_model_read_mode_dynamic_rows_reuses_dispatch_data) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    ov::Core core;
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = 32;
    const size_t CD = C * D;

    auto residual_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD});
    auto w_down_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{L, CD});
    auto w_up_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD, L});
    auto w_inject_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{C, CD});

    auto gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(ov::OutputVector{residual_p, gain_p, w_down_p, w_up_p, w_inject_p},
                                                                 ov::intel_gpu::op::GatedResidual::Mode::READ,
                                                                 C,
                                                                 D,
                                                                 L,
                                                                 1e-6f,
                                                                 false,
                                                                 true,
                                                                 ov::element::f16);
    auto mixed_result = std::make_shared<ov::op::v0::Result>(gr->output(0));
    auto gate_result = std::make_shared<ov::op::v0::Result>(gr->output(1));
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{mixed_result, gate_result}, ov::ParameterVector{residual_p, gain_p, w_down_p, w_up_p, w_inject_p});

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, "GPU"));
    auto infer_request = compiled.create_infer_request();

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * L, 0.1f);
    const auto w_inject = random_vec(rg, C * CD, 0.1f);

    auto fill_tensor = [](ov::Tensor tensor, const std::vector<float>& data) {
        auto* dst = tensor.data<ov::float16>();
        ASSERT_EQ(tensor.get_size(), data.size());
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    fill_tensor(infer_request.get_input_tensor(1), gain);
    fill_tensor(infer_request.get_input_tensor(2), w_down);
    fill_tensor(infer_request.get_input_tensor(3), w_up);
    fill_tensor(infer_request.get_input_tensor(4), w_inject);

    // Walks the dynamic impl through the decode, mid and wide tilings and back again.
    for (const size_t rows : {size_t{5}, size_t{37}, size_t{1}, size_t{300}, size_t{1500}, size_t{2}}) {
        tests::random_generator input_rg;
        input_rg.set_seed(static_cast<uint32_t>(rows));
        const auto residual = random_vec(input_rg, rows * CD, 1.0f);
        ov::Tensor residual_tensor(ov::element::f16, ov::Shape{rows, CD});
        fill_tensor(residual_tensor, residual);
        infer_request.set_tensor(residual_p, residual_tensor);

        gr_ref_params p;
        p.rows = rows;
        p.C = C;
        p.D = D;
        p.L = L;
        std::vector<float> ref_mixed;
        std::vector<float> ref_gate;
        reference_read(p, residual, gain, w_down, w_up, &w_inject, ref_mixed, ref_gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto mixed = infer_request.get_output_tensor(0);
        const auto gate = infer_request.get_output_tensor(1);
        const auto* mixed_data = mixed.data<ov::float16>();
        const auto* gate_data = gate.data<ov::float16>();
        ASSERT_EQ(mixed.get_size(), ref_mixed.size());
        ASSERT_EQ(gate.get_size(), ref_gate.size());
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(mixed_data[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
        for (size_t i = 0; i < ref_gate.size(); i++) {
            ASSERT_NEAR(static_cast<float>(gate_data[i]), ref_gate[i], 2e-2f) << "write gate mismatch at rows=" << rows << " index=" << i;
        }
    }
}

// Companion to ov_model_read_mode_dynamic_rows_reuses_dispatch_data above, but for the
// scalar (non-DPAS) gr_read.cm kernel: L=16 fails can_use_dpas_gr_read()'s "lowrank must
// be a multiple of 32" check, so GatedResidualCMImpl selects the single-kernel gr_read
// stage instead of the rbar/down/up dpas pipeline. Exercises the SAME
// dispatch-data bug (the unconditional `if (params.is_dynamic()) return;` guard) in the
// one generator the DPAS-path test above never reaches.
TEST(gated_residual_gpu_test, ov_model_read_mode_dynamic_rows_scalar_kernel) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    ov::Core core;
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = 16;  // not a multiple of 32 -> forces the scalar gr_read.cm path
    const size_t CD = C * D;

    auto residual_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD});
    auto w_down_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{L, CD});
    auto w_up_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD, L});

    auto gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(ov::OutputVector{residual_p, gain_p, w_down_p, w_up_p},
                                                                 ov::intel_gpu::op::GatedResidual::Mode::READ,
                                                                 C,
                                                                 D,
                                                                 L,
                                                                 1e-6f,
                                                                 false,
                                                                 false,
                                                                 ov::element::f16);
    auto result = std::make_shared<ov::op::v0::Result>(gr);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{residual_p, gain_p, w_down_p, w_up_p});

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, "GPU"));
    auto infer_request = compiled.create_infer_request();

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * L, 0.1f);

    auto fill_tensor = [](ov::Tensor tensor, const std::vector<float>& data) {
        auto* dst = tensor.data<ov::float16>();
        ASSERT_EQ(tensor.get_size(), data.size());
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    fill_tensor(infer_request.get_input_tensor(1), gain);
    fill_tensor(infer_request.get_input_tensor(2), w_down);
    fill_tensor(infer_request.get_input_tensor(3), w_up);

    for (const size_t rows : {size_t{3}, size_t{41}, size_t{1}, size_t{100}, size_t{2}}) {
        tests::random_generator input_rg;
        input_rg.set_seed(static_cast<uint32_t>(rows) + 1000);
        const auto residual = random_vec(input_rg, rows * CD, 1.0f);
        ov::Tensor residual_tensor(ov::element::f16, ov::Shape{rows, CD});
        fill_tensor(residual_tensor, residual);
        infer_request.set_tensor(residual_p, residual_tensor);

        gr_ref_params p;
        p.rows = rows;
        p.C = C;
        p.D = D;
        p.L = L;
        std::vector<float> ref_mixed;
        std::vector<float> unused_gate;
        reference_read(p, residual, gain, w_down, w_up, nullptr, ref_mixed, unused_gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto mixed = infer_request.get_output_tensor(0);
        const auto* mixed_data = mixed.data<ov::float16>();
        ASSERT_EQ(mixed.get_size(), ref_mixed.size());
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(mixed_data[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
    }
}

// Companion covering GatedResidualWriteGenerator's dispatch-data path (the gr_write.cm
// stage, used both for WRITE and reached whenever `desc->mode == Mode::WRITE`) under a
// dynamic row count on the SAME compiled model / infer request.
TEST(gated_residual_gpu_test, ov_model_write_mode_dynamic_rows_reuses_dispatch_data) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    ov::Core core;
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t CD = C * D;

    auto residual_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(CD)});
    auto y_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(D)});
    auto gate_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(C)});

    auto gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(ov::OutputVector{residual_p, y_p, gate_p},
                                                                 ov::intel_gpu::op::GatedResidual::Mode::WRITE,
                                                                 C,
                                                                 D,
                                                                 /*lowrank=*/0,
                                                                 /*epsilon=*/0.0f,
                                                                 /*norm_gain_plus_one=*/false,
                                                                 /*produce_write_gate=*/false,
                                                                 ov::element::f16);
    auto result = std::make_shared<ov::op::v0::Result>(gr);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{residual_p, y_p, gate_p});

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, "GPU"));
    auto infer_request = compiled.create_infer_request();

    auto fill_tensor = [](ov::Tensor tensor, const std::vector<float>& data) {
        auto* dst = tensor.data<ov::float16>();
        ASSERT_EQ(tensor.get_size(), data.size());
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };

    for (const size_t rows : {size_t{5}, size_t{37}, size_t{1}}) {
        tests::random_generator rg;
        rg.set_seed(static_cast<uint32_t>(rows) + 2000);
        const auto residual = random_vec(rg, rows * CD, 1.0f);
        const auto y = random_vec(rg, rows * D, 1.0f);
        const auto gate = random_vec(rg, rows * C, 1.0f, 1.0f);

        ov::Tensor residual_tensor(ov::element::f16, ov::Shape{rows, CD});
        ov::Tensor y_tensor(ov::element::f16, ov::Shape{rows, D});
        ov::Tensor gate_tensor(ov::element::f16, ov::Shape{rows, C});
        fill_tensor(residual_tensor, residual);
        fill_tensor(y_tensor, y);
        fill_tensor(gate_tensor, gate);
        infer_request.set_tensor(residual_p, residual_tensor);
        infer_request.set_tensor(y_p, y_tensor);
        infer_request.set_tensor(gate_p, gate_tensor);

        std::vector<float> ref(rows * CD, 0.0f);
        for (size_t row = 0; row < rows; row++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t d = 0; d < D; d++) {
                    ref[row * CD + c * D + d] = residual[row * CD + c * D + d] + gate[row * C + c] * y[row * D + d];
                }
            }
        }

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto out_tensor = infer_request.get_output_tensor(0);
        const auto* out = out_tensor.data<ov::float16>();
        ASSERT_EQ(out_tensor.get_size(), ref.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at rows=" << rows << " index=" << i;
        }
    }
}

// --- roofline perf harness (token_size sweep) ------------------------------
//
// Roofline harness: builds the production-shaped
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

/// Extracts the GPU-side "executing" duration of `ev`, in milliseconds, or -1.0 if `ev`
/// is null or has no such interval. GPU-side timing is used instead of host wall-clock so
/// host dispatch overhead (which the roofline model does not account for) is excluded.
double executing_ms(const event::ptr& ev) {
    if (ev == nullptr) {
        return -1.0;
    }
    for (const auto& interval : ev->get_profiling_info()) {
        if (interval.stage == instrumentation::profiling_stage::executing && interval.value != nullptr) {
            return static_cast<double>(interval.value->value().count()) / 1.0e6;
        }
    }
    return -1.0;
}

/// GPU-side total duration of one gated_residual call, summed across every CM kernel
/// dispatch recorded via gr_stage_events.hpp. The systolic READ/FINAL_MIX path runs 3
/// kernels (RBAR, down-proj, up-proj) behind one clDNN primitive, so
/// `network::get_executed_primitives()` alone would only expose the last of them (see
/// gr_stage_events.hpp). `per_stage`, if
/// given, receives each stage's mean time in dispatch order.
double time_primitive_ms(network& net,
                         const std::string& primitive_id,
                         int warmup,
                         int iterations,
                         std::vector<std::pair<std::string, double>>* per_stage = nullptr) {
    for (int i = 0; i < warmup; i++) {
        auto outputs = net.execute();
        outputs.at(primitive_id).get_memory();
        for (auto& stage_event : ov::intel_gpu::cm::gr_debug::take_stage_events()) {
            if (stage_event.second != nullptr) {
                stage_event.second->wait();
            }
        }
    }
    // Iterations are enqueued back-to-back and only synchronized at the end: a host wait per
    // iteration leaves the GPU idle long enough for DVFS to drop the clock (650 vs 2900 MHz
    // observed on B580), which made the per-kernel times drift by >2x within one run.
    std::vector<std::vector<std::pair<std::string, event::ptr>>> iteration_events;
    iteration_events.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; i++) {
        net.execute();
        iteration_events.push_back(ov::intel_gpu::cm::gr_debug::take_stage_events());
    }
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iterations));
    std::vector<std::pair<std::string, double>> stage_sums;
    for (const auto& stage_events : iteration_events) {
        double total = 0.0;
        bool any = false;
        size_t idx = 0;
        for (const auto& stage_event : stage_events) {
            if (stage_event.second != nullptr) {
                stage_event.second->wait();
            }
            const double ms = executing_ms(stage_event.second);
            if (ms >= 0.0) {
                total += ms;
                any = true;
                if (stage_sums.size() <= idx) {
                    stage_sums.emplace_back(stage_event.first, 0.0);
                }
                stage_sums[idx].second += ms;
            }
            idx++;
        }
        if (any) {
            samples.push_back(total);
        }
    }
    if (samples.empty()) {
        ADD_FAILURE() << "no 'executing' profiling interval was captured for " << primitive_id;
        return 0.0;
    }
    if (per_stage != nullptr) {
        for (auto& s : stage_sums) {
            per_stage->emplace_back(s.first, s.second / static_cast<double>(samples.size()));
        }
    }
    return mean_ms(samples);
}

void print_measurement(const char* phase, size_t tokens, const char* stage, double ms) {
    std::cout << "ROOFLINE_JSON {\"component\":\"GR\",\"phase\":\"" << phase << "\",\"tokens\":" << tokens << ",\"stage\":\"" << stage << "\",\"ms\":" << ms
              << "}" << std::endl;
}

/// `phase == "prefill"` runs the whole context as one call (`rows = tokens`);
/// `phase == "decode"` runs a single row, matching the QSA harness's query counts. The
/// shape is the production hyper-connection width: hidden_size = 2560, C = 4 branches
/// (D = 640), low-rank width = 320.
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
    // Device-resident like real activations; the default lockable allocation is usm_shared
    // on dGPU, which would make RBAR measure PCIe reads instead of the kernel.
    auto residual_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, data_types::f16, format::bfyx},
                                               allocation_type::usm_device);
    const auto residual_f16 = to_f16(residual);
    residual_mem->copy_from(get_test_stream(), residual_f16.data(), true);

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

    std::vector<std::pair<std::string, double>> per_stage;
    const double ms = time_primitive_ms(net, "gr_read", /*warmup=*/20, /*iterations=*/100, &per_stage);
    for (const auto& s : per_stage) {
        print_measurement(phase, tokens, s.first.c_str(), s.second);
    }
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
                                           gr_roofline_case{4096, true},
                                           gr_roofline_case{8192, false}));

// Equivalence of the decomposed source graph and the fused kernel: the decomposed READ/MIX
// model is compiled through the normal plugin pipeline (which must fuse it into a single
// gated_residual node) and compared against the FP32 reference of the *decomposed* math.
TEST(gated_residual_gpu_test, decomposed_read_graph_fuses_and_matches_reference) {
    if (!cm_gr_available()) {
        GTEST_SKIP() << "CM JIT / systolic GPU is required for gated_residual";
    }

    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = 32;
    const size_t CD = C * D;
    const float eps = 1e-5f;
    const auto et = ov::element::f16;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto gain_w = random_vec(rg, CD, 0.25f);
    const auto w_down = random_vec(rg, L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * L, 0.1f);

    auto i64 = [](std::vector<int64_t> v) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
    };
    auto scalar = [&](float v) {
        return ov::op::v0::Constant::create(et, ov::Shape{}, {v});
    };
    auto f16_param = [&](const ov::Shape& shape) {
        return std::make_shared<ov::op::v0::Parameter>(et, shape);
    };

    // Gain and weights are model inputs so ConstantFolding cannot fold the 1/C scale into W_down
    // (real checkpoints keep their compressed weights opaque in the same way).
    auto residual_p = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain_p = f16_param(ov::Shape{CD});
    auto w_down_p = f16_param(ov::Shape{L, CD});
    auto w_up_p = f16_param(ov::Shape{CD, L});
    auto r3 = std::make_shared<ov::op::v1::Reshape>(residual_p, i64({-1, static_cast<int64_t>(C), static_cast<int64_t>(D)}), false);
    auto sq = std::make_shared<ov::op::v1::Power>(r3, scalar(2.0f));
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(sq, i64({-1}), true);
    auto rms = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(mean, scalar(eps)));
    auto inv_rms = std::make_shared<ov::op::v1::Divide>(scalar(1.0f), rms);
    auto normalized = std::make_shared<ov::op::v1::Multiply>(r3, inv_rms);
    auto normalized_flat = std::make_shared<ov::op::v1::Reshape>(normalized, i64({-1, static_cast<int64_t>(CD)}), false);
    auto gain = std::make_shared<ov::op::v1::Add>(gain_p, scalar(1.0f));
    auto rbar = std::make_shared<ov::op::v1::Multiply>(normalized_flat, gain);

    auto low = std::make_shared<ov::op::v0::MatMul>(rbar, w_down_p, false, true);
    auto act = std::make_shared<ov::op::v4::Swish>(std::make_shared<ov::op::v1::Multiply>(low, scalar(1.0f / static_cast<float>(C))));
    auto up = std::make_shared<ov::op::v0::MatMul>(act, w_up_p, false, true);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Sigmoid>(up), rbar);
    auto weighted3 = std::make_shared<ov::op::v1::Reshape>(weighted, i64({-1, static_cast<int64_t>(C), static_cast<int64_t>(D)}), false);
    auto mixed =
        std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v1::ReduceSum>(weighted3, i64({1}), false), scalar(1.0f / static_cast<float>(C)));
    auto model = std::make_shared<ov::Model>(ov::OutputVector{mixed}, ov::ParameterVector{residual_p, gain_p, w_down_p, w_up_p});

    ov::Core core;
    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, "GPU"));
    size_t fused_nodes = 0;
    std::string runtime_layers;
    for (const auto& op : compiled.get_runtime_model()->get_ops()) {
        const auto it = op->get_rt_info().find("layerType");
        const auto layer_type = it != op->get_rt_info().end() ? it->second.as<std::string>() : std::string("?");
        // Fused nodes whose primitive id differs from the op name are recorded with the primitive
        // type string ("gated_residual") instead of the op type name ("GatedResidual").
        fused_nodes += layer_type == "GatedResidual" || layer_type == "gated_residual";
        runtime_layers += layer_type + " ";
    }
    EXPECT_EQ(fused_nodes, 1u) << "decomposed READ graph was not fused into gated_residual; runtime layers: " << runtime_layers;

    auto infer_request = compiled.create_infer_request();
    auto fill = [](ov::Tensor tensor, const std::vector<float>& data) {
        auto* dst = tensor.data<ov::float16>();
        ASSERT_EQ(tensor.get_size(), data.size());
        for (size_t i = 0; i < data.size(); i++) {
            dst[i] = ov::float16(data[i]);
        }
    };
    fill(infer_request.get_input_tensor(1), gain_w);
    fill(infer_request.get_input_tensor(2), w_down);
    fill(infer_request.get_input_tensor(3), w_up);
    for (const size_t rows : {size_t{3}, size_t{64}, size_t{700}}) {
        tests::random_generator input_rg;
        input_rg.set_seed(static_cast<uint32_t>(rows));
        const auto residual = random_vec(input_rg, rows * CD, 1.0f);
        ov::Tensor residual_tensor(et, ov::Shape{rows, CD});
        auto* dst = residual_tensor.data<ov::float16>();
        for (size_t i = 0; i < residual.size(); i++) {
            dst[i] = ov::float16(residual[i]);
        }
        infer_request.set_tensor(residual_p, residual_tensor);

        // The model applies the gain as `1 + w`, so the reference sees the folded gain.
        std::vector<float> gain_folded(gain_w);
        for (auto& g : gain_folded) {
            g += 1.0f;
        }
        gr_ref_params p;
        p.rows = rows;
        p.C = C;
        p.D = D;
        p.L = L;
        p.eps = eps;
        std::vector<float> ref_mixed;
        std::vector<float> ref_gate;
        reference_read(p, residual, gain_folded, w_down, w_up, nullptr, ref_mixed, ref_gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto out_tensor = infer_request.get_output_tensor(0);
        const auto* out = out_tensor.data<ov::float16>();
        ASSERT_EQ(out_tensor.get_size(), ref_mixed.size());
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(out[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
    }
}

}  // namespace
