// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the gated_residual primitive.
//
// Covered:
//  - READ / MIX: per-branch, per-channel comparison with an FP32 reference, including
//    the branch-wise grouped RMSNorm, the 1/C inside SiLU(W_down * Rbar / C) and the
//    branch mean;
//  - the optional write gate produced by READ: 2 * sigmoid(W_inject * Rbar / C), and
//    that it stays inside (0, 2);
//  - WRITE / COMBINE: R'_c = R_c + S_c * Y for both in-place and out-of-place buffers;
//  - FINAL_MIX: the model-tail mixer that replaces the usual final norm;
//  - negative case: an unsupported branch count must not select any implementation.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gated_residual.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/runtime/profiling.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gated_residual_inst.h"
#include "impls/ocl_v2/gated_residual/gated_residual_stage_events.hpp"
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

bool gr_available() {
    const auto& info = get_test_engine().get_device_info();
    return info.supports_intel_subgroups && info.supports_intel_subgroups_short;
}

std::vector<std::pair<std::string, cldnn::event::ptr>> take_stage_events() {
    return ov::intel_gpu::ocl::gr_debug::take_stage_events();
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

/// Stage names recorded by the last GatedResidualImpl::execute() on this thread.
std::vector<std::string> executed_stages() {
    std::vector<std::string> names;
    for (const auto& stage_event : take_stage_events()) {
        names.push_back(stage_event.first);
    }
    return names;
}

/// With C = 4, C*D is always a multiple of 64, so only L decides the DPAS READ path. Above
/// 4 rows per EU (and D % 32 == 0) the single-kernel gr_read_dpas.cl replaces the 3-kernel
/// pipeline, whose DOWN_PROJ carries the write gate and UP_PROJ the branch mix. Devices without Xe2+ systolic
/// arrays always run the scalar gr_read.cl.
std::vector<std::string> expected_read_stages(size_t rows, size_t D, size_t L) {
    const auto& info = get_test_engine().get_device_info();
    const bool xmx = info.arch >= cldnn::gpu_arch::xe2 && info.supports_immad;
    if (!xmx || L % 32 != 0) {
        return {"READ"};
    }
    const size_t eu = info.execution_units_count;
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
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
                                           // Below: C*D and L allow the dpas paths, so these route
                                           // through RBAR -> DOWN_PROJ[+write gate] ->
                                           // UP_PROJ[+branch mix] instead of gr_read.cl. Up to 32
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
                                           // L=16 keeps these on gr_read.cl; above 32 rows one work-group owns
                                           // several rows, the last group only partially filled.
                                           gr_read_case{37, 64, 16, true},
                                           gr_read_case{100, 48, 16, false},
                                           gr_read_case{50, 64, 16, true, true},
                                           gr_read_case{65, 2560, 16, true}));  // real width: 4 rows per work-group

class gated_residual_final_mix_test : public ::testing::TestWithParam<gr_read_case> {};

TEST_P(gated_residual_final_mix_test, matches_read) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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

// --- compressed weights: u4 (group 32) / u8 (per-channel), asymmetric ------------------------------------------------------

bool gr_compression_available() {
    const auto& info = get_test_engine().get_device_info();
    return ov::intel_gpu::op::GatedResidual::is_compression_arch_supported(info);
}

/// One quantized [N, K] weight matrix with its [G, N] scale / zp and the exact f16 tensor the decompression graph produces.
struct quant_matrix {
    size_t N = 0;
    size_t K = 0;
    size_t G = 0;
    bool u4 = false;
    std::vector<uint8_t> q;      // [N * K]
    std::vector<uint8_t> zp;     // [G * N]
    std::vector<float> scale;    // [G * N], f16 values
    std::vector<float> dequant;  // [N * K] = half(half(q - zp) * scale)
};

quant_matrix make_quant_matrix(std::mt19937& gen, size_t N, size_t K, bool u4) {
    quant_matrix m;
    m.N = N;
    m.K = K;
    m.u4 = u4;
    m.G = u4 ? K / 32 : 1;
    const size_t gs = K / m.G;
    m.q.resize(N * K);
    m.zp.resize(m.G * N);
    m.scale.resize(m.G * N);
    m.dequant.resize(N * K);
    for (size_t i = 0; i < m.G * N; i++) {
        m.zp[i] = u4 ? static_cast<uint8_t>(gen() % 16) : static_cast<uint8_t>(100 + gen() % 60);
        m.scale[i] = static_cast<float>(ov::float16(static_cast<float>(1 + gen() % 8) / (u4 ? 512.0f : 2048.0f)));
    }
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            const size_t g = k / gs;
            const uint8_t q = static_cast<uint8_t>(gen() % (u4 ? 16 : 256));
            m.q[n * K + k] = q;
            const float diff = static_cast<float>(ov::float16(static_cast<float>(static_cast<int>(q) - static_cast<int>(m.zp[g * N + n]))));
            m.dequant[n * K + k] = static_cast<float>(ov::float16(diff * m.scale[g * N + n]));
        }
    }
    return m;
}

memory::ptr make_quant_weights(engine& eng, const quant_matrix& m) {
    auto mem = eng.allocate_memory(
        layout{ov::PartialShape{static_cast<int64_t>(m.N), static_cast<int64_t>(m.K)}, m.u4 ? data_types::u4 : data_types::u8, format::bfyx});
    cldnn::mem_lock<uint8_t> lock(mem, get_test_stream());
    if (m.u4) {
        for (size_t i = 0; i < m.q.size(); i += 2) {
            lock[i / 2] = static_cast<uint8_t>(m.q[i] | (m.q[i + 1] << 4));  // low nibble = even k
        }
    } else {
        std::copy(m.q.begin(), m.q.end(), lock.begin());
    }
    return mem;
}

memory::ptr make_quant_zp(engine& eng, const quant_matrix& m) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(m.G), static_cast<int64_t>(m.N)}, data_types::u8, format::bfyx});
    cldnn::mem_lock<uint8_t> lock(mem, get_test_stream());
    std::copy(m.zp.begin(), m.zp.end(), lock.begin());
    return mem;
}

struct gr_run_result {
    std::vector<ov::float16> mixed;
    std::vector<std::string> stages;
};

/// Runs a READ / FINAL_MIX gated_residual whose weight inputs (from index 2 on) are the given memories.
gr_run_result run_gr_read(const gr_ref_params& p,
                          gated_residual::Mode mode,
                          const std::vector<float>& residual,
                          const std::vector<float>& gain,
                          const std::vector<memory::ptr>& weights) {
    auto& engine = get_test_engine();
    const size_t CD = p.C * p.D;
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(p.rows), static_cast<int64_t>(CD)}, residual);
    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(data("gain", make_f16(engine, {static_cast<int64_t>(CD)}, gain)));
    std::vector<input_info> inputs = {input_info("residual"), input_info("gain")};
    for (size_t i = 0; i < weights.size(); i++) {
        const std::string id = "w" + std::to_string(i);
        topology.add(data(id, weights[i]));
        inputs.emplace_back(id);
    }
    topology.add(gated_residual("gr", inputs, mode, p.C, p.D, p.L, p.eps, p.gain_plus_one, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("residual", residual_mem);
    auto outputs = network.execute();
    gr_run_result result;
    cldnn::mem_lock<ov::float16> out(outputs.at("gr").get_memory(), get_test_stream());
    result.mixed.assign(out.begin(), out.end());
    result.stages = executed_stages();
    return result;
}

struct gr_quant_case {
    size_t rows;
    size_t D;
    size_t L;
    bool u4;
    bool final_mix = false;
};

size_t scratch_limit_bytes() {
    const char* env = std::getenv("OV_GPU_GR_Q_SCRATCH_LIMIT_MB");
    return (env != nullptr ? static_cast<size_t>(std::strtoull(env, nullptr, 10)) : size_t{256}) << 20;
}

size_t pd_min_rows() {
    const char* env = std::getenv("OV_GPU_GR_PD_MIN_ROWS");
    return env != nullptr ? static_cast<size_t>(std::strtoull(env, nullptr, 10))
                          : 5 * static_cast<size_t>(get_test_engine().get_device_info().execution_units_count);
}

class gated_residual_quant_test : public ::testing::TestWithParam<gr_quant_case> {};

// The kernels dequantize in registers with the arithmetic of the decompression graph, so the result must equal the f16 path
// fed with the decompressed weights bit for bit (same tiling, same accumulation order), and match the fp32 reference.
TEST_P(gated_residual_quant_test, matches_decompressed_f16_path) {
    if (!gr_compression_available()) {
        GTEST_SKIP() << "compressed GatedResidual weights need an Xe2+ systolic GPU";
    }
    const auto tc = GetParam();
    gr_ref_params p;
    p.rows = tc.rows;
    p.D = tc.D;
    p.L = tc.L;
    p.gain_plus_one = true;
    const size_t CD = p.C * p.D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    std::mt19937 gen(1234 + static_cast<uint32_t>(tc.rows * 7 + tc.D + tc.L + (tc.u4 ? 1 : 0)));
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    const auto gain = random_vec(rg, CD, 0.5f);
    const auto wd = make_quant_matrix(gen, p.L, CD, tc.u4);
    const auto wu = make_quant_matrix(gen, CD, p.L, tc.u4);

    auto& engine = get_test_engine();
    const auto mode = tc.final_mix ? gated_residual::Mode::FINAL_MIX : gated_residual::Mode::READ;
    const auto f16_run = run_gr_read(p,
                                     mode,
                                     residual,
                                     gain,
                                     {make_f16(engine, {static_cast<int64_t>(p.L), static_cast<int64_t>(CD)}, wd.dequant),
                                      make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(p.L)}, wu.dequant)});
    const auto q_run = run_gr_read(p,
                                   mode,
                                   residual,
                                   gain,
                                   {make_quant_weights(engine, wd),
                                    make_quant_weights(engine, wu),
                                    make_f16(engine, {static_cast<int64_t>(wd.G), static_cast<int64_t>(p.L)}, wd.scale),
                                    make_f16(engine, {static_cast<int64_t>(wu.G), static_cast<int64_t>(CD)}, wu.scale),
                                    make_quant_zp(engine, wd),
                                    make_quant_zp(engine, wu)});

    // compressed weights always use the 3-dispatch pipeline unless its Rbar scratch exceeds the limit
    const auto expected = expected_read_stages(p.rows, p.D, p.L);
    EXPECT_EQ(f16_run.stages, expected);
    const bool single_q = p.D % 32 == 0 && p.rows * p.C * p.D * 2 > scratch_limit_bytes();
    std::vector<std::string> expected_q = single_q ? std::vector<std::string>{"READ_DPAS"} : std::vector<std::string>{"RBAR", "DOWN_PROJ", "UP_PROJ"};
    if (p.rows > pd_min_rows()) {
        // above the predequantization threshold: both matrices are dequantized into f16 scratch and the f16 kernels (f16 routing) run on them
        expected_q = expected;
        expected_q.insert(expected_q.begin(), {"DEQUANT", "DEQUANT"});
    }
    EXPECT_EQ(q_run.stages, expected_q);

    std::vector<float> ref_mixed;
    std::vector<float> unused;
    reference_read(p, residual, gain, wd.dequant, wu.dequant, nullptr, ref_mixed, unused);
    ASSERT_EQ(q_run.mixed.size(), ref_mixed.size());
    ASSERT_EQ(f16_run.mixed.size(), ref_mixed.size());
    // The compressed kernels share work differently (split-K factors, single kernel vs pipeline), so only the accumulation
    // order differs from the f16 path fed with the decompressed weights: the results agree to a few f16 ulps.
    size_t bit_diffs = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref_mixed.size(); i++) {
        ASSERT_NEAR(static_cast<float>(q_run.mixed[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at " << i;
        bit_diffs += q_run.mixed[i].to_bits() != f16_run.mixed[i].to_bits() ? 1 : 0;
        max_diff = std::max(max_diff, std::abs(static_cast<float>(q_run.mixed[i]) - static_cast<float>(f16_run.mixed[i])));
    }
    std::cout << "GR_QUANT_ACC rows=" << p.rows << " D=" << p.D << " L=" << p.L << " " << (tc.u4 ? "u4" : "u8") << " differing_elems=" << bit_diffs << "/"
              << ref_mixed.size() << " max_abs_diff_vs_f16=" << max_diff << std::endl;
    EXPECT_LE(max_diff, 4e-3f) << "compressed and decompressed-f16 outputs differ";
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         gated_residual_quant_test,
                         ::testing::Values(gr_quant_case{1, 64, 32, true},  // decode, one K32 group of W_up
                                           gr_quant_case{1, 64, 32, false},
                                           gr_quant_case{7, 128, 32, true},
                                           gr_quant_case{7, 128, 32, false},
                                           gr_quant_case{32, 96, 32, true},  // D % 64 != 0: narrow raw loads
                                           gr_quant_case{37, 96, 64, false},
                                           gr_quant_case{33, 128, 64, true},  // mid tiling
                                           gr_quant_case{200, 64, 32, false},
                                           gr_quant_case{17, 640, 320, true},  // production width, decode
                                           gr_quant_case{17, 640, 320, false},
                                           gr_quant_case{100, 640, 320, true},
                                           gr_quant_case{257, 640, 320, false},     // mid tiling
                                           gr_quant_case{250, 64, 32, true, true},  // FINAL_MIX
                                           gr_quant_case{1000, 64, 32, true},       // single kernel
                                           gr_quant_case{1000, 64, 32, false},
                                           gr_quant_case{1111, 128, 64, true},  // partial 32-row tile
                                           gr_quant_case{700, 640, 320, true},
                                           gr_quant_case{700, 640, 320, false},
                                           gr_quant_case{17, 2560, 320, true},  // real width
                                           gr_quant_case{17, 2560, 320, false},
                                           gr_quant_case{100, 2560, 320, true},
                                           gr_quant_case{2000, 2560, 320, true},
                                           gr_quant_case{2000, 2560, 320, false}));

TEST(gated_residual_gpu_test, write_combines_branches) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
// back to GR_WRITE_TILE=16. This
// exercises that fallback path (the smoke test above only ever hits GR_WRITE_TILE=64).
TEST(gated_residual_gpu_test, write_combines_branches_narrow_tile) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
    // C != 4 is outside the first supported configuration; the implementation
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
// WRITE-mode kernel segfaults when reached via the full plugin pipeline instead of the
// raw topology API above.
TEST(gated_residual_gpu_test, ov_model_write_mode_direct_construction) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
// scalar (non-DPAS) gr_read.cl kernel: L=16 fails the dpas path's "lowrank must
// be a multiple of 32" check, so GatedResidualImpl selects the single-kernel gr_read
// stage instead of the rbar/down/up dpas pipeline. Exercises the SAME
// dispatch-data bug (the unconditional `if (params.is_dynamic()) return;` guard) in the
// one generator the DPAS-path test above never reaches.
TEST(gated_residual_gpu_test, ov_model_read_mode_dynamic_rows_scalar_kernel) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    ov::Core core;
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = 16;  // not a multiple of 32 -> forces the scalar gr_read.cl path
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

// Companion covering GrWriteGenerator's dispatch-data path (the gr_write.cl
// stage, used both for WRITE and reached whenever `desc->mode == Mode::WRITE`) under a
// dynamic row count on the SAME compiled model / infer request.
TEST(gated_residual_gpu_test, ov_model_write_mode_dynamic_rows_reuses_dispatch_data) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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

/// GPU-side total duration of one gated_residual call, summed across every kernel
/// dispatch recorded via gated_residual_stage_events.hpp. The systolic READ/FINAL_MIX path runs 3
/// kernels (RBAR, down-proj, up-proj) behind one clDNN primitive, so
/// `network::get_executed_primitives()` alone would only expose the last of them (see
/// gated_residual_stage_events.hpp). `per_stage`, if
/// given, receives each stage's mean time in dispatch order.
double time_primitive_ms(network& net,
                         const std::string& primitive_id,
                         int warmup,
                         int iterations,
                         std::vector<std::pair<std::string, double>>* per_stage = nullptr) {
    for (int i = 0; i < warmup; i++) {
        auto outputs = net.execute();
        outputs.at(primitive_id).get_memory();
        for (auto& stage_event : take_stage_events()) {
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
        iteration_events.push_back(take_stage_events());
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
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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

// --- f16 vs u4 vs u8 weights: same-process interleaved A/B (DISABLED; run with --gtest_also_run_disabled_tests) -------------
//
// Every configuration owns NL independent layers (own weight buffers) that are used round-robin, so the weights stream from DRAM
// like in a real model instead of staying resident in L2. Output: one GR_AB json line per (rows, D, weight type, stage).
namespace quant_ab {

struct ab_config {
    const char* name;
    int wtype;  // 0 f16, 1 u4, 2 u8
    std::vector<std::unique_ptr<network>> layers;
};

/// Weights live in device memory like in a real model; the default lockable allocation is host / shared USM.
memory::ptr to_device(engine& eng, const memory::ptr& host) {
    auto dev = eng.allocate_memory(host->get_layout(), allocation_type::usm_device);
    cldnn::mem_lock<uint8_t> lock(host, get_test_stream());
    dev->copy_from(get_test_stream(), lock.data(), true);
    return dev;
}

double measure_cycle(ab_config& cfg, int warmup, int iterations, std::vector<std::pair<std::string, double>>& per_stage) {
    const size_t nl = cfg.layers.size();
    for (int i = 0; i < warmup; i++) {
        auto& net = *cfg.layers[static_cast<size_t>(i) % nl];
        net.execute();
        for (auto& e : take_stage_events()) {
            if (e.second != nullptr) {
                e.second->wait();
            }
        }
    }
    std::vector<std::vector<std::pair<std::string, event::ptr>>> events;
    events.reserve(static_cast<size_t>(iterations));
    for (int i = 0; i < iterations; i++) {
        cfg.layers[static_cast<size_t>(i) % nl]->execute();
        events.push_back(take_stage_events());
    }
    std::vector<std::pair<std::string, double>> sums;
    size_t counted = 0;
    for (const auto& stage_events : events) {
        size_t idx = 0;
        for (const auto& e : stage_events) {
            if (e.second != nullptr) {
                e.second->wait();
            }
            const double ms = roofline::executing_ms(e.second);
            if (ms >= 0.0) {
                if (sums.size() <= idx) {
                    sums.emplace_back(e.first, 0.0);
                }
                sums[idx].second += ms;
            }
            idx++;
        }
        counted++;
    }
    per_stage.clear();
    double total = 0.0;
    for (auto& s : sums) {
        per_stage.emplace_back(s.first, s.second / static_cast<double>(counted));
        total += s.second / static_cast<double>(counted);
    }
    return total;
}

}  // namespace quant_ab

TEST(gated_residual_quant_ab, DISABLED_weights) {
    if (!gr_compression_available()) {
        GTEST_SKIP() << "compressed GatedResidual weights need an Xe2+ systolic GPU";
    }
    auto& engine = get_test_engine();
    const size_t eu = engine.get_device_info().execution_units_count;
    std::vector<std::pair<size_t, size_t>> points = {{1, 640},
                                                     {32, 640},
                                                     {128, 640},
                                                     {1024, 640},
                                                     {4096, 640},
                                                     {1, 2560},
                                                     {8, 2560},
                                                     {32, 2560},
                                                     {33, 2560},
                                                     {128, 2560},
                                                     {512, 2560},
                                                     {4 * eu, 2560},
                                                     {4 * eu + 1, 2560},
                                                     {1024, 2560},
                                                     {4096, 2560}};
    // GR_AB_ROWS=8192,16384 [GR_AB_D=2560,640] replaces the default points (long-context runs).
    if (const char* rows_env = std::getenv("GR_AB_ROWS")) {
        std::vector<size_t> row_list;
        std::vector<size_t> d_list = {2560};
        auto parse = [](const char* s, std::vector<size_t>& out) {
            out.clear();
            for (std::stringstream ss(s); ss.good();) {
                std::string tok;
                std::getline(ss, tok, ',');
                if (!tok.empty()) {
                    out.push_back(std::stoull(tok));
                }
            }
        };
        parse(rows_env, row_list);
        if (const char* d_env = std::getenv("GR_AB_D")) {
            parse(d_env, d_list);
        }
        points.clear();
        for (const auto d : d_list) {
            for (const auto r : row_list) {
                points.emplace_back(r, d);
            }
        }
    }
    const size_t L = 320;
    for (const auto& [rows, D] : points) {
        const size_t CD = kBranches * D;
        gr_ref_params p;
        p.rows = rows;
        p.D = D;
        p.L = L;
        tests::random_generator rg;
        rg.set_seed(GET_SUITE_NAME);
        std::mt19937 gen(99);
        // Long runs tile a 4096-row random block instead of generating rows * CD values.
        const size_t base_rows = std::min<size_t>(rows, 4096);
        const auto residual = random_vec(rg, base_rows * CD, 1.0f);
        const auto gain = random_vec(rg, CD, 0.5f);
        const quant_matrix wd4 = make_quant_matrix(gen, L, CD, true);
        const quant_matrix wu4 = make_quant_matrix(gen, CD, L, true);
        const quant_matrix wd8 = make_quant_matrix(gen, L, CD, false);
        const quant_matrix wu8 = make_quant_matrix(gen, CD, L, false);

        auto residual_mem =
            engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, data_types::f16, format::bfyx},
                                   allocation_type::usm_device);
        const auto residual_base_f16 = to_f16(residual);
        std::vector<ov::float16> residual_f16(rows * CD);
        for (size_t i = 0; i < residual_f16.size(); i++) {
            residual_f16[i] = residual_base_f16[i % residual_base_f16.size()];
        }
        residual_mem->copy_from(get_test_stream(), residual_f16.data(), true);

        const bool long_rows = rows >= 8192;
        const size_t nl = long_rows ? 1 : (rows <= 4 * eu ? 16 : 4);
        std::vector<quant_ab::ab_config> configs;
        configs.push_back({"f16", 0, {}});
        configs.push_back({"u4", 1, {}});
        configs.push_back({"u8", 2, {}});
        for (auto& cfg : configs) {
            for (size_t layer = 0; layer < nl; layer++) {
                const auto& wd = cfg.wtype == 1 ? wd4 : wd8;
                const auto& wu = cfg.wtype == 1 ? wu4 : wu8;
                std::vector<memory::ptr> w;
                if (cfg.wtype == 0) {
                    w = {make_f16(engine, {static_cast<int64_t>(L), static_cast<int64_t>(CD)}, wd4.dequant),
                         make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(L)}, wu4.dequant)};
                } else {
                    w = {make_quant_weights(engine, wd),
                         make_quant_weights(engine, wu),
                         make_f16(engine, {static_cast<int64_t>(wd.G), static_cast<int64_t>(L)}, wd.scale),
                         make_f16(engine, {static_cast<int64_t>(wu.G), static_cast<int64_t>(CD)}, wu.scale),
                         make_quant_zp(engine, wd),
                         make_quant_zp(engine, wu)};
                }
                topology topology;
                topology.add(input_layout("residual", residual_mem->get_layout()));
                topology.add(data("gain", quant_ab::to_device(engine, make_f16(engine, {static_cast<int64_t>(CD)}, gain))));
                std::vector<input_info> inputs = {input_info("residual"), input_info("gain")};
                for (size_t i = 0; i < w.size(); i++) {
                    topology.add(data("w" + std::to_string(i), quant_ab::to_device(engine, w[i])));
                    inputs.emplace_back("w" + std::to_string(i));
                }
                topology.add(gated_residual("gr", inputs, gated_residual::Mode::READ, p.C, D, L, p.eps, false, false));
                ExecutionConfig config = get_test_default_config(engine);
                config.set_property(ov::intel_gpu::optimize_data(true));
                config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
                config.set_property(ov::enable_profiling(true));
                auto net = std::make_unique<network>(engine, topology, config);
                net->set_input_data("residual", residual_mem);
                cfg.layers.push_back(std::move(net));
            }
        }

        // keep the best round of every configuration; rounds alternate between the configurations
        std::vector<double> best(configs.size(), 1e30);
        std::vector<std::vector<std::pair<std::string, double>>> best_stages(configs.size());
        const int rounds = long_rows ? 3 : 8;
        for (int r = 0; r < rounds; r++) {
            for (size_t c = 0; c < configs.size(); c++) {
                std::vector<std::pair<std::string, double>> stages;
                const double ms = quant_ab::measure_cycle(configs[c], long_rows ? 3 : 20, long_rows ? 8 : (rows >= 4096 ? 40 : 100), stages);
                if (ms < best[c]) {
                    best[c] = ms;
                    best_stages[c] = stages;
                }
            }
        }
        for (size_t c = 0; c < configs.size(); c++) {
            for (const auto& st : best_stages[c]) {
                std::cout << "GR_AB {\"rows\":" << rows << ",\"D\":" << D << ",\"w\":\"" << configs[c].name << "\",\"stage\":\"" << st.first
                          << "\",\"us\":" << st.second * 1000.0 << "}" << std::endl;
            }
            std::cout << "GR_AB {\"rows\":" << rows << ",\"D\":" << D << ",\"w\":\"" << configs[c].name << "\",\"stage\":\"TOTAL\",\"us\":" << best[c] * 1000.0
                      << "}" << std::endl;
        }
    }
}

struct gr_roofline_case {
    size_t tokens;
    bool decode;
};

class gated_residual_roofline_perf : public ::testing::TestWithParam<gr_roofline_case> {};

TEST_P(gated_residual_roofline_perf, measures_kernel_time) {
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
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
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
        if (layer_type == "GatedResidual" || layer_type == "gated_residual") {
            fused_nodes++;
        }
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
