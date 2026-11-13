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
//  - the READ without a write gate, which is the model-tail mixer that replaces the usual final norm.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gated_residual.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "gated_residual_inst.h"
#include "intel_gpu/op/gated_residual.hpp"
#include "intel_gpu/primitives/gated_residual_caps.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/device_query.hpp"
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
namespace caps = cldnn::gated_residual_caps;

namespace {

constexpr size_t kBranches = 4;

/// The device of the test engine (--device_suffix) for tests that compile through ov::Core.
std::string test_device_name() {
    const int id = cldnn::device_query::device_id;
    return id >= 0 ? "GPU." + std::to_string(id) : "GPU";
}
bool gr_available() {
    return caps::is_device_supported(get_test_engine().get_device_info());
}

/// The test networks' config after finalization, i.e. with the GPU_GR_* debug options read from the environment.
ExecutionConfig finalized_test_config() {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.finalize(engine);
    return config;
}

bool gr_systolic() {
    return caps::uses_dpas_kernels(get_test_engine().get_device_info(), finalized_test_config());
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
                const float g = gain[c * p.D + d];
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

/// Stages the last execution of `id` dispatched, named after their kernels' entry points.
std::vector<std::string> executed_stages(network& net, const primitive_id& id) {
    const auto inst = net.get_primitive(id);
    std::istringstream entries(inst->get_impl()->get_kernels_dump_info(*inst->get_impl_params()).get_entries());
    std::vector<std::string> names;
    for (std::string kernel; entries >> kernel;) {
        auto starts_with = [&](const char* prefix) {
            return kernel.rfind(prefix, 0) == 0;
        };
        if (starts_with("gr_write_")) {
            names.emplace_back("WRITE");
        } else if (starts_with("gr_read_pipeline_rbar")) {
            names.emplace_back("RBAR");
        } else if (starts_with("gr_read_pipeline_down")) {
            names.emplace_back("DOWN_PROJ");
        } else if (starts_with("gr_read_pipeline_up")) {
            names.emplace_back("UP_PROJ");
        } else if (starts_with("gr_read_pipeline_gate")) {
            names.emplace_back("GATE");
        } else if (starts_with("gr_read_dpas_")) {
            names.emplace_back("READ_DPAS");
        } else if (starts_with("gr_read_")) {
            names.emplace_back("READ");
        } else if (starts_with("gr_common_")) {
            names.emplace_back("DEQUANT");
        } else {
            names.push_back(kernel);
        }
    }
    return names;
}

/// With C = 4, C*D is always a multiple of 64, so only L decides the DPAS READ path. Above
/// 4 rows per EU (and D % 32 == 0) the single-kernel gr_read_dpas.cl replaces the 3-kernel
/// pipeline, whose DOWN_PROJ carries the write gate and UP_PROJ the branch mix. Devices without Xe2+ systolic
/// arrays always run the scalar gr_read.cl.
std::vector<std::string> pipeline_stages() {
    return {"RBAR", "DOWN_PROJ", "UP_PROJ"};
}

std::vector<std::string> expected_read_stages(size_t rows, size_t D, size_t L) {
    if (!caps::uses_dpas_read(L, gr_systolic())) {
        return {"READ"};
    }
    const size_t eu = get_test_engine().get_device_info().execution_units_count;
    if (D % 32 == 0 && rows > 4 * eu) {
        return {"READ_DPAS"};
    }
    return pipeline_stages();
}

struct gr_read_case {
    size_t rows;
    size_t D;
    size_t L;
    bool produce_write_gate;
    bool is_caching_test = false;
};

class gated_residual_read_test : public ::testing::TestWithParam<gr_read_case> {};

TEST_P(gated_residual_read_test, matches_fp32_reference) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    const auto tc = GetParam();
    if (!caps::is_geometry_supported(gated_residual::Mode::READ, 4, tc.D, tc.L, gr_systolic())) {
        GTEST_SKIP() << "the geometry needs the systolic READ kernels";
    }
    gr_ref_params p;
    p.rows = tc.rows;
    p.D = tc.D;
    p.L = tc.L;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const size_t CD = p.C * p.D;
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    // Gain in [0.5, 1.5].
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

    gated_residual prim("gr_read", inputs, gated_residual::Mode::READ, p.C, p.D, p.L, p.eps, tc.produce_write_gate);
    topology.add(prim);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), tc.is_caching_test);
    network->set_input_data("residual", residual_mem);

    auto outputs = network->execute();
    ASSERT_TRUE(outputs.count("gr_read") > 0);

    cldnn::mem_lock<ov::float16> mixed_out(outputs.at("gr_read").get_memory(), get_test_stream());
    EXPECT_EQ(executed_stages(*network, "gr_read"), expected_read_stages(p.rows, p.D, p.L));
    ASSERT_EQ(mixed_out.size(), ref_mixed.size());
    float mixed_err = 0.0f;
    for (size_t i = 0; i < ref_mixed.size(); i++) {
        ASSERT_NEAR(static_cast<float>(mixed_out[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at " << i;
        mixed_err = std::max(mixed_err, std::abs(static_cast<float>(mixed_out[i]) - ref_mixed[i]));
    }
    std::cout << "GR_F16_ACC rows=" << p.rows << " D=" << p.D << " L=" << p.L << " err_vs_fp32_ref=" << mixed_err;

    if (tc.produce_write_gate) {
        // Secondary outputs are reached through the instance, not the result map.
        auto inst = network->get_primitive("gr_read");
        cldnn::mem_lock<ov::float16> gate_out(inst->output_memory_ptr(1), get_test_stream());
        ASSERT_EQ(gate_out.size(), ref_gate.size());
        float gate_err = 0.0f;
        for (size_t i = 0; i < ref_gate.size(); i++) {
            const float v = static_cast<float>(gate_out[i]);
            // 2 * sigmoid(x) lives in (0, 2); a (0, 1) clamp would be a semantic bug.
            ASSERT_GT(v, 0.0f);
            ASSERT_LT(v, 2.0f);
            ASSERT_NEAR(v, ref_gate[i], 2e-2f) << "write gate mismatch at " << i;
            gate_err = std::max(gate_err, std::abs(v - ref_gate[i]));
        }
        std::cout << " gate_err_vs_fp32_ref=" << gate_err;
    }
    std::cout << std::endl;
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
                                           gr_read_case{32, 64, 32, true},      // exact token-tile boundary
                                           gr_read_case{37, 96, 32, false},     // partial last token tile, no write gate
                                           gr_read_case{41, 48, 64, true},      // partial tile; D=48, split-K over 6 K steps
                                           gr_read_case{16, 64, 32, true},      // exactly one N-tile
                                           gr_read_case{48, 64, 32, false},     // three full N-tiles
                                           gr_read_case{3, 32, 32, true},       // narrowest D
                                           gr_read_case{5, 64, 16, true},       // scalar path
                                           gr_read_case{19, 64, 32, true},      // DPAS path
                                           gr_read_case{1, 640, 320, true},     // production width, decode
                                           gr_read_case{17, 640, 320, true},    // production width, prefill with a partial tile
                                           gr_read_case{193, 32, 32, true},     // mid tiling
                                           gr_read_case{200, 64, 32, true},     // mid tiling, partial 32-token tile
                                           gr_read_case{300, 48, 64, false},    // mid tiling, D=48 column tiles
                                           gr_read_case{257, 640, 320, true},   // production-like width, mid tiling
                                           gr_read_case{17, 2560, 320, true},   // wide, decode tiling
                                           gr_read_case{100, 2560, 320, true},  // wide, mid tiling
                                           gr_read_case{1, 5120, 640, true},    // wider (C*D > 16384, L > 512), decode
                                           gr_read_case{17, 5120, 640, true},   // wider, prefill with a partial tile
                                           gr_read_case{100, 5120, 640, true},  // wider, mid tiling
                                           gr_read_case{17, 5120, 640, false},  // wider, no write gate
                                           gr_read_case{1000, 64, 32, true},    // wide tiling / single kernel (> 4 rows per EU)
                                           gr_read_case{777, 48, 64, false},    // wide tiling, partial tiles (D % 32 != 0)
                                           gr_read_case{1111, 128, 64, true},   // single kernel, partial 32-row tile
                                           gr_read_case{700, 640, 320, false},  // single kernel, production-like width
                                           // L=16 keeps these on gr_read.cl; above 32 rows one work-group owns
                                           // several rows, the last group only partially filled.
                                           gr_read_case{37, 64, 16, true},
                                           gr_read_case{100, 48, 16, false},
                                           gr_read_case{50, 64, 16, true},
                                           gr_read_case{65, 2560, 16, true},  // real width: 4 rows per work-group
                                           // The model tail mixer: READ without a write gate.
                                           gr_read_case{4, 64, 16, false},     // scalar gr_read
                                           gr_read_case{17, 64, 32, false},    // DPAS pipeline, decode tiling
                                           gr_read_case{250, 64, 32, false},   // DPAS pipeline, mid tiling
                                           gr_read_case{1100, 64, 32, false},  // DPAS pipeline, wide tiling
                                           // Model cache: the primitive is exported and imported before it runs.
                                           gr_read_case{19, 64, 32, true, true},     // DPAS pipeline with a write gate (two outputs)
                                           gr_read_case{5, 64, 16, true, true},      // scalar gr_read
                                           gr_read_case{1000, 64, 32, false, true},  // single kernel
                                           gr_read_case{1, 640, 320, true, true}));  // decode tiling

// --- compressed weights: u4 (group 32) / u8 (per-channel), asymmetric ------------------------------------------------------

bool gr_compression_available() {
    return gr_systolic();
}

/// One quantized [N, K] weight matrix with its [G, N] scale / optional zp and the exact f16 tensor the decompression graph produces.
struct quant_matrix {
    size_t N = 0;
    size_t K = 0;
    size_t G = 0;
    bool u4 = false;  // 4-bit (group 32), otherwise 8-bit (per-channel)
    bool is_signed = false;
    bool has_zp = true;
    std::vector<uint8_t> q;      // [N * K] raw element bits (two's complement for the signed types)
    std::vector<uint8_t> zp;     // [G * N] raw element bits, empty without zero points
    std::vector<float> scale;    // [G * N], f16 values
    std::vector<float> dequant;  // [N * K] = half(half(q - zp) * scale)

    data_types type() const {
        return u4 ? (is_signed ? data_types::i4 : data_types::u4) : (is_signed ? data_types::i8 : data_types::u8);
    }

    int value(uint8_t raw) const {
        if (!is_signed) {
            return raw;
        }
        return u4 ? static_cast<int>(raw ^ 8) - 8 : static_cast<int>(static_cast<int8_t>(raw));
    }
};

/// group == 0: the default of the bit width (32 for 4-bit, per-channel for 8-bit); kPerChannel: group size == K; otherwise the
/// requested group size when it divides K, else the default (W_up has a different K than W_down / W_inject).
constexpr size_t kPerChannel = static_cast<size_t>(-1);

quant_matrix make_quant_matrix(std::mt19937& gen, size_t N, size_t K, bool u4, bool is_signed = false, bool has_zp = true, size_t group = 0) {
    quant_matrix m;
    m.N = N;
    m.K = K;
    m.u4 = u4;
    m.is_signed = is_signed;
    m.has_zp = has_zp;
    size_t gs = u4 ? 32 : K;
    if (group == kPerChannel) {
        gs = K;
    } else if (group != 0 && K % group == 0) {
        gs = group;
    }
    m.G = K / gs;
    m.q.resize(N * K);
    m.zp.resize(has_zp ? m.G * N : 0);
    m.scale.resize(m.G * N);
    m.dequant.resize(N * K);
    // unsigned weights without a zero point are all positive: a smaller scale keeps the activations in the f16 range
    const float scale_div = (!has_zp && !is_signed ? 8.0f : 1.0f) * (u4 ? 512.0f : 2048.0f);
    for (size_t i = 0; i < m.G * N; i++) {
        if (has_zp) {
            if (u4) {
                m.zp[i] = static_cast<uint8_t>(gen() % 16);
            } else {
                m.zp[i] = is_signed ? static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(gen() % 60) - 30)) : static_cast<uint8_t>(100 + gen() % 60);
            }
        }
        m.scale[i] = static_cast<float>(ov::float16(static_cast<float>(1 + gen() % 8) / scale_div));
    }
    for (size_t n = 0; n < N; n++) {
        for (size_t k = 0; k < K; k++) {
            const size_t g = k / gs;
            const uint8_t q = static_cast<uint8_t>(gen() % (u4 ? 16 : 256));
            m.q[n * K + k] = q;
            const int zp = has_zp ? m.value(m.zp[g * N + n]) : 0;
            const float diff = static_cast<float>(ov::float16(static_cast<float>(m.value(q) - zp)));
            m.dequant[n * K + k] = static_cast<float>(ov::float16(diff * m.scale[g * N + n]));
        }
    }
    return m;
}

memory::ptr make_quant_weights(engine& eng, const quant_matrix& m) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(m.N), static_cast<int64_t>(m.K)}, m.type(), format::bfyx});
    cldnn::mem_lock<uint8_t> lock(mem, get_test_stream());
    if (m.u4) {
        for (size_t i = 0; i < m.q.size(); i += 2) {
            lock[i / 2] = static_cast<uint8_t>((m.q[i] & 0xF) | (m.q[i + 1] << 4));  // low nibble = even k
        }
    } else {
        std::copy(m.q.begin(), m.q.end(), lock.begin());
    }
    return mem;
}

memory::ptr make_quant_zp(engine& eng, const quant_matrix& m) {
    auto mem = eng.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(m.G), static_cast<int64_t>(m.N)}, m.type(), format::bfyx});
    cldnn::mem_lock<uint8_t> lock(mem, get_test_stream());
    if (m.u4) {
        for (size_t i = 0; i < m.zp.size(); i += 2) {
            lock[i / 2] = static_cast<uint8_t>((m.zp[i] & 0xF) | (m.zp[i + 1] << 4));  // low nibble = even index
        }
    } else {
        std::copy(m.zp.begin(), m.zp.end(), lock.begin());
    }
    return mem;
}

/// f16 scale tensor [G, N] of a quantized matrix.
memory::ptr make_quant_scale(engine& eng, const quant_matrix& m) {
    return make_f16(eng, {static_cast<int64_t>(m.G), static_cast<int64_t>(m.N)}, m.scale);
}

struct gr_run_result {
    std::vector<ov::float16> mixed;
    std::vector<ov::float16> gate;  // only with a write gate
    std::vector<std::string> stages;
};

/// Runs a READ gated_residual whose weight inputs (from index 2 on) are the given memories. With `write_gate` the
/// last inputs are W_inject (f16, or compressed followed by its scale and optional zp) and output 1 is read back.
gr_run_result run_gr_read(const gr_ref_params& p,
                          const std::vector<float>& residual,
                          const std::vector<float>& gain,
                          const std::vector<memory::ptr>& weights,
                          bool write_gate = false,
                          const std::string& path = "",
                          bool is_caching_test = false) {
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
    topology.add(gated_residual("gr", inputs, gated_residual::Mode::READ, p.C, p.D, p.L, p.eps, write_gate));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
#ifdef GPU_DEBUG_CONFIG
    if (!path.empty()) {
        config.set_property(ov::intel_gpu::gr_kernel_selection(path));
    }
#endif
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    network->set_input_data("residual", residual_mem);
    auto outputs = network->execute();
    gr_run_result result;
    cldnn::mem_lock<ov::float16> out(outputs.at("gr").get_memory(), get_test_stream());
    result.mixed.assign(out.begin(), out.end());
    if (write_gate) {
        cldnn::mem_lock<ov::float16> gate(network->get_primitive("gr")->output_memory_ptr(1), get_test_stream());
        result.gate.assign(gate.begin(), gate.end());
    }
    result.stages = executed_stages(*network, "gr");
    return result;
}

struct gr_quant_case {
    size_t rows;
    size_t D;
    size_t L;
    bool u4;  // 4-bit weights, otherwise 8-bit
    bool write_gate = false;
    bool is_signed = false;
    bool has_zp = true;
    size_t group = 0;     // see make_quant_matrix
    bool scalar = false;  // the scalar gr_read.cl (GPU_GR_KERNEL_SELECTION=scalar), which dequantizes in registers too
    bool is_caching_test = false;
};

size_t scratch_limit_bytes() {
    return GPU_DEBUG_VALUE_OR(finalized_test_config().get_gr_scratch_limit_mb(), size_t{256}) << 20;
}

size_t pd_min_rows() {
    const int64_t rows = GPU_DEBUG_VALUE_OR(finalized_test_config().get_gr_pd_row_threshold(), int64_t{-1});
    return rows >= 0 ? static_cast<size_t>(rows) : 5 * static_cast<size_t>(get_test_engine().get_device_info().execution_units_count);
}

class gated_residual_quant_test : public ::testing::TestWithParam<gr_quant_case> {};

/// Largest |a - b| over two equally long vectors.
template <typename T>
float max_abs_diff(const std::vector<T>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < b.size(); i++) {
        m = std::max(m, std::abs(static_cast<float>(a[i]) - b[i]));
    }
    return m;
}

// The kernels dequantize in registers with the arithmetic of the decompression graph, so the result must equal the f16 path
// fed with the decompressed weights bit for bit (same tiling, same accumulation order), and match the fp32 reference.
TEST_P(gated_residual_quant_test, matches_decompressed_f16_path) {
    const auto tc = GetParam();
    if (!(tc.scalar ? gr_available() : gr_compression_available())) {
        GTEST_SKIP() << "compressed GatedResidual weights need an Xe2+ systolic GPU or the scalar kernel";
    }
#ifndef GPU_DEBUG_CONFIG
    if (tc.scalar) {
        GTEST_SKIP() << "GPU_GR_KERNEL_SELECTION is a debug option";
    }
#endif
    gr_ref_params p;
    p.rows = tc.rows;
    p.D = tc.D;
    p.L = tc.L;
    const size_t CD = p.C * p.D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    std::mt19937 gen(1234 + static_cast<uint32_t>(tc.rows * 7 + tc.D + tc.L + (tc.u4 ? 1 : 0) + (tc.is_signed ? 2 : 0) + (tc.has_zp ? 0 : 4) + tc.group));
    const auto residual = random_vec(rg, p.rows * CD, 1.0f);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto wd = make_quant_matrix(gen, p.L, CD, tc.u4, tc.is_signed, tc.has_zp, tc.group);
    const auto wu = make_quant_matrix(gen, CD, p.L, tc.u4, tc.is_signed, tc.has_zp, tc.group);
    // W_inject: C rows only, compressed like W_down / W_up
    const auto wi = make_quant_matrix(gen, p.C, CD, tc.u4, tc.is_signed, tc.has_zp, tc.group);

    auto& engine = get_test_engine();
    std::vector<memory::ptr> f16_weights = {make_f16(engine, {static_cast<int64_t>(p.L), static_cast<int64_t>(CD)}, wd.dequant),
                                            make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(p.L)}, wu.dequant)};
    std::vector<memory::ptr> q_weights = {make_quant_weights(engine, wd),
                                          make_quant_weights(engine, wu),
                                          make_quant_scale(engine, wd),
                                          make_quant_scale(engine, wu)};
    if (tc.has_zp) {
        q_weights.push_back(make_quant_zp(engine, wd));
        q_weights.push_back(make_quant_zp(engine, wu));
    }
    if (tc.write_gate) {
        f16_weights.push_back(make_f16(engine, {static_cast<int64_t>(p.C), static_cast<int64_t>(CD)}, wi.dequant));
        q_weights.push_back(make_quant_weights(engine, wi));
        q_weights.push_back(make_quant_scale(engine, wi));
        if (tc.has_zp) {
            q_weights.push_back(make_quant_zp(engine, wi));
        }
    }
    const auto f16_run = run_gr_read(p, residual, gain, f16_weights, tc.write_gate, tc.scalar ? "scalar" : "", tc.is_caching_test);
    const auto q_run = run_gr_read(p, residual, gain, q_weights, tc.write_gate, tc.scalar ? "scalar" : "", tc.is_caching_test);

    // compressed weights use the 3-dispatch pipeline unless its Rbar scratch exceeds the limit
    const auto expected = tc.scalar ? std::vector<std::string>{"READ"} : expected_read_stages(p.rows, p.D, p.L);
    EXPECT_EQ(f16_run.stages, expected);
    const bool single_q = p.D % 32 == 0 && p.rows * p.C * p.D * 2 > scratch_limit_bytes();
    // The compressed single kernel has no write-gate tile: with a write gate that regime predequantizes as well.
    const bool predequant = p.rows > pd_min_rows() || (tc.write_gate && single_q);
    std::vector<std::string> expected_q;
    if (tc.scalar) {
        // one kernel, the weights are dequantized in registers
        expected_q = expected;
    } else if (predequant) {
        // both matrices (and W_inject) are dequantized into f16 scratch and the f16 kernels (f16 routing) run on them
        expected_q = {"DEQUANT", "DEQUANT"};
        if (tc.write_gate) {
            expected_q.emplace_back("DEQUANT");
        }
        expected_q.insert(expected_q.end(), expected.begin(), expected.end());
    } else if (single_q) {
        expected_q = {"READ_DPAS"};
    } else if (p.rows <= 32) {
        // decode: the write gate is a column tile of the compressed DOWN_PROJ
        expected_q = pipeline_stages();
    } else {
        // W_inject is dequantized first; the write gate is its own dispatch between RBAR and DOWN_PROJ
        if (tc.write_gate) {
            expected_q.emplace_back("DEQUANT");
        }
        expected_q.emplace_back("RBAR");
        if (tc.write_gate) {
            expected_q.emplace_back("GATE");
        }
        expected_q.insert(expected_q.end(), {"DOWN_PROJ", "UP_PROJ"});
    }
    EXPECT_EQ(q_run.stages, expected_q);

    std::vector<float> ref_mixed;
    std::vector<float> ref_gate;
    reference_read(p, residual, gain, wd.dequant, wu.dequant, tc.write_gate ? &wi.dequant : nullptr, ref_mixed, ref_gate);
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
    std::cout << "GR_QUANT_ACC rows=" << p.rows << " D=" << p.D << " L=" << p.L << " " << (tc.is_signed ? "i" : "u") << (tc.u4 ? "4" : "8")
              << " g_down=" << CD / wd.G << " g_up=" << p.L / wu.G << (tc.has_zp ? "" : " nozp") << (tc.write_gate ? " gate" : "")
              << " differing_elems=" << bit_diffs << "/" << ref_mixed.size() << " max_abs_diff_vs_f16=" << max_diff
              << " err_vs_fp32_ref: f16=" << max_abs_diff(f16_run.mixed, ref_mixed) << " q=" << max_abs_diff(q_run.mixed, ref_mixed);
    if (tc.write_gate) {
        std::cout << " gate_f16=" << max_abs_diff(f16_run.gate, ref_gate) << " gate_q=" << max_abs_diff(q_run.gate, ref_gate);
    }
    std::cout << std::endl;
    // the scalar kernel runs the same code on both inputs: bit-exact
    EXPECT_LE(max_diff, tc.scalar ? 0.0f : 4e-3f) << "compressed and decompressed-f16 outputs differ";

    if (tc.write_gate) {
        ASSERT_EQ(q_run.gate.size(), ref_gate.size());
        ASSERT_EQ(f16_run.gate.size(), ref_gate.size());
        float gate_diff = 0.0f;
        for (size_t i = 0; i < ref_gate.size(); i++) {
            const float v = static_cast<float>(q_run.gate[i]);
            // 2 * sigmoid(x) lives in [0, 2] (f16 rounds a saturated gate to the bound); a (0, 1) clamp would be a semantic bug.
            ASSERT_GE(v, 0.0f);
            ASSERT_LE(v, 2.0f);
            ASSERT_NEAR(v, ref_gate[i], 2e-2f) << "write gate mismatch at " << i;
            gate_diff = std::max(gate_diff, std::abs(v - static_cast<float>(f16_run.gate[i])));
        }
        EXPECT_LE(gate_diff, tc.scalar ? 0.0f : 4e-3f) << "compressed and decompressed-f16 write gates differ";
    }
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
                                           gr_quant_case{257, 640, 320, false},  // mid tiling
                                           gr_quant_case{250, 64, 32, true},     // mid tiling
                                           gr_quant_case{1000, 64, 32, true},    // single kernel
                                           gr_quant_case{1000, 64, 32, false},
                                           gr_quant_case{1111, 128, 64, true},  // partial 32-row tile
                                           gr_quant_case{700, 640, 320, true},
                                           gr_quant_case{700, 640, 320, false},
                                           gr_quant_case{17, 2560, 320, true},  // real width
                                           gr_quant_case{17, 2560, 320, false},
                                           gr_quant_case{100, 2560, 320, true},
                                           gr_quant_case{2000, 2560, 320, true},
                                           gr_quant_case{2000, 2560, 320, false},
                                           // compressed W_inject: the write gate of a compressed READ (11 inputs)
                                           gr_quant_case{1, 64, 32, true, true},  // decode
                                           gr_quant_case{1, 64, 32, false, true},
                                           gr_quant_case{7, 128, 32, true, true},
                                           gr_quant_case{7, 128, 32, false, true},
                                           gr_quant_case{33, 128, 64, true, true},  // mid tiling
                                           gr_quant_case{200, 64, 32, false, true},
                                           gr_quant_case{37, 96, 64, false, true},   // partial token tile, D % 64 != 0
                                           gr_quant_case{17, 640, 320, true, true},  // production width
                                           gr_quant_case{17, 640, 320, false, true},
                                           gr_quant_case{257, 640, 320, false, true},
                                           gr_quant_case{1000, 64, 32, true, true},  // above the predequantization threshold
                                           gr_quant_case{1000, 64, 32, false, true},
                                           gr_quant_case{1111, 128, 64, true, true},
                                           gr_quant_case{17, 2560, 320, true, true},  // real width
                                           gr_quant_case{17, 2560, 320, false, true},
                                           gr_quant_case{100, 2560, 320, true, true},
                                           gr_quant_case{2000, 2560, 320, true, true},
                                           gr_quant_case{2000, 2560, 320, false, true},
                                           // gemv tier (<= 8 rows) at production widths
                                           gr_quant_case{1, 640, 320, true},
                                           gr_quant_case{1, 640, 320, false},
                                           gr_quant_case{8, 640, 320, true},
                                           gr_quant_case{8, 640, 320, false},
                                           gr_quant_case{1, 2560, 320, true},
                                           gr_quant_case{1, 2560, 320, false},
                                           gr_quant_case{8, 2560, 320, true},
                                           gr_quant_case{8, 2560, 320, false},
                                           gr_quant_case{1, 640, 320, true, true},
                                           gr_quant_case{1, 640, 320, false, true},
                                           gr_quant_case{8, 640, 320, true, true},
                                           gr_quant_case{8, 640, 320, false, true},
                                           gr_quant_case{1, 2560, 320, true, true},
                                           gr_quant_case{1, 2560, 320, false, true},
                                           gr_quant_case{8, 2560, 320, true, true},
                                           gr_quant_case{8, 2560, 320, false, true},
                                           // RBAR splits the 5120-wide branch rows across subgroups at decode
                                           gr_quant_case{1, 5120, 640, true},
                                           gr_quant_case{1, 5120, 640, false},
                                           gr_quant_case{17, 5120, 640, true},
                                           gr_quant_case{17, 5120, 640, false},
                                           gr_quant_case{100, 5120, 640, true},
                                           gr_quant_case{1, 5120, 640, true, true},
                                           gr_quant_case{17, 5120, 640, false, true},
                                           // signed weights (i4 / i8) with zero points: gemv, decode, mid and predequantized
                                           gr_quant_case{1, 64, 32, true, false, true},
                                           gr_quant_case{1, 64, 32, false, false, true},
                                           gr_quant_case{7, 128, 32, true, true, true},
                                           gr_quant_case{7, 128, 32, false, true, true},
                                           gr_quant_case{1, 2560, 320, true, true, true},
                                           gr_quant_case{8, 2560, 320, false, true, true},
                                           gr_quant_case{17, 640, 320, true, true, true},
                                           gr_quant_case{17, 640, 320, false, false, true},
                                           gr_quant_case{33, 128, 64, true, true, true},
                                           gr_quant_case{257, 640, 320, false, true, true},
                                           gr_quant_case{250, 64, 32, true, false, true},
                                           gr_quant_case{1000, 64, 32, true, true, true},
                                           gr_quant_case{1000, 64, 32, false, false, true},
                                           gr_quant_case{2000, 2560, 320, true, true, true},
                                           // no zero points (the zero point is 0), unsigned and signed
                                           gr_quant_case{1, 64, 32, true, false, false, false},
                                           gr_quant_case{1, 64, 32, false, true, false, false},
                                           gr_quant_case{7, 128, 32, true, true, true, false},
                                           gr_quant_case{7, 128, 32, false, false, true, false},
                                           gr_quant_case{1, 2560, 320, true, true, true, false},
                                           gr_quant_case{8, 2560, 320, false, true, false, false},
                                           gr_quant_case{17, 640, 320, true, true, false, false},
                                           gr_quant_case{17, 640, 320, false, true, true, false},
                                           gr_quant_case{33, 128, 64, true, true, true, false},
                                           gr_quant_case{257, 640, 320, false, true, false, false},
                                           gr_quant_case{250, 64, 32, false, false, false, false},
                                           gr_quant_case{1000, 64, 32, true, true, true, false},
                                           gr_quant_case{1000, 64, 32, false, false, false, false},
                                           gr_quant_case{2000, 2560, 320, false, true, true, false},
                                           // Model cache: the primitive is exported and imported before it runs.
                                           gr_quant_case{17, 640, 320, true, true, false, true, 0, false, true},     // u4 with zero points and a write gate
                                           gr_quant_case{17, 640, 320, true, false, false, false, 0, false, true},   // u4 without zero points
                                           gr_quant_case{200, 64, 32, false, true, true, true, 0, false, true},      // i8 with a write gate, mid tiling
                                           gr_quant_case{1000, 64, 32, true, false, false, true, 0, false, true}));  // predequantized weights

/// 32 / 64 / 128 and per-channel group sizes of both bit widths on every kernel path; W_up keeps the default group size when it does
/// not divide its K (= L).
std::vector<gr_quant_case> group_size_cases() {
    struct shape {
        size_t rows;
        size_t D;
        size_t L;
        bool write_gate;
    };
    // gemv, decode, mid (shared dequantization), predequantized, production widths
    const std::vector<shape> shapes = {{1, 64, 32, false},
                                       {7, 128, 32, true},
                                       {17, 640, 320, false},
                                       {33, 128, 64, true},
                                       {257, 640, 320, true},
                                       {1000, 64, 32, false},
                                       {1, 2560, 320, true},
                                       {8, 2560, 320, false},
                                       {2000, 2560, 320, true}};
    std::vector<gr_quant_case> cases;
    for (const size_t group : {size_t{32}, size_t{64}, size_t{128}, kPerChannel}) {
        for (const bool u4 : {true, false}) {
            for (const auto& s : shapes) {
                // The bit width's default group size is already exercised by the main list, except for these two shapes.
                const bool is_default = (group == 32 && u4) || (group == kPerChannel && !u4);
                const bool in_main_list = is_default && (s.rows != 33 || u4) && (s.rows != 257 || !u4);
                if (in_main_list) {
                    continue;
                }
                cases.push_back({s.rows, s.D, s.L, u4, s.write_gate, false, true, group});
            }
        }
    }
    // signed weights and weights without zero points
    for (const size_t group : {size_t{64}, size_t{128}}) {
        for (const bool u4 : {true, false}) {
            cases.push_back({17, 640, 320, u4, true, true, false, group});
            cases.push_back({1, 2560, 320, u4, false, true, true, group});
            cases.push_back({257, 640, 320, u4, true, false, false, group});
        }
    }
    // L = 640 divides by 64 / 128, so W_up really uses the group size too instead of falling back to the default
    for (const size_t group : {size_t{64}, size_t{128}}) {
        for (const bool u4 : {true, false}) {
            for (const size_t rows : {size_t{1}, size_t{17}, size_t{100}, size_t{257}, size_t{1000}}) {
                cases.push_back({rows, 640, 640, u4, true, false, true, group});
            }
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(smoke_group_sizes, gated_residual_quant_test, ::testing::ValuesIn(group_size_cases()));

/// The scalar gr_read.cl (GPU_GR_KERNEL_SELECTION=scalar, as on a device without systolic arrays) dequantizes in registers too: both weight
/// widths, group sizes, zero points, signedness and the compressed write gate (L <= 512, C * D <= 16384).
std::vector<gr_quant_case> scalar_cases() {
    std::vector<gr_quant_case> cases;
    for (const bool u4 : {true, false}) {
        for (const size_t rows : {size_t{1}, size_t{7}, size_t{20}, size_t{40}}) {
            for (const size_t group : {size_t{0}, size_t{64}, kPerChannel}) {
                cases.push_back({rows, 128, 64, u4, false, false, true, group, true});
            }
        }
        // write gate, signed weights, no zero points, a partial row tile
        cases.push_back({5, 64, 32, u4, true, false, true, 0, true});
        cases.push_back({20, 96, 64, u4, true, true, false, 0, true});
        cases.push_back({9, 64, 32, u4, false, true, true, 0, true});
        cases.push_back({33, 48, 32, u4, false, false, false, 0, true});
        // production widths
        cases.push_back({17, 640, 320, u4, true, false, true, 0, true});
        cases.push_back({40, 2560, 320, u4, false, true, true, 0, true});
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(smoke_scalar, gated_residual_quant_test, ::testing::ValuesIn(scalar_cases()));

/// R'_c = R_c + S_c * Y
std::vector<float> reference_write(size_t rows,
                                   size_t C,
                                   size_t D,
                                   const std::vector<float>& residual,
                                   const std::vector<float>& y,
                                   const std::vector<float>& gate) {
    const size_t CD = C * D;
    std::vector<float> ref(rows * CD, 0.0f);
    for (size_t row = 0; row < rows; row++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t d = 0; d < D; d++) {
                ref[row * CD + c * D + d] = residual[row * CD + c * D + d] + gate[row * C + c] * y[row * D + d];
            }
        }
    }
    return ref;
}

struct gr_write_case {
    size_t rows;
    size_t D;
    bool is_caching_test = false;
};

class gated_residual_write_test : public ::testing::TestWithParam<gr_write_case> {};

TEST_P(gated_residual_write_test, combines_branches) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    const size_t rows = GetParam().rows;
    const size_t C = kBranches;
    const size_t D = GetParam().D;
    const size_t CD = C * D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);

    const auto residual = random_vec(rg, rows * CD, 1.0f);
    const auto y = random_vec(rg, rows * D, 1.0f);
    const auto gate = random_vec(rg, rows * C, 1.0f, 1.0f);

    const auto ref = reference_write(rows, C, D, residual, y, gate);

    auto& engine = get_test_engine();
    auto residual_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, residual);
    auto y_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(D)}, y);
    auto gate_mem = make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(C)}, gate);

    topology topology;
    topology.add(input_layout("residual", residual_mem->get_layout()));
    topology.add(input_layout("y", y_mem->get_layout()));
    topology.add(input_layout("gate", gate_mem->get_layout()));
    topology.add(gated_residual("gr_write", {input_info("residual"), input_info("y"), input_info("gate")}, gated_residual::Mode::WRITE, C, D, 0, 1e-6f, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), GetParam().is_caching_test);
    network->set_input_data("residual", residual_mem);
    network->set_input_data("y", y_mem);
    network->set_input_data("gate", gate_mem);

    auto outputs = network->execute();
    cldnn::mem_lock<ov::float16> out(outputs.at("gr_write").get_memory(), get_test_stream());
    EXPECT_EQ(executed_stages(*network, "gr_write"), std::vector<std::string>{"WRITE"});
    ASSERT_EQ(out.size(), ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at " << i;
    }
}

// D=48 is a multiple of 16 but not of 32 or 64, so gr_best_tile() falls back to GR_WRITE_TILE=16; D=64 and 256 use the wide tile.
INSTANTIATE_TEST_SUITE_P(smoke,
                         gated_residual_write_test,
                         ::testing::Values(gr_write_case{5, 64},
                                           gr_write_case{9, 48},
                                           gr_write_case{37, 256},
                                           gr_write_case{3, 5120},
                                           gr_write_case{33, 5120},
                                           gr_write_case{37, 256, true}));

// A cached dynamic READ holds the stages of every row-count route; the imported network has to run each of them.
TEST(gated_residual_caching, dynamic_read_runs_every_row_route) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = 32;
    const size_t CD = C * D;

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * L, 0.1f);
    const auto w_inject = random_vec(rg, C * CD, 0.1f);

    auto& engine = get_test_engine();
    topology topology;
    topology.add(input_layout("residual", layout{ov::PartialShape{-1, static_cast<int64_t>(CD)}, data_types::f16, format::bfyx}));
    topology.add(data("gain", make_f16(engine, {static_cast<int64_t>(CD)}, gain)));
    topology.add(data("w_down", make_f16(engine, {static_cast<int64_t>(L), static_cast<int64_t>(CD)}, w_down)));
    topology.add(data("w_up", make_f16(engine, {static_cast<int64_t>(CD), static_cast<int64_t>(L)}, w_up)));
    topology.add(data("w_inject", make_f16(engine, {static_cast<int64_t>(C), static_cast<int64_t>(CD)}, w_inject)));
    topology.add(gated_residual("gr",
                                {input_info("residual"), input_info("gain"), input_info("w_down"), input_info("w_up"), input_info("w_inject")},
                                gated_residual::Mode::READ,
                                C,
                                D,
                                L,
                                1e-6f,
                                true));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), true);

    // decode, mid and wide tilings, then the single kernel, and back
    for (const size_t rows : {size_t{5}, size_t{37}, size_t{1}, size_t{300}, size_t{1500}, size_t{2}}) {
        gr_ref_params p;
        p.rows = rows;
        p.D = D;
        p.L = L;
        const auto residual = random_vec(rg, rows * CD, 1.0f);
        std::vector<float> ref_mixed;
        std::vector<float> ref_gate;
        reference_read(p, residual, gain, w_down, w_up, &w_inject, ref_mixed, ref_gate);

        network->set_input_data("residual", make_f16(engine, {static_cast<int64_t>(rows), static_cast<int64_t>(CD)}, residual));
        auto outputs = network->execute();
        // A dynamic output buffer may be larger than the shape of this run, so check the layout and read the used part only.
        const auto inst = network->get_primitive("gr");
        ASSERT_EQ(inst->get_output_layout(0).get_partial_shape(), (ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(D)})) << "rows=" << rows;
        ASSERT_EQ(inst->get_output_layout(1).get_partial_shape(), (ov::PartialShape{static_cast<int64_t>(rows), static_cast<int64_t>(C)})) << "rows=" << rows;
        cldnn::mem_lock<ov::float16> mixed(outputs.at("gr").get_memory(), get_test_stream());
        ASSERT_GE(mixed.size(), ref_mixed.size()) << "rows=" << rows;
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(mixed[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
        cldnn::mem_lock<ov::float16> gate(inst->output_memory_ptr(1), get_test_stream());
        ASSERT_GE(gate.size(), ref_gate.size()) << "rows=" << rows;
        for (size_t i = 0; i < ref_gate.size(); i++) {
            ASSERT_NEAR(static_cast<float>(gate[i]), ref_gate[i], 2e-2f) << "write gate mismatch at rows=" << rows << " index=" << i;
        }
    }
}

// --- ov::Model / ov::Core level tests (plugin pipeline, dynamic rows) ----------
//
// The tests above go through the raw cldnn::topology/network API, i.e. they bypass
// ov::Model -> ProgramBuilder -> CreateGatedResidualOp. The tests below build a tiny ov::Model with a directly-constructed
// ov::intel_gpu::op::GatedResidual node (no pattern fusion involved) and run it through ov::Core::compile_model on the test device and
// ov::InferRequest::infer() with changing row counts on one compiled model.
// Dynamic row counts on one compiled model / infer request: the dispatch data must be recomputed for every shape.
struct gr_dynamic_read_case {
    size_t L;
    bool write_gate;
    std::vector<size_t> rows;
};

class gated_residual_dynamic_read_test : public ::testing::TestWithParam<gr_dynamic_read_case> {};

TEST_P(gated_residual_dynamic_read_test, ov_model_reuses_dispatch_data) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    ov::Core core;
    const auto& tc = GetParam();
    const size_t C = kBranches;
    const size_t D = 64;
    const size_t L = tc.L;
    const size_t CD = C * D;

    auto residual_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD});
    auto w_down_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{L, CD});
    auto w_up_p = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{CD, L});
    ov::ParameterVector params{residual_p, gain_p, w_down_p, w_up_p};
    if (tc.write_gate) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{C, CD}));
    }

    ov::OutputVector inputs(params.begin(), params.end());
    auto gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(inputs, ov::intel_gpu::op::GatedResidual::Mode::READ, C, D, L, 1e-6f, tc.write_gate);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gr->output(0))};
    if (tc.write_gate) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gr->output(1)));
    }
    auto model = std::make_shared<ov::Model>(results, params);

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, test_device_name()));
    auto infer_request = compiled.create_infer_request();

    tests::random_generator rg;
    rg.set_seed(GET_SUITE_NAME);
    const auto gain = random_vec(rg, CD, 0.5f, 1.0f);
    const auto w_down = random_vec(rg, L * CD, 0.1f);
    const auto w_up = random_vec(rg, CD * L, 0.1f);
    const auto w_inject = tc.write_gate ? random_vec(rg, C * CD, 0.1f) : std::vector<float>{};

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
    if (tc.write_gate) {
        fill_tensor(infer_request.get_input_tensor(4), w_inject);
    }

    for (const size_t rows : tc.rows) {
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
        reference_read(p, residual, gain, w_down, w_up, tc.write_gate ? &w_inject : nullptr, ref_mixed, ref_gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto mixed = infer_request.get_output_tensor(0);
        const auto* mixed_data = mixed.data<ov::float16>();
        ASSERT_EQ(mixed.get_size(), ref_mixed.size());
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(mixed_data[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
        if (tc.write_gate) {
            const auto gate = infer_request.get_output_tensor(1);
            const auto* gate_data = gate.data<ov::float16>();
            ASSERT_EQ(gate.get_size(), ref_gate.size());
            for (size_t i = 0; i < ref_gate.size(); i++) {
                ASSERT_NEAR(static_cast<float>(gate_data[i]), ref_gate[i], 2e-2f) << "write gate mismatch at rows=" << rows << " index=" << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         gated_residual_dynamic_read_test,
                         ::testing::Values(
                             // DPAS pipeline: walks the decode, mid and wide tilings and back again
                             gr_dynamic_read_case{32, true, {5, 37, 1, 300, 1500, 2}},
                             // L=16 is not a multiple of 32, so the scalar gr_read.cl stage runs instead
                             gr_dynamic_read_case{16, false, {3, 41, 1, 100, 2}}));

// Dynamic row counts on one compiled model / infer request for GrWriteGenerator's dispatch-data path (gr_write.cl).
struct gr_dynamic_write_case {
    size_t D;
};

class gated_residual_dynamic_write_test : public ::testing::TestWithParam<gr_dynamic_write_case> {};

TEST_P(gated_residual_dynamic_write_test, ov_model_reuses_dispatch_data) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }

    ov::Core core;
    const size_t C = kBranches;
    const size_t D = GetParam().D;
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
                                                                 /*produce_write_gate=*/false);
    auto result = std::make_shared<ov::op::v0::Result>(gr);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{residual_p, y_p, gate_p});

    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, test_device_name()));
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

        const auto ref = reference_write(rows, C, D, residual, y, gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto out_tensor = infer_request.get_output_tensor(0);
        const auto* out = out_tensor.data<ov::float16>();
        ASSERT_EQ(out_tensor.get_size(), ref.size());
        for (size_t i = 0; i < ref.size(); i++) {
            ASSERT_NEAR(static_cast<float>(out[i]), ref[i], 1e-2f) << "residual update mismatch at rows=" << rows << " index=" << i;
        }
    }
}

// D = 256 is hidden_size / C.
INSTANTIATE_TEST_SUITE_P(smoke, gated_residual_dynamic_write_test, ::testing::Values(gr_dynamic_write_case{64}, gr_dynamic_write_case{256}));

// Equivalence of the decomposed source graph and the fused kernel: the decomposed READ/MIX
// model is compiled through the normal plugin pipeline (which must fuse it into a single
// gated_residual node) and compared against the FP32 reference of the *decomposed* math.
// With the write gate (2 * sigmoid(Rbar . W_inject^T / C), on the same Rbar as the READ) the gate branch is fused into the READ as well:
// one gated_residual node with two outputs, no MatMul left, and both outputs match the FP32 reference. A fully static model
// compiles without the new shape inference unless an op requires it, which the two-output READ does.
class gated_residual_decomposed_read_test : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(gated_residual_decomposed_read_test, fuses_and_matches_reference) {
    if (!gr_available()) {
        GTEST_SKIP() << "no GatedResidual implementation is available on this device";
    }
    const bool with_write_gate = std::get<0>(GetParam());
    const bool static_rows = std::get<1>(GetParam());
    if (static_rows && !get_test_engine().get_device_info().supports_immad) {
        // Without immad a static f16 ReduceSum is converted to a Pooling before the fusion runs; dynamic shapes are not.
        GTEST_SKIP() << "static-shape GatedResidual fusion is not available without immad";
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
    const auto w_inject = with_write_gate ? random_vec(rg, C * CD, 0.1f) : std::vector<float>{};

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
    constexpr int64_t kStaticRows = 64;
    auto residual_p = std::make_shared<ov::op::v0::Parameter>(et, ov::PartialShape{static_rows ? kStaticRows : -1, static_cast<int64_t>(CD)});
    auto gain_p = f16_param(ov::Shape{CD});
    auto w_down_p = f16_param(ov::Shape{L, CD});
    auto w_up_p = f16_param(ov::Shape{CD, L});
    ov::ParameterVector params{residual_p, gain_p, w_down_p, w_up_p};
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

    ov::OutputVector outputs{mixed};
    if (with_write_gate) {
        auto w_inject_p = f16_param(ov::Shape{C, CD});
        params.push_back(w_inject_p);
        auto inject = std::make_shared<ov::op::v0::MatMul>(rbar, w_inject_p, false, true);
        auto inject_scaled = std::make_shared<ov::op::v1::Multiply>(inject, scalar(1.0f / static_cast<float>(C)));
        outputs.push_back(std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v0::Sigmoid>(inject_scaled), scalar(2.0f)));
    }
    auto model = std::make_shared<ov::Model>(outputs, params);

    ov::Core core;
    ov::CompiledModel compiled;
    OV_ASSERT_NO_THROW(compiled = core.compile_model(model, test_device_name()));
    size_t fused_nodes = 0;
    size_t gemm_nodes = 0;
    std::string runtime_layers;
    for (const auto& op : compiled.get_runtime_model()->get_ops()) {
        const auto it = op->get_rt_info().find("layerType");
        const auto layer_type = it != op->get_rt_info().end() ? it->second.as<std::string>() : std::string("?");
        if (layer_type == "GatedResidual" || layer_type == "gated_residual") {
            fused_nodes++;
        }
        if (layer_type == "FullyConnected" || layer_type == "fully_connected" || layer_type == "Gemm" || layer_type == "gemm") {
            gemm_nodes++;
        }
        runtime_layers += layer_type + " ";
    }
    EXPECT_EQ(fused_nodes, 1u) << "decomposed READ graph was not fused into one gated_residual; runtime layers: " << runtime_layers;
    EXPECT_EQ(gemm_nodes, 0u) << "a MatMul is left; runtime layers: " << runtime_layers;

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
    if (with_write_gate) {
        fill(infer_request.get_input_tensor(4), w_inject);
    }
    const auto row_counts = static_rows ? std::vector<size_t>{kStaticRows} : std::vector<size_t>{3, 64, 700};
    for (const size_t rows : row_counts) {
        tests::random_generator input_rg;
        input_rg.set_seed(static_cast<uint32_t>(rows));
        const auto residual = random_vec(input_rg, rows * CD, 1.0f);
        ov::Tensor residual_tensor(et, ov::Shape{rows, CD});
        fill(residual_tensor, residual);
        infer_request.set_tensor(residual_p, residual_tensor);

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
        reference_read(p, residual, gain_folded, w_down, w_up, with_write_gate ? &w_inject : nullptr, ref_mixed, ref_gate);

        OV_ASSERT_NO_THROW(infer_request.infer());
        const auto mixed_tensor = infer_request.get_output_tensor(0);
        const auto* mixed_data = mixed_tensor.data<ov::float16>();
        ASSERT_EQ(mixed_tensor.get_size(), ref_mixed.size());
        for (size_t i = 0; i < ref_mixed.size(); i++) {
            ASSERT_NEAR(static_cast<float>(mixed_data[i]), ref_mixed[i], 2e-2f) << "mixed mismatch at rows=" << rows << " index=" << i;
        }
        if (with_write_gate) {
            const auto gate_tensor = infer_request.get_output_tensor(1);
            const auto* gate_data = gate_tensor.data<ov::float16>();
            ASSERT_EQ(gate_tensor.get_size(), ref_gate.size());
            for (size_t i = 0; i < ref_gate.size(); i++) {
                ASSERT_NEAR(static_cast<float>(gate_data[i]), ref_gate[i], 2e-2f) << "write gate mismatch at rows=" << rows << " index=" << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, gated_residual_decomposed_read_test, ::testing::Combine(::testing::Bool(), ::testing::Bool()));

}  // namespace
