// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Conformance tests for the CM u2 weight-only-quantized fully connected implementation
// (impls/cm/fully_connected_woq_u2.cpp, kernel impls/cm/woq_u2_gemm_dual.cm). The kernel takes u8
// per-group zero points, so every case uses those (other zero-point kinds are rejected by validate_impl
// and fall back to the OCL reference); the output is f16 (OUT_F16, what an f16 model uses) or f32.
//
// Every conformance case runs in both of the kernel's weight layouts and both output types (f16 / f32):
//   nmajor : weights [N, K/4], scales / zero points [N, K/64] -- what a compressed FC provides;
//   gmajor : weights [K/64, N, 16 B], scales / zero points [K/64, N] -- the kernel's group-major path,
//            selected through woq_u2_weight_layout_for_tests() with the buffers filled group-major.
// Each case forces the CM implementation, checks that it was actually selected, and compares its output
// with a host reference that uses the kernel's arithmetic (see host_reference(); for f16 output also its
// final rounding to fp16), see compare_values() for the pass criteria.
//
//   smoke_*        : small / edge shapes (M tails, partial column tiles, odd group counts, 3D input,
//                    dynamic M, two u2 layers in one program)
//   perf_shapes_*  : the u2 oneDNN benchdnn matmul cases (bd<n>, numbered as in the benchdnn list), with
//                    their exact input rank, shapes, output type and
//                    post-op (bd2 / bd5: add f32 [M, N]; bd8 / bd11: add f32 per column; bd4: swish x mul f16
//                    [M, N]; bd10: swish x mul f16 per column), in both weight layouts. On M = 512 the
//                    reference computes 8 rows per 256-row tile (all N, all K). Not part of the smoke run;
//                    select them with --gtest_filter=perf_shapes*fully_connected_cm_woq_u2*
//
// Post-ops (Epi): FC bias [N]; a fused eltwise sum with an operand; a fused swish (beta 1) + eltwise prod
// (SwiGLU). The operand has the output's shape and dtype; a per-column operand / bias ([.., 1, N], epi_row)
// only with M = 1, where it is that shape (the kernels do not broadcast). A trailing reorder keeps the post-op from being the network output, so
// prepare_primitive_fusing fuses it into the FC; every post-op case checks it ran inside the CM kernel.

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <sstream>
#include <thread>

#include "fully_connected_inst.h"
#include "impls/cm/fully_connected_woq_u2.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;
using ov::intel_gpu::cm::WoqU2WeightLayout;
using ov::intel_gpu::cm::woq_u2_weight_layout_for_tests;

namespace {

enum class ZpKind { none, tensor_f16, tensor_u8, scalar };

// Post-op of a case: none; FC bias [N] (M = 1 only); fused add (fc + t); fused SwiGLU (swish(fc) * t). t is
// an input shaped like the output, or one row (epi_row, M = 1 only).
enum class Epi { none, bias, add, swiglu };

struct WoqU2Case {
    int64_t M, K, N;
    ZpKind zp;
    int64_t seq = 0;         // > 0: 3D input [M/seq, seq, K]
    bool dynamic = false;    // dynamic M; executed at M and at two more sizes
    bool sample_rows = false;  // reference computes a row sample only (all N columns, all of K)
    Epi epi = Epi::none;
    bool epi_row = false;    // add / SwiGLU operand is one row [.., 1, N] (per column; M = 1 only)
    int bd = 0;              // > 0: number of the benchdnn case it reproduces
};

std::string zp_name(ZpKind z) {
    switch (z) {
    case ZpKind::none: return "nozp";
    case ZpKind::tensor_f16: return "zpf16";
    case ZpKind::tensor_u8: return "zpu8";
    default: return "zpscalar";
    }
}

std::string layout_name(WoqU2WeightLayout l) {
    return l == WoqU2WeightLayout::n_major ? "nmajor" : "gmajor";
}

// (case, weight layout, output type): every conformance case runs in all four combinations.
using WoqU2Param = std::tuple<WoqU2Case, WoqU2WeightLayout, data_types>;

std::string case_name(const testing::TestParamInfo<WoqU2Param>& info) {
    const auto& c = std::get<0>(info.param);
    std::string name = (c.bd ? "bd" + std::to_string(c.bd) + "_" : std::string()) + "M" + std::to_string(c.M) + "_K" + std::to_string(c.K) + "_N" + std::to_string(c.N) + "_" + zp_name(c.zp);
    if (c.seq) name += "_seq" + std::to_string(c.seq);
    if (c.dynamic) name += "_dyn";
    if (c.epi == Epi::bias) name += "_bias";
    if (c.epi == Epi::add) name += c.epi_row ? "_addrow" : "_add";
    if (c.epi == Epi::swiglu) name += c.epi_row ? "_swiglurow" : "_swiglu";
    return name + "_" + layout_name(std::get<1>(info.param)) + (std::get<2>(info.param) == data_types::f32 ? "_f32out" : "_f16out");
}

bool cm_woq_u2_supported(engine& e) {
    const auto& info = e.get_device_info();
    auto config = get_test_default_config(e);
    return (info.arch == gpu_arch::xe2 || info.arch == gpu_arch::xe3) && info.max_local_mem_size >= 98304 &&
           check_cm_jit_support(e, config);
}

// Constant inputs of one u2 FC. Host copies are always N-major: packed weights [N, K/4] (4 per byte, LSB
// first along K), scales / zero points [N, K/64]. The device buffers hold them in `layout`.
struct WoqU2Weights {
    memory::ptr weights, scale, zp;
    float zp_scalar = 1.5f;
    int64_t K = 0, N = 0;
    std::vector<uint8_t> packed;
    std::vector<ov::float16> scales;
    std::vector<uint8_t> zp_u8;  // empty: no zero point
};

// Selects the kernel's weight layout for the lifetime of the scope (network build and execution).
struct WeightLayoutScope {
    WoqU2WeightLayout saved;
    explicit WeightLayoutScope(WoqU2WeightLayout l) : saved(woq_u2_weight_layout_for_tests()) {
        woq_u2_weight_layout_for_tests() = l;
    }
    ~WeightLayoutScope() {
        woq_u2_weight_layout_for_tests() = saved;
    }
};

// Fast deterministic 64-bit generator (splitmix64): the perf shapes need ~10^8 random values, which
// per-element std:: distributions make the dominant cost of the whole suite.
struct SplitMix64 {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

// Activations uniform on the grid k/1024 in [-1, 1), generated from a 2048-entry fp16 table.
std::vector<ov::float16> random_activations(size_t count, uint64_t seed) {
    static const std::vector<ov::float16> table = [] {
        std::vector<ov::float16> t(2048);
        for (int i = 0; i < 2048; i++)
            t[static_cast<size_t>(i)] = ov::float16(static_cast<float>(i - 1024) / 1024.0f);
        return t;
    }();
    std::vector<ov::float16> v(count);
    SplitMix64 rng{seed};
    size_t i = 0;
    while (i < count) {
        uint64_t bits = rng.next();
        for (int j = 0; j < 5 && i < count; j++, bits >>= 11)  // five 11-bit indices per draw
            v[i++] = table[static_cast<size_t>(bits & 2047u)];
    }
    return v;
}

WoqU2Weights make_weights(engine& engine, int64_t K, int64_t N, ZpKind zk, tests::random_generator& rg, uint32_t seed,
                          WoqU2WeightLayout layout = WoqU2WeightLayout::n_major) {
    const int64_t KG = K / 64;
    WoqU2Weights r;
    r.K = K;
    r.N = N;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> q_dist(0, 3);
    // Uniformly random bytes are uniformly random u2 values (4 per byte, LSB first along K).
    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 4));
    SplitMix64 rng{seed};
    for (size_t i = 0; i < packed.size(); i += 8) {
        const uint64_t bits = rng.next();
        for (size_t j = 0; j < 8 && i + j < packed.size(); j++)
            packed[i + j] = static_cast<uint8_t>(bits >> (8 * j));
    }

    auto scales = rg.generate_random_1d<ov::float16>(N * KG, 0.005f, 0.025f, 10000);
    std::vector<ov::float16> zp_f16;
    std::vector<uint8_t> zp_u8;
    if (zk == ZpKind::tensor_f16)
        zp_f16 = rg.generate_random_1d<ov::float16>(N * KG, 0.0f, 3.0f, 1000);  // non-integer zero points
    if (zk == ZpKind::tensor_u8) {
        zp_u8.resize(static_cast<size_t>(N * KG));
        for (auto& v : zp_u8)
            v = static_cast<uint8_t>(q_dist(gen));
    }

    OPENVINO_ASSERT(layout == WoqU2WeightLayout::n_major || zk == ZpKind::tensor_u8, "group-major weights need u8 zero points");
    r.packed = packed;
    r.scales = scales;
    r.zp_u8 = zp_u8;
    if (layout == WoqU2WeightLayout::group_major) {
        // Same byte count, group-major order: Wq[g][n][16 B], scales / zero points [g][n].
        std::vector<uint8_t> packed_g(packed.size()), zp_g(zp_u8.size());
        std::vector<ov::float16> scales_g(scales.size());
        for (int64_t n = 0; n < N; n++)
            for (int64_t g = 0; g < KG; g++) {
                std::copy_n(packed.begin() + n * (K / 4) + g * 16, 16, packed_g.begin() + (g * N + n) * 16);
                scales_g[static_cast<size_t>(g * N + n)] = scales[static_cast<size_t>(n * KG + g)];
                zp_g[static_cast<size_t>(g * N + n)] = zp_u8[static_cast<size_t>(n * KG + g)];
            }
        packed = packed_g;
        scales = scales_g;
        zp_u8 = zp_g;
    }

    r.weights = engine.allocate_memory({{N, K}, data_types::u2, format::bfyx});
    r.scale = engine.allocate_memory({{N, KG}, data_types::f16, format::bfyx});
    set_values(r.weights, packed);
    set_values(r.scale, scales);
    if (zk == ZpKind::tensor_f16) {
        r.zp = engine.allocate_memory({{N, KG}, data_types::f16, format::bfyx});
        set_values(r.zp, zp_f16);
    } else if (zk == ZpKind::tensor_u8) {
        r.zp = engine.allocate_memory({{N, KG}, data_types::u8, format::bfyx});
        set_values(r.zp, zp_u8);
    }

    return r;
}

void add_weights(topology& topo, const std::string& prefix, const WoqU2Weights& w) {
    topo.add(data(prefix + "weights", w.weights));
    topo.add(data(prefix + "scale", w.scale));
    if (w.zp)
        topo.add(data(prefix + "zp", w.zp));
}

fully_connected make_fc(const std::string& id, const std::string& input, const std::string& prefix, ZpKind zk, data_types out_dt,
                        size_t input_rank, float zp_scalar, bool with_bias = false) {
    const bool zp_tensor = zk == ZpKind::tensor_f16 || zk == ZpKind::tensor_u8;
    auto fc = fully_connected(id, input_info(input), prefix + "weights", with_bias ? prefix + "bias" : "", prefix + "scale",
                              zp_tensor ? prefix + "zp" : "", out_dt, input_rank, 2);
    if (zk == ZpKind::scalar)
        fc.decompression_zero_point_scalar = zp_scalar;
    return fc;
}

ExecutionConfig woq_config(engine& engine, const std::vector<std::string>& ids) {
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplForcingMap forcing;
    for (const auto& id : ids)
        forcing[id] = ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::cm};
    config.set_property(ov::intel_gpu::force_implementations(forcing));
    config.set_user_property(ov::hint::dynamic_quantization_group_size(0));
    return config;
}

void expect_selected(network& net, const std::string& id) {
    auto impl = net.get_primitive(id)->get_impl();
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->get_kernel_name().find("woq_u2"), std::string::npos)
        << "CM u2 FC implementation was not selected for " << id << " (got " << impl->get_kernel_name() << ")";
}

// First `count` elements of an f16 / f32 output (dynamic-shape outputs may be over-allocated).
std::vector<float> read_output(memory::ptr mem, size_t count, bool out_f32) {
    std::vector<float> out(count);
    if (out_f32) {
        cldnn::mem_lock<float, mem_lock_type::read> lk(mem, get_test_stream());
        OPENVINO_ASSERT(lk.size() >= count, "output smaller than expected");
        std::copy(lk.begin(), lk.begin() + count, out.begin());
    } else {
        cldnn::mem_lock<ov::float16, mem_lock_type::read> lk(mem, get_test_stream());
        OPENVINO_ASSERT(lk.size() >= count, "output smaller than expected");
        for (size_t i = 0; i < count; i++)
            out[i] = static_cast<float>(lk[i]);
    }
    return out;
}

// Host reference for the given rows of A ([*, K] f16, row-major), all N columns, with the CM kernel's
// arithmetic: each weight is dequantized to fp16 as the kernel does -- one round-to-nearest-even of
// (q - zp) * scale, where q - zp is a small exact integer -- and the fp16 x fp16 products (exact in f32)
// are summed in double. The only difference left to the kernel is the order of its f32 summation.
std::vector<float> host_reference(const std::vector<ov::float16>& a, const std::vector<int64_t>& rows, const WoqU2Weights& w) {
    const int64_t K = w.K, N = w.N, KG = K / 64, R = static_cast<int64_t>(rows.size());
    std::vector<float> a32(static_cast<size_t>(R * K));
    for (int64_t i = 0; i < R; i++)
        for (int64_t k = 0; k < K; k++)
            a32[static_cast<size_t>(i * K + k)] = static_cast<float>(a[static_cast<size_t>(rows[static_cast<size_t>(i)] * K + k)]);
    std::vector<float> out(static_cast<size_t>(R * N));
    auto columns = [&](int64_t n0, int64_t n1) {
        std::vector<float> deq(static_cast<size_t>(K));
        for (int64_t n = n0; n < n1; n++) {
            for (int64_t k = 0; k < K; k++) {
                const int q = (w.packed[static_cast<size_t>(n * (K / 4) + k / 4)] >> (2 * (k % 4))) & 3;
                const size_t gi = static_cast<size_t>(n * KG + k / 64);
                const int zp = w.zp_u8.empty() ? 0 : w.zp_u8[gi];
                deq[static_cast<size_t>(k)] = static_cast<float>(ov::float16(static_cast<float>(q - zp) * static_cast<float>(w.scales[gi])));
            }
            for (int64_t i = 0; i < R; i++) {
                const float* ar = a32.data() + i * K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; k++)
                    acc += static_cast<double>(ar[k]) * deq[static_cast<size_t>(k)];
                out[static_cast<size_t>(i * N + n)] = static_cast<float>(acc);
            }
        }
    };
    const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(static_cast<int64_t>(std::thread::hardware_concurrency()), N / 32));
    std::vector<std::thread> pool;
    for (int64_t t = 0; t < nt; t++)
        pool.emplace_back(columns, N * t / nt, N * (t + 1) / nt);
    for (auto& th : pool)
        th.join();
    return out;
}

// Pass criterion, CM output vs host reference, the same threshold for both output types:
//   f32 output: |CM - ref| <= max_abs_diff per element. Only the order of the f32 summation differs from
//               the reference (measured <= 1.5e-5 up to K = 17920); 1e-4 is OpenVINO's functional-test
//               f32 threshold.
//   f16 output: the kernel rounds its f32 result to fp16 (nearest even), so the same bound is applied
//               before that rounding: the CM value must be the fp16 rounding of some value within
//               max_abs_diff of ref, i.e. ref must lie within max_abs_diff of the CM value's rounding
//               interval (the reals that round to it). That distance is the smallest f32 error that could
//               have produced the CM value -- the kernel's own f32 result is never visible. A result next to
//               a rounding midpoint may therefore land on either neighbouring fp16 value (the two sides
//               differ only by the summation noise); at most max_midpoint_fraction of the elements may do
//               so (measured <= 0.29 %; a subtle dequant defect measured 9-37 %).
constexpr float max_abs_diff = 1e-4f;
constexpr double max_midpoint_fraction = 0.01;

// Distance from x to the interval of reals that round (to nearest) to the fp16 value h; 0 if x is inside.
float distance_to_fp16_rounding_interval(float x, ov::float16 h) {
    const uint16_t b = h.to_bits();
    const float v = static_cast<float>(h);
    // Neighbouring fp16 values below and above h (sign-magnitude encoding; +-0 neighbours are +-2^-24).
    auto neighbour = [&](bool up) {
        const bool neg = (b & 0x8000) != 0;
        const uint16_t mag = b & 0x7fff;
        if (mag == 0)
            return up ? 1.0f / (1 << 24) : -1.0f / (1 << 24);
        const bool away = up != neg;  // moving away from zero
        return static_cast<float>(ov::float16::from_bits(static_cast<uint16_t>((b & 0x8000) | (away ? mag + 1 : mag - 1))));
    };
    const float lo = 0.5f * (v + neighbour(false)), hi = 0.5f * (v + neighbour(true));
    return x < lo ? lo - x : (x > hi ? x - hi : 0.0f);
}

// CM output vs the host reference, row-major [rows, N]; `row_ids` names the output row of each compared
// row. `ref` is always the unrounded reference.
void compare_values(const std::vector<float>& out, const std::vector<float>& ref, int64_t N, const std::vector<int64_t>* row_ids = nullptr,
                    bool out_f16 = false) {
    size_t bad = 0, first_bad = 0, differing = 0;
    float max_err = 0.0f;
    for (size_t i = 0; i < out.size(); i++) {
        // f32: distance to ref; f16: distance from ref to the values that round to the CM value.
        const float err = out_f16 ? distance_to_fp16_rounding_interval(ref[i], ov::float16(out[i])) : std::fabs(out[i] - ref[i]);
        max_err = std::max(max_err, err);
        if (out_f16 && out[i] != static_cast<float>(ov::float16(ref[i])))
            differing++;
        if (!(err <= max_abs_diff)) {
            if (bad++ == 0)
                first_bad = i;
        }
    }
    if (out_f16)
        std::cout << "        [cm vs host ref, f16] min possible f32 error = " << max_err << " (limit " << max_abs_diff << "), "
                  << differing << " / " << out.size() << " at a rounding midpoint (" << 100.0 * static_cast<double>(differing) / static_cast<double>(out.size())
                  << " %, allowed <= " << 100.0 * max_midpoint_fraction << " %)" << "\n";
    else
        std::cout << "        [cm vs host ref, f32] max|d| = " << max_err << " (limit " << max_abs_diff << ")" << "\n";
    const int64_t bad_row = row_ids ? (*row_ids)[static_cast<size_t>(first_bad / N)] : static_cast<int64_t>(first_bad / N);
    ASSERT_EQ(bad, size_t(0)) << bad << " of " << out.size() << " compared outputs are more than " << max_abs_diff
                              << (out_f16 ? " from any value that rounds to them" : " off") << "; first at row " << bad_row << ", col "
                              << first_bad % N << ": got " << out[first_bad] << ", reference " << ref[first_bad];
    if (out_f16)
        ASSERT_LE(static_cast<double>(differing), max_midpoint_fraction * static_cast<double>(out.size()))
            << differing << " of " << out.size() << " f16 outputs differ from the fp16-rounded reference (more than "
            << 100.0 * max_midpoint_fraction << " %): too many for rounding-midpoint cases";
}

// CM output rows `rows` of an [m, N] output vs the host reference.
// Host epilogue: kind, operand values (row-major, [m, N] or one row [N] when `row`).
struct HostEpi {
    Epi kind = Epi::none;
    const std::vector<float>* t = nullptr;
    bool row = false;
};

// The reference applies the epilogue in double, in the kernel's order: acc + t, or acc / (1 + exp(-acc)) * t.
void check_rows(memory::ptr out_mem, int64_t m, const std::vector<ov::float16>& a, const std::vector<int64_t>& rows, const WoqU2Weights& w,
                bool out_f32, const HostEpi& epi = {}) {
    const int64_t N = w.N, R = static_cast<int64_t>(rows.size());
    const auto out_all = read_output(out_mem, static_cast<size_t>(m * N), out_f32);
    std::vector<float> out_rows(static_cast<size_t>(R * N));
    for (int64_t i = 0; i < R; i++)
        std::copy_n(out_all.begin() + rows[static_cast<size_t>(i)] * N, N, out_rows.begin() + i * N);
    auto ref = host_reference(a, rows, w);
    if (epi.kind != Epi::none) {
        for (int64_t i = 0; i < R; i++)
            for (int64_t n = 0; n < N; n++) {
                const int64_t trow = epi.row ? 0 : rows[static_cast<size_t>(i)];
                const double t = (*epi.t)[static_cast<size_t>(trow * N + n)];
                double v = ref[static_cast<size_t>(i * N + n)];
                v = epi.kind == Epi::swiglu ? v / (1.0 + std::exp(-v)) * t : v + t;
                ref[static_cast<size_t>(i * N + n)] = static_cast<float>(v);
            }
    }
    compare_values(out_rows, ref, N, &rows, !out_f32);
}

std::vector<int64_t> all_rows(int64_t M) {
    std::vector<int64_t> rows(static_cast<size_t>(M));
    for (int64_t i = 0; i < M; i++)
        rows[static_cast<size_t>(i)] = i;
    return rows;
}

// Rows compared on the large shapes: 8 per 256-row work-group tile, one in each of the tile's 8
// row-threads (32 rows each) at a different offset inside the thread's 4 x 8-row blocks, plus the last
// row. All N columns and all of K are always checked.
std::vector<int64_t> sample_rows(int64_t M) {
    std::vector<int64_t> rows;
    for (int64_t base = 0; base < M; base += 256)
        for (int64_t t = 0; t < 8; t++) {
            const int64_t r = base + t * 32 + (t * 5) % 32;
            if (r < M)
                rows.push_back(r);
        }
    if (rows.back() != M - 1)
        rows.push_back(M - 1);
    return rows;
}

class fully_connected_cm_woq_u2 : public ::testing::TestWithParam<WoqU2Param> {};

TEST_P(fully_connected_cm_woq_u2, conformance) {
    auto& engine = get_test_engine();
    if (!cm_woq_u2_supported(engine))
        GTEST_SKIP() << "CM u2 FC requires Xe2 / Xe3 with CM JIT support and >= 96 KB SLM";

    const auto p = std::get<0>(GetParam());
    const auto wl = std::get<1>(GetParam());
    const auto out_dt = std::get<2>(GetParam());
    const bool out_f32 = out_dt == data_types::f32;
    const WeightLayoutScope layout_scope(wl);
    const int64_t M = p.M, K = p.K, N = p.N;
    ASSERT_EQ(K % 64, 0);
    ASSERT_EQ(N % 32, 0);
    if (p.seq)
        ASSERT_EQ(M % p.seq, 0);

    tests::random_generator rg(GET_SUITE_NAME);
    const auto w = make_weights(engine, K, N, p.zp, rg, static_cast<uint32_t>(M * 131 + K * 7 + N), wl);

    const size_t input_rank = p.seq ? 3 : 2;
    auto make_shape = [&](int64_t m) {
        return p.seq ? ov::PartialShape{m / p.seq, p.seq, K} : ov::PartialShape{m, K};
    };
    auto net_in_shape = p.dynamic ? (p.seq ? ov::PartialShape{-1, p.seq, K} : ov::PartialShape{-1, K}) : make_shape(M);

    // The host reference computes a row sample only on the large shapes (all N columns, all of K).
    if (p.sample_rows)
        ASSERT_FALSE(p.dynamic);
    const bool fused_post_op = p.epi == Epi::add || p.epi == Epi::swiglu;
    auto make_out_shape = [&](int64_t m) {
        return p.seq ? ov::PartialShape{m / p.seq, p.seq, N} : ov::PartialShape{m, N};
    };
    // Shape of the add / SwiGLU operand: the output's, or one row [.., 1, N].
    auto make_epi_shape = [&](int64_t m) {
        if (p.epi_row)
            return p.seq ? ov::PartialShape{1, 1, N} : ov::PartialShape{1, N};
        return make_out_shape(m);
    };
    auto net_epi_shape = p.epi_row ? make_epi_shape(1)
                                   : (p.dynamic ? (p.seq ? ov::PartialShape{-1, p.seq, N} : ov::PartialShape{-1, N}) : make_out_shape(M));
    // Operand / bias values: f16-representable, so the same values serve an f16 and an f32 tensor.
    auto make_epi_values = [&](size_t count, uint64_t seed) {
        const auto h = random_activations(count, seed);
        std::vector<float> v(count);
        for (size_t i = 0; i < count; i++)
            v[i] = static_cast<float>(h[i]);
        return v;
    };
    auto upload = [&](const ov::PartialShape& shape, const std::vector<float>& v) {
        auto mem = engine.allocate_memory({shape, out_dt, format::bfyx});
        if (out_dt == data_types::f32) {
            set_values(mem, v);
        } else {
            std::vector<ov::float16> h(v.size());
            for (size_t i = 0; i < v.size(); i++)
                h[i] = ov::float16(v[i]);
            set_values(mem, h);
        }
        return mem;
    };
    std::vector<float> bias_values;
    if (p.epi == Epi::bias)
        bias_values = make_epi_values(static_cast<size_t>(N), static_cast<uint64_t>(N * 31 + K));

    auto build = [&](const ov::PartialShape& in_shape) {
        topology topo(input_layout("input", layout{in_shape, data_types::f16, format::bfyx}));
        add_weights(topo, "", w);
        if (p.epi == Epi::bias)
            topo.add(data("bias", upload(ov::PartialShape{1, N}, bias_values)));
        topo.add(make_fc("fc_prim", "input", "", p.zp, out_dt, input_rank, w.zp_scalar, p.epi == Epi::bias));
        std::string last = "fc_prim";
        if (fused_post_op) {
            topo.add(input_layout("epi", layout{net_epi_shape, out_dt, format::bfyx}));
            if (p.epi == Epi::swiglu) {
                // gate_proj-style SwiGLU: swish(fc) * t.
                topo.add(activation("swish", input_info("fc_prim"), activation_func::swish, {1.0f, 0.0f}));
                topo.add(eltwise("post", input_info("swish"), input_info("epi"), eltwise_mode::prod));
            } else {
                topo.add(eltwise("post", input_info("fc_prim"), input_info("epi"), eltwise_mode::sum));
            }
            last = "post";
        }
        // Keeps the post-op from being the network output (outputs are not fused); removed later as redundant.
        topo.add(reorder("out", input_info(last), format::bfyx, out_dt));
        return topo;
    };
    network::ptr network = get_network(engine, build(net_in_shape), woq_config(engine, {"fc_prim"}), get_test_stream_ptr(), false);

    std::vector<int64_t> run_ms = {M};
    if (p.dynamic) {
        const int64_t step = p.seq ? p.seq : 1;
        run_ms.push_back(std::max<int64_t>(step, (M / 3) / step * step));
        run_ms.push_back(M + 8 * step);
    }

    for (const int64_t m : run_ms) {
        SCOPED_TRACE("M = " + std::to_string(m));
        auto a = random_activations(static_cast<size_t>(m * K), static_cast<uint64_t>(m * 1000003 + K));
        auto input_mem = engine.allocate_memory({make_shape(m), data_types::f16, format::bfyx});
        set_values(input_mem, a);
        network->set_input_data("input", input_mem);
        std::vector<float> epi_values;
        if (fused_post_op) {
            const auto shape = make_epi_shape(m);
            epi_values = make_epi_values(static_cast<size_t>(ov::shape_size(shape.to_shape())), static_cast<uint64_t>(m * 7919 + N));
            network->set_input_data("epi", upload(shape, epi_values));
        }
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        // The CM FC must have run. Its id depends on graph cleanup (with the trailing reorder removed it
        // takes the output's id), so look for it among the executed primitives. A post-op must run inside
        // it (epi_mode 1 / 2), not as a separate kernel.
        bool cm_ran = false;
        for (const auto& id : network->get_executed_primitive_ids()) {
            const auto impl = network->get_primitive(id)->get_impl();
            cm_ran |= impl && impl->get_kernel_name().find("woq_u2") != std::string::npos;
            ASSERT_TRUE(id != "swish" && id != "post") << "post-op was not fused into the CM FC (" << id << " ran separately)";
        }
        ASSERT_TRUE(cm_ran) << "CM u2 FC implementation was not selected";
        HostEpi he;
        he.kind = p.epi;
        he.row = p.epi == Epi::bias || p.epi_row;
        he.t = p.epi == Epi::bias ? &bias_values : &epi_values;
        check_rows(outputs.at("out").get_memory(), m, a, p.sample_rows ? sample_rows(m) : all_rows(m), w, out_f32, he);
    }
}

// Performance of the CM implementation on the benchmark shapes of test_woq_cm.py, measured with GPU
// profiling events (kernel execution time of the FC only), for every weight layout and output type.
// Not a conformance check and disabled by default; run with
//   --gtest_also_run_disabled_tests --gtest_filter=*fully_connected_cm_woq_u2_perf*
// Environment: OV_WOQ_U2_PERF_ITERS (CM runs per shape, default 100); OV_WOQ_U2_PERF_SHAPES replaces the
// built-in shapes with a list of MxKxN, separated by ',' (e.g. "1x5120x7680,128x5120x17920"; K % 64 == 0,
// N % 32 == 0).
// Device-resident copy: test buffers default to the lockable allocation type (host USM on a discrete
// GPU), while a compiled model keeps constants and activations in device memory.
memory::ptr to_device(engine& engine, const memory::ptr& m) {
    if (!m)
        return nullptr;
    auto d = engine.allocate_memory(m->get_layout(), allocation_type::usm_device, false);
    d->copy_from(get_test_stream(), *m, true);
    return d;
}

TEST(fully_connected_cm_woq_u2_perf, DISABLED_benchmark_shapes) {
    auto& engine = get_test_engine();
    if (!cm_woq_u2_supported(engine))
        GTEST_SKIP() << "CM u2 FC requires Xe2 / Xe3 with CM JIT support and >= 96 KB SLM";

    auto env_int = [](const char* name, int def) {
        const char* v = std::getenv(name);
        return v ? std::max(0, std::atoi(v)) : def;
    };
    const int cm_iters = std::max(1, env_int("OV_WOQ_U2_PERF_ITERS", 100));

    // XMX roof: EUs x clock x 256 fp16 flop/clk/EU (Xe2). On Arc B580 this is the 116.7 TFLOP/s measured
    // with a DPAS-only probe.
    const auto& info = engine.get_device_info();
    const double peak_tflops = static_cast<double>(info.execution_units_count) * info.gpu_frequency * 1e-3 * 256.0 * 1e-3;

    struct Shape {
        int64_t M, K, N;
        ZpKind zp;
    };
    std::vector<Shape> shapes = {{512, 5120, 7680, ZpKind::tensor_u8},
                                       {512, 5120, 5120, ZpKind::tensor_u8},
                                       {512, 5120, 17920, ZpKind::tensor_u8},
                                       {512, 17920, 5120, ZpKind::tensor_u8},
                                       {1024, 5120, 7680, ZpKind::tensor_u8},
                                       {1024, 5120, 17920, ZpKind::tensor_u8},
                                       {2048, 5120, 7680, ZpKind::tensor_u8},
                                       {2048, 5120, 17920, ZpKind::tensor_u8},
                                       {2048, 17920, 5120, ZpKind::tensor_u8}};
    if (const char* env = std::getenv("OV_WOQ_U2_PERF_SHAPES")) {
        shapes.clear();
        std::stringstream list(env);
        std::string item;
        while (std::getline(list, item, ',')) {
            int64_t m = 0, k = 0, n = 0;
            char x1 = 0, x2 = 0;
            std::stringstream one(item);
            ASSERT_TRUE((one >> m >> x1 >> k >> x2 >> n) && x1 == 'x' && x2 == 'x' && m > 0 && k > 0 && n > 0)
                << "OV_WOQ_U2_PERF_SHAPES: expected MxKxN, got '" << item << "'";
            ASSERT_TRUE(k % 64 == 0 && n % 32 == 0) << "OV_WOQ_U2_PERF_SHAPES: " << item << " needs K % 64 == 0 and N % 32 == 0";
            shapes.push_back({m, k, n, ZpKind::tensor_u8});
        }
        ASSERT_FALSE(shapes.empty()) << "OV_WOQ_U2_PERF_SHAPES is empty";
    }

    // Kernel execution times (ms) of the FC over `iters` runs after `warmup` runs.
    auto measure = [&](const WoqU2Weights& w, ZpKind zk, data_types out_dt, memory::ptr input_mem, int64_t M, int64_t K, const ExecutionConfig& cfg,
                       int warmup, int iters, std::string& impl_name) {
        topology topo(input_layout("input", layout{ov::PartialShape{M, K}, data_types::f16, format::bfyx}));
        add_weights(topo, "", w);
        topo.add(make_fc("fc_prim", "input", "", zk, out_dt, 2, w.zp_scalar));
        auto config = cfg;
        config.set_property(ov::enable_profiling(true));
        network net(engine, topo, config);
        net.set_input_data("input", input_mem);
        std::vector<double> ms;
        for (int i = 0; i < warmup + iters; i++) {
            auto outputs = net.execute();
            outputs.at("fc_prim").get_memory();  // waits for completion
            if (i < warmup)
                continue;
            auto ev = net.get_executed_primitives().at("fc_prim");
            for (const auto& interval : ev->get_profiling_info())
                if (interval.stage == instrumentation::profiling_stage::executing)
                    ms.push_back(static_cast<double>(interval.value->value().count()) * 1e-6);
        }
        impl_name = net.get_primitive("fc_prim")->get_impl()->get_kernel_name();
        std::sort(ms.begin(), ms.end());
        return ms;
    };

    std::cout << "\n  XMX roof " << std::fixed << std::setprecision(1) << peak_tflops << " TFLOP/s (" << info.execution_units_count << " EUs x "
              << info.gpu_frequency << " MHz x 256). CM: " << cm_iters << " runs/shape.\n"
              << "      M      K      N zp   layout out |  CM best  CM median  TFLOP/s  %roof\n";
    tests::random_generator rg(GET_SUITE_NAME);
    for (const auto& s : shapes) {
      for (const auto wl : {WoqU2WeightLayout::n_major, WoqU2WeightLayout::group_major}) {
       for (const auto out_dt : {data_types::f16, data_types::f32}) {
        const WeightLayoutScope layout_scope(wl);
        auto w = make_weights(engine, s.K, s.N, s.zp, rg, 5, wl);
        w.weights = to_device(engine, w.weights);
        w.scale = to_device(engine, w.scale);
        w.zp = to_device(engine, w.zp);
        auto a = random_activations(static_cast<size_t>(s.M * s.K), 17);
        auto host_input = engine.allocate_memory({ov::PartialShape{s.M, s.K}, data_types::f16, format::bfyx});
        set_values(host_input, a);
        auto input_mem = to_device(engine, host_input);

        std::string cm_impl;
        const auto cm = measure(w, s.zp, out_dt, input_mem, s.M, s.K, woq_config(engine, {"fc_prim"}), 5, cm_iters, cm_impl);
        ASSERT_FALSE(cm.empty()) << "no profiling data";
        ASSERT_NE(cm_impl.find("woq_u2"), std::string::npos) << "CM implementation not selected (got " << cm_impl << ")";
        const double best = cm.front(), median = cm[cm.size() / 2];
        const double tflops = 2.0 * static_cast<double>(s.M) * s.K * s.N / (best * 1e-3) * 1e-12;
        std::cout << std::setw(7) << s.M << std::setw(7) << s.K << std::setw(7) << s.N << " " << std::left << std::setw(4) << zp_name(s.zp).substr(2)
                  << std::setw(7) << layout_name(wl) << std::setw(4) << (out_dt == data_types::f16 ? "f16" : "f32") << std::right << " | " << std::setprecision(4) << std::setw(8) << best << std::setw(11) << median << std::setprecision(1)
                  << std::setw(9) << tflops << std::setw(6) << 100.0 * tflops / peak_tflops << "%\n" << std::flush;
       }
      }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_cm_woq_u2,
                         ::testing::Combine(::testing::Values(
                             // correctness shapes of test_woq_cm.py (u8 zero points)
                             WoqU2Case{8, 128, 256, ZpKind::tensor_u8},
                             WoqU2Case{5, 128, 256, ZpKind::tensor_u8},
                             WoqU2Case{13, 192, 512, ZpKind::tensor_u8},   // KG = 3 (odd group count)
                             WoqU2Case{16, 256, 256, ZpKind::tensor_u8},
                             WoqU2Case{3, 320, 512, ZpKind::tensor_u8},    // KG = 5
                             WoqU2Case{300, 448, 1024, ZpKind::tensor_u8}, // KG = 7
                             WoqU2Case{517, 5120, 512, ZpKind::tensor_u8},
                             // edge cases: tiny, partial column work-group (N % 256 != 0), 3D input
                             WoqU2Case{1, 64, 32, ZpKind::tensor_u8},
                             WoqU2Case{64, 256, 288, ZpKind::tensor_u8},
                             WoqU2Case{33, 1024, 800, ZpKind::tensor_u8},
                             WoqU2Case{256, 512, 256, ZpKind::tensor_u8},
                             WoqU2Case{24, 384, 512, ZpKind::tensor_u8, 6},
                             // dynamic M (shape-agnostic kernel, re-dispatched per shape)
                             WoqU2Case{96, 512, 512, ZpKind::tensor_u8, 0, true},
                             WoqU2Case{40, 256, 544, ZpKind::tensor_u8, 8, true},
                             // post-ops: tiled kernel (M > 8) and GEMV (M <= 8), operand shaped like the output
                             WoqU2Case{64, 256, 288, ZpKind::tensor_u8, 0, false, false, Epi::add},
                             WoqU2Case{40, 512, 512, ZpKind::tensor_u8, 0, false, false, Epi::swiglu},
                             WoqU2Case{24, 384, 512, ZpKind::tensor_u8, 6, false, false, Epi::add},
                             WoqU2Case{5, 128, 256, ZpKind::tensor_u8, 0, false, false, Epi::add},
                             WoqU2Case{8, 128, 256, ZpKind::tensor_u8, 0, false, false, Epi::swiglu},
                             // per-column operand / bias: M = 1 only
                             WoqU2Case{1, 256, 288, ZpKind::tensor_u8, 0, false, false, Epi::add, true},
                             WoqU2Case{1, 512, 256, ZpKind::tensor_u8, 0, false, false, Epi::swiglu, true},
                             WoqU2Case{1, 64, 32, ZpKind::tensor_u8, 0, false, false, Epi::bias},
                             // post-ops with dynamic M
                             WoqU2Case{96, 512, 512, ZpKind::tensor_u8, 0, true, false, Epi::add},
                             WoqU2Case{40, 256, 544, ZpKind::tensor_u8, 8, true, false, Epi::swiglu}),
                                            ::testing::Values(WoqU2WeightLayout::n_major, WoqU2WeightLayout::group_major),
                                            ::testing::Values(data_types::f16, data_types::f32)),
                         case_name);

// The u2 oneDNN benchdnn matmul cases: src BxMxK : wei 1xKxN, --wtag=cab (weights [N, K]), scales f16 /
// zero points u8 per 64 along K, --attr-fpmath=f16. Input rank and shapes as given (BxM flattened by the
// FC), output type fixed per case; both weight layouts each.
std::vector<WoqU2Param> benchdnn_u2_cases() {
    struct BenchCase {
        WoqU2Case c;
        data_types out;
    };
    const auto u8 = ZpKind::tensor_u8;
    const std::vector<BenchCase> cases = {
        // src : wei, post-op (benchdnn mask 6 = full [M, N] tensor, mask 4 = per column)
        {WoqU2Case{512, 5120, 7680, u8, 1, false, true, Epi::none, false, 1}, data_types::f16},     // 512x1x5120:1x5120x7680
        {WoqU2Case{512, 5120, 5120, u8, 512, false, true, Epi::add, false, 2}, data_types::f32},    // 1x512x5120:1x5120x5120, add f32:6
        {WoqU2Case{512, 5120, 17920, u8, 1, false, true, Epi::none, false, 3}, data_types::f16},    // 512x1x5120:1x5120x17920
        {WoqU2Case{512, 5120, 17920, u8, 512, false, true, Epi::swiglu, false, 4}, data_types::f16},  // 1x512x5120:1x5120x17920, swish+mul f16:6
        {WoqU2Case{512, 17920, 5120, u8, 512, false, true, Epi::add, false, 5}, data_types::f32},   // 1x512x17920:1x17920x5120, add f32:6
        {WoqU2Case{1, 5120, 7680, u8, 1, false, false, Epi::none, false, 7}, data_types::f16},      // 1x1x5120:1x5120x7680
        {WoqU2Case{1, 5120, 5120, u8, 1, false, false, Epi::add, true, 8}, data_types::f32},        // 1x1x5120:1x5120x5120, add f32:4
        {WoqU2Case{1, 5120, 17920, u8, 1, false, false, Epi::none, false, 9}, data_types::f16},     // 1x1x5120:1x5120x17920
        {WoqU2Case{1, 5120, 17920, u8, 1, false, false, Epi::swiglu, true, 10}, data_types::f16},   // 1x1x5120:1x5120x17920, swish+mul f16:4
        {WoqU2Case{1, 17920, 5120, u8, 1, false, false, Epi::add, true, 11}, data_types::f32},      // 1x1x17920:1x17920x5120, add f32:4
    };
    std::vector<WoqU2Param> params;
    for (const auto& bc : cases)
        for (const auto wl : {WoqU2WeightLayout::n_major, WoqU2WeightLayout::group_major})
            params.emplace_back(bc.c, wl, bc.out);
    return params;
}

INSTANTIATE_TEST_SUITE_P(perf_shapes, fully_connected_cm_woq_u2, ::testing::ValuesIn(benchdnn_u2_cases()), case_name);

// Kernel time of the 10 u2 benchdnn cases, one row each, exactly as benchdnn runs them: input rank, shapes,
// output type and post-op of benchdnn_u2_cases(), weights N-major (--wtag=cab), the post-op fused into the
// CM FC; measured with GPU profiling events on the CM FC kernel alone. Reports
// TFLOP/s against the XMX peak and the effective weight bandwidth (u2 weights + f16 scales + u8 zero points,
// 0.297 B per K x N element), the meaningful number for M = 1. Disabled by default; run with
//   --gtest_also_run_disabled_tests --gtest_filter=*fully_connected_cm_woq_u2_perf.DISABLED_benchdnn_cases*
// Environment: OV_WOQ_U2_PERF_ITERS (runs per case, default 100).
TEST(fully_connected_cm_woq_u2_perf, DISABLED_benchdnn_cases) {
    auto& engine = get_test_engine();
    if (!cm_woq_u2_supported(engine))
        GTEST_SKIP() << "CM u2 FC requires Xe2 / Xe3 with CM JIT support and >= 96 KB SLM";
    const char* iters_env = std::getenv("OV_WOQ_U2_PERF_ITERS");
    const int iters = std::max(1, iters_env ? std::atoi(iters_env) : 100);
    const int warmup = 5;
    const auto& info = engine.get_device_info();
    const double peak_tflops = static_cast<double>(info.execution_units_count) * info.gpu_frequency * 1e-3 * 256.0 * 1e-3;

    auto post_name = [](const WoqU2Case& c) -> std::string {
        switch (c.epi) {
        case Epi::bias: return "bias";
        case Epi::add: return c.epi_row ? "add/col" : "add";
        case Epi::swiglu: return c.epi_row ? "swiglu/col" : "swiglu";
        default: return "-";
        }
    };
    std::cout << "\n  XMX roof " << std::fixed << std::setprecision(1) << peak_tflops << " TFLOP/s. " << iters << " runs per case." << "\n"
              << "  bd      M      K      N  out  post-op     kernel |  CM best  CM median  TFLOP/s  %roof  weight GB/s" << "\n";
    tests::random_generator rg(GET_SUITE_NAME);
    for (const auto& prm : benchdnn_u2_cases()) {
        const auto& c = std::get<0>(prm);
        const auto wl = std::get<1>(prm);
        const auto out_dt = std::get<2>(prm);
        if (wl != WoqU2WeightLayout::n_major)
            continue;  // benchdnn's weights are N-major (--wtag=cab)
        const WeightLayoutScope layout_scope(wl);
        const bool fused_post_op = c.epi == Epi::add || c.epi == Epi::swiglu;

        auto w = make_weights(engine, c.K, c.N, c.zp, rg, 5, wl);
        w.weights = to_device(engine, w.weights);
        w.scale = to_device(engine, w.scale);
        w.zp = to_device(engine, w.zp);
        const auto in_shape = c.seq ? ov::PartialShape{c.M / c.seq, c.seq, c.K} : ov::PartialShape{c.M, c.K};
        const auto out_shape = c.seq ? ov::PartialShape{c.M / c.seq, c.seq, c.N} : ov::PartialShape{c.M, c.N};
        const auto epi_shape = c.epi_row ? (c.seq ? ov::PartialShape{1, 1, c.N} : ov::PartialShape{1, c.N}) : out_shape;
        auto device_tensor = [&](const ov::PartialShape& shape, uint64_t seed) {
            const auto h = random_activations(ov::shape_size(shape.to_shape()), seed);
            auto host = engine.allocate_memory({shape, out_dt, format::bfyx});
            if (out_dt == data_types::f32) {
                std::vector<float> f(h.size());
                for (size_t i = 0; i < h.size(); i++)
                    f[i] = static_cast<float>(h[i]);
                set_values(host, f);
            } else {
                set_values(host, h);
            }
            return to_device(engine, host);
        };

        const size_t input_rank = c.seq ? 3 : 2;
        topology topo(input_layout("input", layout{in_shape, data_types::f16, format::bfyx}));
        add_weights(topo, "", w);
        if (c.epi == Epi::bias)
            topo.add(data("bias", device_tensor(ov::PartialShape{1, c.N}, 3)));
        topo.add(make_fc("fc_prim", "input", "", c.zp, out_dt, input_rank, w.zp_scalar, c.epi == Epi::bias));
        std::string last = "fc_prim";
        if (fused_post_op) {
            topo.add(input_layout("epi", layout{epi_shape, out_dt, format::bfyx}));
            if (c.epi == Epi::swiglu) {
                topo.add(activation("swish", input_info("fc_prim"), activation_func::swish, {1.0f, 0.0f}));
                topo.add(eltwise("post", input_info("swish"), input_info("epi"), eltwise_mode::prod));
            } else {
                topo.add(eltwise("post", input_info("fc_prim"), input_info("epi"), eltwise_mode::sum));
            }
            last = "post";
        }
        topo.add(reorder("out", input_info(last), format::bfyx, out_dt));

        auto config = woq_config(engine, {"fc_prim"});
        config.set_property(ov::enable_profiling(true));
        network net(engine, topo, config);
        auto host_in = engine.allocate_memory({in_shape, data_types::f16, format::bfyx});
        set_values(host_in, random_activations(static_cast<size_t>(c.M * c.K), 17));
        net.set_input_data("input", to_device(engine, host_in));
        if (fused_post_op)
            net.set_input_data("epi", device_tensor(epi_shape, 29));

        std::vector<double> ms;
        std::string kernel = "-";
        bool post_op_separate = false;
        for (int i = 0; i < warmup + iters; i++) {
            auto outputs = net.execute();
            outputs.at("out").get_memory();  // waits for completion
            if (i < warmup)
                continue;
            for (const auto& [id, ev] : net.get_executed_primitives()) {
                post_op_separate |= id == "swish" || id == "post";
                const auto impl = net.get_primitive(id)->get_impl();
                if (!impl || impl->get_kernel_name().find("woq_u2") == std::string::npos)
                    continue;
                kernel = impl->get_kernel_name();
                for (const auto& interval : ev->get_profiling_info())
                    if (interval.stage == instrumentation::profiling_stage::executing)
                        ms.push_back(static_cast<double>(interval.value->value().count()) * 1e-6);
            }
        }
        ASSERT_FALSE(ms.empty()) << "bd" << c.bd << ": the CM u2 FC did not run";
        ASSERT_FALSE(post_op_separate) << "bd" << c.bd << ": the post-op was not fused into the CM FC";
        std::sort(ms.begin(), ms.end());
        const double best = ms.front(), median = ms[ms.size() / 2];
        const double tflops = 2.0 * static_cast<double>(c.M) * c.K * c.N / (best * 1e-3) * 1e-12;
        const double weight_gbs = static_cast<double>(c.K) * c.N * (0.25 + 2.0 / 64 + 1.0 / 64) / (best * 1e-3) * 1e-9;
        std::cout << "  " << std::left << std::setw(4) << ("bd" + std::to_string(c.bd)) << std::right << std::setw(5) << c.M << std::setw(7) << c.K
                  << std::setw(7) << c.N << "  " << (out_dt == data_types::f16 ? "f16" : "f32") << "  " << std::left << std::setw(11) << post_name(c)
                  << std::setw(6) << (c.M <= 8 ? "gemv" : "tiled") << std::right << " | " << std::setprecision(4) << std::setw(8) << best
                  << std::setw(11) << median << std::setprecision(1) << std::setw(9) << tflops << std::setw(6) << 100.0 * tflops / peak_tflops << "%"
                  << std::setw(13) << weight_gbs << "\n" << std::flush;
    }
}

}  // namespace
