// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_ternocl_int2.hpp"

#ifdef OV_GPU_WITH_OCL_RT

#    include <array>
#    include <atomic>
#    include <cmath>
#    include <cstdint>
#    include <cstdio>
#    include <cstdlib>
#    include <iostream>
#    include <map>
#    include <mutex>
#    include <sstream>
#    include <unordered_map>
#    include <vector>

#    include "data_inst.h"
#    include "fully_connected_inst.h"
#    include "intel_gpu/runtime/kernel_args.hpp"
#    include "intel_gpu/runtime/memory.hpp"
#    include "openvino/util/env_util.hpp"
#    include "primitive_inst.h"
#    include "registry/registry.hpp"
#    include "reorder_inst.h"
#    include "runtime/ocl/ocl_engine.hpp"
#    include "runtime/ocl/ocl_kernel.hpp"
#    include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

namespace cldnn {
namespace ocl {

namespace {

// OpenVINO stores u2 little-endian inside each byte: value j at bits [2j, 2j+1].
inline uint8_t read_u2(const uint8_t* base, size_t index) {
    return static_cast<uint8_t>((base[index >> 2] >> (2 * (index & 0x3))) & 0x3);
}

// [N, K] u2 codes -> [K/16, N] uint32, re-encoding (code - zp) to the kernel's
// two's-complement {0, +1, -1} = {0, 1, 3}; row 16*kb+j of column n sits at bits [2j, 2j+1].
void pack_weights(const uint8_t* src, uint32_t* dst, size_t N, size_t K, int32_t zp) {
    std::fill(dst, dst + (K / kTernoclPackFactor) * N, 0u);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            const int32_t code = static_cast<int32_t>(read_u2(src, n * K + k)) - zp;
            OPENVINO_ASSERT(code >= -1 && code <= 1, "[GPU] ternocl int2: weight code ", code, " is outside the ternary range {-1, 0, +1}");
            dst[(k / kTernoclPackFactor) * N + n] |= static_cast<uint32_t>(code & 0x3) << (2 * (k % kTernoclPackFactor));
        }
    }
}

// The zero point carries whatever precision the model used, so read it by type.
float read_scalar(const memory::ptr& mem, stream& s) {
    switch (mem->get_layout().data_type) {
    case data_types::f16: {
        mem_lock<uint16_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(ov::float16::from_bits(l.data()[0]));
    }
    case data_types::f32: {
        mem_lock<float, mem_lock_type::read> l{mem, s};
        return l.data()[0];
    }
    case data_types::u8: {
        mem_lock<uint8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(l.data()[0]);
    }
    case data_types::i8: {
        mem_lock<int8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(l.data()[0]);
    }
    case data_types::u2: {
        mem_lock<uint8_t, mem_lock_type::read> l{mem, s};
        return static_cast<float>(read_u2(l.data(), 0));
    }
    default:
        OPENVINO_THROW("[GPU] ternocl int2: unsupported zero point type ", mem->get_layout().data_type);
    }
}

// Constant inputs may be fed through a reorder inserted by the graph optimizer;
// only the data node behind it can be read at compile time.
const program_node* const_source(const program_node* node) {
    while (node != nullptr && !node->is_type<data>()) {
        if (!node->is_type<reorder>() || node->get_dependencies().size() != 1)
            return nullptr;
        node = &node->get_dependency(0);
    }
    return node;
}

// ---------------------------------------------------------------------------
// Tile selection. GEMV (M <= 8): sub-group rows SGM, work-group width WGN
// (16 per sub-group), local k-slicing LS, loads issued ahead U. M-tiled GEMM
// (M > 8): sub-group tile MT_M x MT_N, work-group WG_M x WG_N sub-groups.
// Tuned with TernOCL's bench.sh (>= 2 GiB rotating weights, median of 3).
struct GemvTile {
    int wgn, ls, u;
};
struct MtTile {
    int mt_m, mt_n, wg_m, wg_n;
};

bool env_ints(const char* name, int* v, int n) {
    const char* e = std::getenv(name);
    if (e == nullptr)
        return false;
    std::stringstream ss(e);
    for (int i = 0; i < n; ++i) {
        if (!(ss >> v[i]))
            return false;
        ss.ignore(1, ',');
    }
    return true;
}

GemvTile gemv_tile(size_t K, size_t N, bool integrated) {
    GemvTile t{0, 0, 0};
    if (env_ints("OV_TERNOCL_INT2_GEMV", &t.wgn, 3))  // "wgn,ls,u" for sweeps
        return t;
    struct Entry {
        size_t k, n;
        GemvTile t;
    };
    // Arc Pro B70, M = 1 (TernOCL int2_fp16_upcvt default_tiles).
    static const Entry discrete[] = {
        {4096, 6144, {16, 4, 1}},    // 8B qkv
        {4096, 4096, {32, 8, 1}},    // 8B o_proj
        {4096, 24576, {64, 1, 1}},   // 8B gate_up (merged)
        {12288, 4096, {32, 8, 1}},   // 8B down
        {4096, 151680, {32, 1, 1}},  // 8B lm_head
        {5120, 34816, {32, 8, 1}},   // 27B gate_up (merged)
        {17408, 5120, {16, 6, 1}},   // 27B down
        {5120, 16384, {32, 2, 1}},   // 27B in_proj_qkvz
        {6144, 5120, {16, 6, 1}},    // 27B out_proj / o_proj
        {5120, 14336, {32, 2, 1}},   // 27B qkv
        {5120, 248320, {16, 1, 1}},  // 27B lm_head
    };
    // Arc 140V (Lunar Lake), M = 1, paced sweep.
    static const Entry igpu[] = {
        {5120, 34816, {16, 4, 2}},
        {17408, 5120, {64, 1, 2}},
        {5120, 16384, {16, 1, 2}},
        {6144, 5120, {64, 1, 2}},
        {5120, 14336, {16, 4, 2}},
        {5120, 248320, {32, 2, 2}},
    };
    if (integrated) {
        for (const auto& e : igpu) {
            if (e.k == K && e.n == N)
                return e.t;
        }
    }
    for (const auto& e : discrete) {
        if (e.k == K && e.n == N)
            return e.t;
    }
    return N <= 8192 ? GemvTile{32, 4, 1} : GemvTile{32, 2, 1};
}

MtTile mt_tile(size_t K, size_t N, size_t M, bool integrated) {
    MtTile t{0, 0, 0, 0};
    if (M < 64) {
        if (env_ints("OV_TERNOCL_INT2_MID", &t.mt_m, 4))  // "mt_m,mt_n,wg_m,wg_n"
            return t;
        // Sweep at M = 12 / 20 / 32 / 48 over the 27B shapes; the winners group
        // by output width (narrow N <= 8192, head N >= 65536).
        const int band = M <= 16 ? 0 : (M <= 32 ? 1 : 2);
        static const MtTile narrow[3] = {{16, 16, 2, 2}, {32, 16, 2, 4}, {32, 16, 1, 4}};  // Arc Pro B70
        static const MtTile wide[3] = {{16, 16, 1, 8}, {32, 16, 1, 4}, {64, 16, 1, 8}};
        static const MtTile head[3] = {{16, 16, 1, 4}, {64, 32, 1, 8}, {64, 32, 1, 8}};
        static const MtTile i_narrow[3] = {{16, 32, 1, 4}, {32, 32, 1, 8}, {64, 32, 1, 8}};  // Arc 140V
        static const MtTile i_wide[3] = {{16, 16, 1, 8}, {64, 32, 1, 4}, {64, 32, 1, 4}};
        static const MtTile i_head[3] = {{16, 16, 1, 4}, {32, 16, 1, 8}, {64, 32, 1, 4}};
        if (integrated)
            return N <= 8192 ? i_narrow[band] : (N >= 65536 ? i_head[band] : i_wide[band]);
        return N <= 8192 ? narrow[band] : (N >= 65536 ? head[band] : wide[band]);
    }
    if (env_ints("OV_TERNOCL_INT2_MT", &t.mt_m, 4))
        return t;
    struct Entry {
        size_t k, n;
        MtTile t;
    };
    // Arc Pro B70, M = 1024 (TernOCL int2_fp16_upcvt README).
    static const Entry table[] = {
        {4096, 6144, {64, 32, 4, 4}},
        {4096, 4096, {64, 16, 2, 4}},
        {4096, 24576, {64, 16, 2, 2}},
        {12288, 4096, {64, 32, 4, 2}},
        {4096, 151680, {64, 32, 4, 4}},
        {5120, 34816, {64, 32, 4, 4}},
        {17408, 5120, {32, 32, 1, 8}},
        {5120, 16384, {64, 16, 2, 2}},
        {6144, 5120, {64, 16, 1, 8}},
        {5120, 14336, {64, 32, 4, 4}},
        {5120, 248320, {64, 32, 4, 4}},
    };
    // Arc 140V (Lunar Lake), M = 512.
    static const Entry igpu[] = {
        {5120, 34816, {128, 16, 2, 2}},
        {17408, 5120, {64, 32, 4, 2}},
        {5120, 16384, {128, 16, 2, 4}},
        {6144, 5120, {128, 16, 2, 4}},
        {5120, 14336, {128, 16, 2, 4}},
        {5120, 248320, {128, 16, 2, 2}},
    };
    if (integrated) {
        for (const auto& e : igpu) {
            if (e.k == K && e.n == N)
                return e.t;
        }
        return MtTile{128, 16, 2, 4};
    }
    for (const auto& e : table) {
        if (e.k == K && e.n == N)
            return e.t;
    }
    return MtTile{64, 32, 4, 4};
}

// OV_TERNOCL_INT2_INT8_PREFILL=1: M > 8 runs the int2 x int8 DPAS GEMM on the same packed
// weights, quantizing the activations to int8 per (row, 128-group) inside the GEMM.
bool int8_prefill_enabled() {
    static const bool on = ov::util::getenv_bool("OV_TERNOCL_INT2_INT8_PREFILL");
    return on;
}

// Arc Pro B70 sweep of the 27B shapes, per M band (<= 16, <= 32, < 64, >= 64).
MtTile int8_tile(size_t K, size_t N, size_t M) {
    MtTile t{0, 0, 0, 0};
    if (env_ints("OV_TERNOCL_INT2_INT8_MT", &t.mt_m, 4))  // "mt_m,mt_n,wg_m,wg_n" for sweeps
        return t;
    const int band = M <= 16 ? 0 : (M <= 32 ? 1 : (M < 64 ? 2 : 3));
    struct Entry {
        size_t k, n;
        MtTile t[4];
    };
    static const Entry table[] = {
        {5120, 34816, {{8, 128, 1, 4}, {8, 64, 4, 1}, {8, 128, 8, 2}, {8, 128, 8, 2}}},     // gate_up (merged)
        {17408, 5120, {{8, 32, 2, 4}, {8, 32, 2, 4}, {8, 64, 8, 2}, {8, 128, 8, 2}}},       // down
        {5120, 16384, {{8, 32, 2, 4}, {8, 64, 2, 2}, {8, 128, 2, 4}, {8, 128, 16, 1}}},     // in_proj_qkvz
        {6144, 5120, {{8, 32, 2, 4}, {8, 32, 2, 4}, {8, 64, 8, 2}, {8, 128, 8, 2}}},        // out_proj / o_proj
        {5120, 14336, {{8, 32, 2, 4}, {8, 64, 2, 2}, {8, 128, 8, 2}, {8, 128, 4, 2}}},      // qkv
        {5120, 248320, {{8, 128, 2, 4}, {8, 128, 4, 4}, {8, 128, 8, 2}, {8, 128, 16, 1}}},  // lm_head
    };
    for (const auto& e : table) {
        if (e.k == K && e.n == N)
            return e.t[band];
    }
    static const MtTile fallback[4] = {{8, 32, 2, 4}, {8, 64, 2, 2}, {8, 128, 8, 2}, {8, 128, 8, 2}};
    return fallback[band];
}

// ---------------------------------------------------------------------------
// Programs are built once per (context, source, options) and shared; every impl
// creates its own cl_kernel from them so argument state is never shared.
cl::Program get_program(const ocl_engine& engine, const char* src, const std::string& opts) {
    static std::mutex m;
    static auto* cache = new std::map<std::string, cl::Program>;
    const std::string key =
        std::to_string(reinterpret_cast<uintptr_t>(engine.get_cl_context().get())) + "|" + std::to_string(reinterpret_cast<uintptr_t>(src)) + "|" + opts;
    std::lock_guard<std::mutex> lock(m);
    auto it = cache->find(key);
    if (it != cache->end())
        return it->second;
    cl::Program prog(engine.get_cl_context(), std::string(src));
    try {
        prog.build(std::vector<cl::Device>{engine.get_cl_device()}, opts.c_str());
    } catch (const cl::Error&) {
        OPENVINO_THROW("[GPU] ternocl int2: kernel build failed (", opts, "):\n", prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(engine.get_cl_device()));
    }
    if (ov::util::getenv_bool("OV_TERNOCL_INT2_CFG_DEBUG"))
        std::cerr << "[ternocl-int2] built " << opts << std::endl;
    return cache->emplace(key, prog).first->second;
}

kernel::ptr make_kernel(const ocl_engine& engine, const cl::Program& prog, const char* name) {
    return std::make_shared<ocl_kernel>(ocl_kernel_type(cl::Kernel(prog, name), engine.get_usm_helper()), name);
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

}  // namespace

// OpenVINO caches primitive_impl objects by kernel_impl_params, so one impl can
// serve several FullyConnected nodes of identical shape: packed weights are
// resolved by node id per execution.
struct TernoclInt2Packed {
    memory::ptr weights;
    memory::ptr scales;
    memory::ptr had_signs;  // i8 [K] +-1, or null
    // Weights repacked in place are taken from the network at execution: the program moves
    // host constants to device memory after the impls are built.
    primitive_id weights_id;
};

static std::mutex& ternocl_packed_mutex() {
    static std::mutex m;
    return m;
}

// Keyed by program id and node id. Process lifetime: the plugin can be unloaded after the context is gone.
static std::unordered_map<std::string, TernoclInt2Packed>& ternocl_packed_cache() {
    static auto* c = new std::unordered_map<std::string, TernoclInt2Packed>;
    return *c;
}

// Constant buffers already repacked in place, keyed by program id and buffer address.
static std::unordered_map<std::string, TernoclInt2Packed>& ternocl_packed_buffers() {
    static auto* c = new std::unordered_map<std::string, TernoclInt2Packed>;
    return *c;
}

struct fully_connected_ternocl_int2 : typed_primitive_impl<fully_connected> {
    using parent = typed_primitive_impl<fully_connected>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_ternocl_int2)

    const ocl_engine* _engine = nullptr;
    TernoclInt2Packed _own;  // fallback when the node-id lookup misses
    size_t _N = 0;
    size_t _K = 0;
    int _postop = 0;
    size_t _other_dep = 0;
    bool _out_f32 = false;
    size_t _had_block = 0;
    bool _integrated = false;

    // GEMV for M = 1, 2, <= 4, <= 8; M-tiled for M <= 16, <= 32, < 64, >= 64.
    struct Launch {
        kernel::ptr k;
        GemvTile g{};
        MtTile t{};
    };
    std::array<Launch, 8> _launch;
    // int8 prefill, M-tiled classes 4..7: scale pre-kernel + GEMM from one program.
    struct Int8Launch {
        kernel::ptr quant, gemm;
        MtTile t{};
    };
    std::array<Int8Launch, 4> _int8_launch;
    kernel::ptr _fwht;

    fully_connected_ternocl_int2() : parent("ternocl_int2") {}
    fully_connected_ternocl_int2(const ocl_engine& engine,
                                 TernoclInt2Packed own,
                                 size_t N,
                                 size_t K,
                                 int postop,
                                 size_t other_dep,
                                 bool out_f32,
                                 size_t had_block)
        : parent("ternocl_int2"),
          _engine(&engine),
          _own(std::move(own)),
          _N(N),
          _K(K),
          _postop(postop),
          _other_dep(other_dep),
          _out_f32(out_f32),
          _had_block(had_block),
          _integrated(engine.get_device_info().dev_type == device_type::integrated_gpu) {}

    std::unique_ptr<primitive_impl> clone() const override {
        // Shares the kernel handles like the stock OCL impls: OpenVINO clones
        // cached impls on every shape update, and rebuilding costs tens of us per FC.
        return std::make_unique<fully_connected_ternocl_int2>(*this);
    }

    bool is_cpu() const override {
        return false;
    }
    bool is_onednn() const override {
        return false;
    }

    // Scratch comes from the network memory pool, so it is shared across primitives
    // and resized with M: [had] f16 [M, K] rotated activation, [int8] f16 [K/128, LDSA(M)] scales.
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto& out = params.output_layouts[0];
        if (out.is_dynamic() || _N == 0)
            return {};
        const size_t M = ov::shape_size(out.get_shape()) / _N;
        std::vector<BufferDescriptor> descs;
        if (_had_block != 0)
            descs.emplace_back(M * _K, ov::element::f16);
        if (int8_prefill_enabled() && M > 8)
            descs.emplace_back((_K / kTernoclGroupSize) * ((M + 31) & ~size_t{31}), ov::element::f16);
        return descs;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&, kernel_arguments_data&) override {}

    std::string epi_opts() const {
        return " -DPOSTOP=" + std::to_string(_postop) + (_out_f32 ? " -DOUT_F32" : "");
    }

    static size_t launch_class(size_t M) {
        if (M <= 8)
            return M == 1 ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : 3));
        return M <= 16 ? 4 : (M <= 32 ? 5 : (M < 64 ? 6 : 7));
    }

    Launch& get_launch(size_t M) {
        auto& l = _launch[launch_class(M)];
        if (l.k)
            return l;
        std::string opts = "-cl-std=CL3.0";
        if (M <= 8) {
            const int sgm = M == 1 ? 1 : (M == 2 ? 2 : (M <= 4 ? 4 : 8));
            l.g = gemv_tile(_K, _N, _integrated);
            opts += " -DSGM=" + std::to_string(sgm) + " -DNSG_N=" + std::to_string(l.g.wgn / 16) + " -DLS=" + std::to_string(l.g.ls) +
                    " -DU=" + std::to_string(l.g.u) + " -DPF=0";
            l.k = make_kernel(*_engine, get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()), "int2_fp16_upcvt_gemm");
        } else {
            l.t = mt_tile(_K, _N, M, _integrated);
            opts += " -DMT_M=" + std::to_string(l.t.mt_m) + " -DMT_N=" + std::to_string(l.t.mt_n) + " -DWG_M=" + std::to_string(l.t.wg_m) +
                    " -DWG_N=" + std::to_string(l.t.wg_n) + " -cl-intel-256-GRF-per-thread";
            l.k = make_kernel(*_engine, get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()), "int2_fp16_upcvt_gemm_mt");
        }
        if (ov::util::getenv_bool("OV_TERNOCL_INT2_CFG_DEBUG"))
            std::cerr << "[ternocl-int2] K=" << _K << " N=" << _N << " M-class " << launch_class(M) << ": " << opts << epi_opts() << std::endl;
        return l;
    }

    Int8Launch& get_int8_launch(size_t M) {
        auto& l = _int8_launch[launch_class(M) - 4];
        if (l.gemm)
            return l;
        l.t = int8_tile(_K, _N, M);
        const std::string opts = "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -DQMODE=1 -DMT_M=" + std::to_string(l.t.mt_m) +
                                 " -DMT_N=" + std::to_string(l.t.mt_n) + " -DWG_M=" + std::to_string(l.t.wg_m) + " -DWG_N=" + std::to_string(l.t.wg_n) +
                                 " -cl-intel-256-GRF-per-thread" + epi_opts();
        const auto prog = get_program(*_engine, kTernoclInt8Source, opts);
        l.quant = make_kernel(*_engine, prog, "quant_a");
        l.gemm = make_kernel(*_engine, prog, "int2_int8_gemm_mt");
        if (ov::util::getenv_bool("OV_TERNOCL_INT2_CFG_DEBUG"))
            std::cerr << "[ternocl-int2] K=" << _K << " N=" << _N << " M-class " << launch_class(M) << " int8: " << opts << std::endl;
        return l;
    }

    event::ptr execute_int8(typed_primitive_inst<fully_connected>& instance, TernoclInt2Packed& pk, memory::cptr in, std::vector<event::ptr> deps, size_t M) {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto& l = get_int8_launch(M);
        const size_t groups = _K / kTernoclGroupSize;
        const auto& sa = instance.get_intermediates_memories().at(_had_block != 0 ? 1 : 0);

        // quant_a(A, SA, Aq, M, K): SA[g, m] = 127 / absmax of row m over group g. Aq is not written in QMODE 1.
        kernel_arguments_desc qd;
        qd.workGroups.global = {groups * 16, M, 1};
        qd.workGroups.local = {16, 1, 1};
        qd.arguments = {{argument_desc::Types::INPUT, 0},
                        {argument_desc::Types::OUTPUT, 0},
                        {argument_desc::Types::INPUT, 0},
                        {argument_desc::Types::SCALAR, 0},
                        {argument_desc::Types::SCALAR, 1}};
        scalars_desc qs(2);
        qs[0].t = qs[1].t = scalar_desc::Types::INT32;
        qs[0].v.s32 = static_cast<int32_t>(M);
        qs[1].v.s32 = static_cast<int32_t>(_K);
        kernel_arguments_data qa;
        qa.inputs = {in};
        qa.outputs = {sa};
        qa.scalars = &qs;
        stream.set_arguments(*l.quant, qd, qa);
        deps = {stream.enqueue_kernel(*l.quant, qd, qa, deps, false)};

        // A, Aq (unused), SA, B, SB, C, Other, Bias, M, N, K
        kernel_arguments_desc d;
        const size_t tn = static_cast<size_t>(l.t.mt_n * l.t.wg_n), tm = static_cast<size_t>(l.t.mt_m * l.t.wg_m);
        d.workGroups.local = {16 * static_cast<size_t>(l.t.wg_n * l.t.wg_m), 1, 1};
        d.workGroups.global = {ceil_div(_N, tn) * d.workGroups.local[0], ceil_div(M, tm), 1};
        d.arguments = {{argument_desc::Types::INPUT, 0},
                       {argument_desc::Types::INPUT, 0},
                       {argument_desc::Types::INPUT, 1},
                       {argument_desc::Types::INPUT, 2},
                       {argument_desc::Types::INPUT, 3},
                       {argument_desc::Types::OUTPUT, 0},
                       {argument_desc::Types::INPUT, 4},
                       {argument_desc::Types::INPUT, 5},
                       {argument_desc::Types::SCALAR, 0},
                       {argument_desc::Types::SCALAR, 1},
                       {argument_desc::Types::SCALAR, 2}};
        scalars_desc sc(3);
        for (auto& s : sc)
            s.t = scalar_desc::Types::INT32;
        sc[0].v.s32 = static_cast<int32_t>(M);
        sc[1].v.s32 = static_cast<int32_t>(_N);
        sc[2].v.s32 = static_cast<int32_t>(_K);
        memory::cptr other = (_postop == 1 || _postop == 2) ? instance.dep_memory_ptr(_other_dep) : in;
        memory::cptr bias = _postop == 3 ? instance.bias_memory() : in;
        kernel_arguments_data a;
        a.inputs = {in, sa, pk.weights, pk.scales, other, bias};
        a.outputs = {instance.output_memory_ptr(0)};
        a.scalars = &sc;
        stream.set_arguments(*l.gemm, d, a);
        return stream.enqueue_kernel(*l.gemm, d, a, deps, instance.is_output());
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, typed_primitive_inst<fully_connected>& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        const auto& params = instance.get_impl_params();
        const size_t M = ov::shape_size(params->output_layouts[0].get_shape()) / _N;

        TernoclInt2Packed* pk = &_own;
        {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            const auto prog = network.get_program();
            auto it = prog ? ternocl_packed_cache().find(std::to_string(prog->get_id()) + "|" + params->desc->id) : ternocl_packed_cache().end();
            if (it != ternocl_packed_cache().end())
                pk = &it->second;
            if (!pk->weights)
                pk->weights = network.get_primitive(pk->weights_id)->output_memory_ptr();
        }

        memory::cptr in = instance.input_memory_ptr(0);
        std::vector<event::ptr> deps = events;
        if (_had_block != 0) {
            // Rotated-basis checkpoint: the GEMM consumes H_1024(s * x) / 32.
            const auto& rotated = instance.get_intermediates_memories().at(0);
            if (!_fwht)
                _fwht = make_kernel(*_engine, get_program(*_engine, kTernoclFwhtSource, "-cl-std=CL3.0"), "hadamard_fwht_1024");
            kernel_arguments_desc d;
            d.workGroups.global = {M * (_K / 1024) * 128, 1, 1};
            d.workGroups.local = {128, 1, 1};
            d.arguments = {{argument_desc::Types::INPUT, 0},
                           {argument_desc::Types::INPUT, 1},
                           {argument_desc::Types::OUTPUT, 0},
                           {argument_desc::Types::SCALAR, 0},
                           {argument_desc::Types::SCALAR, 1}};
            scalars_desc sc(2);
            sc[0].t = sc[1].t = scalar_desc::Types::INT32;
            sc[0].v.s32 = static_cast<int32_t>(_K);
            sc[1].v.s32 = pk->had_signs ? 1 : 0;
            kernel_arguments_data a;
            a.inputs = {in, pk->had_signs ? memory::cptr(pk->had_signs) : in};
            a.outputs = {rotated};
            a.scalars = &sc;
            stream.set_arguments(*_fwht, d, a);
            deps = {stream.enqueue_kernel(*_fwht, d, a, deps, false)};
            in = rotated;
        }

        if (M > 8 && int8_prefill_enabled())
            return execute_int8(instance, *pk, in, deps, M);

        auto& l = get_launch(M);
        kernel_arguments_desc d;
        if (M <= 8) {
            const size_t sgm = M == 1 ? 1 : (M == 2 ? 2 : (M <= 4 ? 4 : 8));
            const size_t wgn = static_cast<size_t>(l.g.wgn);
            d.workGroups.local = {wgn * static_cast<size_t>(l.g.ls), 1, 1};
            d.workGroups.global = {ceil_div(_N, wgn) * d.workGroups.local[0], ceil_div(M, sgm), 1};
        } else {
            const size_t tn = static_cast<size_t>(l.t.mt_n * l.t.wg_n), tm = static_cast<size_t>(l.t.mt_m * l.t.wg_m);
            d.workGroups.local = {16 * static_cast<size_t>(l.t.wg_n * l.t.wg_m), 1, 1};
            d.workGroups.global = {ceil_div(_N, tn) * d.workGroups.local[0], ceil_div(M, tm), 1};
        }
        // A, B, S, C, Other, Bias, M, N, K
        d.arguments = {{argument_desc::Types::INPUT, 0},
                       {argument_desc::Types::INPUT, 1},
                       {argument_desc::Types::INPUT, 2},
                       {argument_desc::Types::OUTPUT, 0},
                       {argument_desc::Types::INPUT, 3},
                       {argument_desc::Types::INPUT, 4},
                       {argument_desc::Types::SCALAR, 0},
                       {argument_desc::Types::SCALAR, 1},
                       {argument_desc::Types::SCALAR, 2}};
        scalars_desc sc(3);
        for (auto& s : sc)
            s.t = scalar_desc::Types::INT32;
        sc[0].v.s32 = static_cast<int32_t>(M);
        sc[1].v.s32 = static_cast<int32_t>(_N);
        sc[2].v.s32 = static_cast<int32_t>(_K);
        // Unused Other / Bias slots are never read, but need a valid buffer.
        memory::cptr other = (_postop == 1 || _postop == 2) ? instance.dep_memory_ptr(_other_dep) : in;
        memory::cptr bias = _postop == 3 ? instance.bias_memory() : in;
        kernel_arguments_data a;
        a.inputs = {in, pk->weights, pk->scales, other, bias};
        a.outputs = {instance.output_memory_ptr(0)};
        a.scalars = &sc;
        stream.set_arguments(*l.k, d, a);
        return stream.enqueue_kernel(*l.k, d, a, deps, instance.is_output());
    }

public:
    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& prog = arg.get_program();
        auto& engine = prog.get_engine();
        auto& stream = prog.get_stream();
        const bool dbg = ov::util::getenv_bool("OV_TERNOCL_INT2_DEBUG");

        const auto wei_shape = arg.weights().get_output_layout(false).get_shape();
        const size_t N = wei_shape[0];
        const size_t K = wei_shape[1];
        const auto& desc = arg.get_primitive();

        size_t other_dep = 0;
        const int postop = ternocl_int2_postop(impl_params.fused_desc, desc->bias.is_valid(), &other_dep);
        OPENVINO_ASSERT(postop >= 0, "[GPU] ternocl int2: ", arg.id(), " has no folded epilogue for its fused chain");
        const bool out_f32 = impl_params.output_layouts[0].data_type == data_types::f32;

        const std::string key = std::to_string(prog.get_id()) + "|" + arg.id();
        TernoclInt2Packed own;
        {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            auto it = ternocl_packed_cache().find(key);
            if (it != ternocl_packed_cache().end())
                own = it->second;
        }
        auto wei_mem = arg.weights().as<data>().get_attached_memory_ptr();
        std::ostringstream buffer_key;
        buffer_key << prog.get_id() << "|" << wei_mem->buffer_ptr();
        const auto packed_already = [](const TernoclInt2Packed& p) {
            return p.weights || !p.weights_id.empty();
        };
        if (!packed_already(own)) {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            auto it = ternocl_packed_buffers().find(buffer_key.str());
            if (it != ternocl_packed_buffers().end())
                own = it->second;
        }
        if (!packed_already(own)) {
            // Dependencies are input, weights, [bias], scale, [zero point].
            const size_t scale_dep_idx = desc->bias.is_valid() ? 3 : 2;
            int32_t zp = 0;
            if (desc->decompression_zero_point_scalar.has_value()) {
                zp = static_cast<int32_t>(std::lround(desc->decompression_zero_point_scalar.value()));
            } else if (desc->decompression_zero_point.is_valid()) {
                const auto* zp_node = const_source(&arg.get_dependency(scale_dep_idx + 1));
                OPENVINO_ASSERT(zp_node != nullptr, "[GPU] ternocl int2: zero point is not constant");
                zp = static_cast<int32_t>(std::lround(read_scalar(zp_node->as<data>().get_attached_memory_ptr(), stream)));
            }

            const size_t packed_bytes = N * K / 4;
            OPENVINO_ASSERT(wei_mem->size() >= packed_bytes,
                            "[GPU] ternocl int2: weight buffer ",
                            wei_mem->size(),
                            " B is smaller than the dense ",
                            packed_bytes,
                            " B");
            // The kernel layout has the constant's byte size, so it is written over the constant and the
            // weights stay resident once. Other readers, or an export (cache_dir) saving the constant's
            // bytes, still need the original, so those cases get a separate buffer.
            const bool in_place = prog.get_config().get_cache_dir().empty() && arg.weights().get_users().size() == 1;
            // Blocking copies: mapping a large device constant can expose it before it is resident.
            std::vector<uint8_t> wei_host(wei_mem->size());
            wei_mem->copy_to(stream, wei_host.data(), true);
            std::vector<uint32_t> packed((K / kTernoclPackFactor) * N);
            pack_weights(wei_host.data(), packed.data(), N, K, zp);
            if (in_place) {
                wei_mem->copy_from(stream, packed.data(), 0, 0, packed_bytes, true);
                own.weights_id = arg.weights().id();
            } else {
                own.weights = engine.allocate_memory(
                    layout{ov::PartialShape{static_cast<int64_t>(K / kTernoclPackFactor), static_cast<int64_t>(N)}, data_types::i32, format::bfyx},
                    allocation_type::usm_device,
                    false);
                own.weights->copy_from(stream, packed.data(), true);
            }

            // OpenVINO keeps scales per output channel, [N, groups]; the kernel wants [groups, N].
            const size_t groups = K / kTernoclGroupSize;
            const auto* scale_node = const_source(&arg.get_dependency(scale_dep_idx));
            OPENVINO_ASSERT(scale_node != nullptr, "[GPU] ternocl int2: decompression scale is not constant");
            auto scale_mem = scale_node->as<data>().get_attached_memory_ptr();
            OPENVINO_ASSERT(scale_mem->get_layout().data_type == data_types::f16, "[GPU] ternocl int2: decompression scale must be f16");
            // const_source() skipped any reorder, so this is the constant's own [N, groups] layout.
            // That node may not survive into the network, so the scales (1/16 of the weights) keep a copy.
            const auto scale_shape = scale_mem->get_layout().get_shape();
            const bool n_major = scale_shape.size() >= 2 && scale_shape[0] == N;
            std::vector<uint16_t> scale_src(scale_mem->size() / sizeof(uint16_t));
            scale_mem->copy_to(stream, scale_src.data(), true);
            std::vector<uint16_t> scale_host(groups * N);
            for (size_t g = 0; g < groups; ++g) {
                for (size_t n = 0; n < N; ++n)
                    scale_host[g * N + n] = n_major ? scale_src[n * groups + g] : scale_src[g * N + n];
            }
            own.scales = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(groups), static_cast<int64_t>(N)}, data_types::f16, format::bfyx},
                                                allocation_type::usm_device,
                                                false);
            own.scales->copy_from(stream, scale_host.data(), true);

            if (desc->hadamard_block != 0 && !desc->hadamard_signs.empty()) {
                OPENVINO_ASSERT(desc->hadamard_signs.size() == K, "[GPU] ternocl int2: hadamard signs length mismatch");
                own.had_signs =
                    engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(K)}, data_types::i8, format::bfyx}, allocation_type::usm_device, false);
                own.had_signs->copy_from(stream, desc->hadamard_signs.data(), true);
            }
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            if (in_place)
                ternocl_packed_buffers().emplace(buffer_key.str(), own);
            // A second impl for a node that is already executing must not swap its buffers.
            own = ternocl_packed_cache().try_emplace(key, own).first->second;
        } else {
            std::lock_guard<std::mutex> lock(ternocl_packed_mutex());
            ternocl_packed_cache().try_emplace(key, own);
        }
        if (dbg) {
            std::cerr << "[ternocl-int2] create " << arg.id() << " N=" << N << " K=" << K << " postop=" << postop << " out_f32=" << out_f32
                      << " hadamard=" << desc->hadamard_block << std::endl;
        }

        return std::make_unique<fully_connected_ternocl_int2>(downcast<const ocl_engine>(engine), own, N, K, postop, other_dep, out_f32, desc->hadamard_block);
    }
};

// A forced impl type (unit tests) skips validate() and takes the first OCL manager,
// so FCs this impl rejects are handed to the next OCL FC manager.
const ImplementationManager& stock_ocl_manager(const ImplementationManager& self, shape_types shape_type) {
    for (const auto& m : ov::intel_gpu::Registry<fully_connected>::get_implementations()) {
        if (m->get_impl_type() == impl_types::ocl && m->get_type_info() != self.get_type_info() && (m->get_shape_type() & shape_type) == shape_type)
            return *m;
    }
    OPENVINO_THROW("[GPU] ternocl int2: no OCL FullyConnected implementation to fall back to");
}

std::unique_ptr<primitive_impl> TernoclInt2FCImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fully_connected>());
    if (!validate_impl(node))
        return stock_ocl_manager(*this, get_shape_type(node)).create_impl(node, params);
    return fully_connected_ternocl_int2::create(static_cast<const fully_connected_node&>(node), params);
}

std::unique_ptr<primitive_impl> TernoclInt2FCImplementationManager::create_impl(const kernel_impl_params& params) const {
    if (params.input_layouts.size() < 2 || params.input_layouts[1].data_type != data_types::u2)
        return stock_ocl_manager(*this, get_shape_type(params)).create_impl(params);
    OPENVINO_NOT_IMPLEMENTED;
}

in_out_fmts_t TernoclInt2FCImplementationManager::query_formats(const program_node& node) const {
    if (!validate_impl(node))
        return stock_ocl_manager(*this, get_shape_type(node)).query_formats(node);
    return ImplementationManager::query_formats(node);
}

}  // namespace ocl
}  // namespace cldnn

#endif  // OV_GPU_WITH_OCL_RT
