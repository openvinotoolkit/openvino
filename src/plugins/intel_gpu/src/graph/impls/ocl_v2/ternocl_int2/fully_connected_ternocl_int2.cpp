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
#    include <vector>

#    include "fully_connected_inst.h"
#    include "intel_gpu/runtime/debug_configuration.hpp"
#    include "intel_gpu/runtime/kernel_args.hpp"
#    include "intel_gpu/runtime/memory.hpp"
#    include "primitive_inst.h"
#    include "registry/registry.hpp"
#    include "runtime/ocl/ocl_engine.hpp"
#    include "runtime/ocl/ocl_kernel.hpp"
#    include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

namespace cldnn {
namespace ocl {

namespace {

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

GemvTile gemv_tile(size_t K, size_t N, bool integrated) {
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
        // Bonsai 1.7B (also CAT-Q 1.7B): qkv, o_proj, gate_up, down, lm_head
        {2048, 4096, {32, 8, 1}},
        {2048, 2048, {32, 8, 1}},
        {2048, 12288, {16, 2, 1}},
        {6144, 2048, {16, 8, 1}},
        {2048, 151680, {32, 1, 1}},
        // Bonsai 4B
        {2560, 6144, {32, 4, 1}},
        {4096, 2560, {32, 8, 1}},
        {2560, 19456, {64, 1, 1}},
        {9728, 2560, {32, 8, 1}},
        {2560, 151680, {32, 1, 1}},
        // CAT-Q (Qwen3) 32B: gate_up, down, qkv, o_proj
        {5120, 51200, {16, 1, 1}},
        {25600, 5120, {16, 6, 1}},
        {5120, 10240, {16, 8, 1}},
        {8192, 5120, {32, 6, 1}},
    };
    // Arc 140V (Lunar Lake), M = 1, paced sweep (select decode).
    static const Entry igpu[] = {
        {5120, 34816, {16, 2, 1}},
        {17408, 5120, {128, 1, 1}},
        {5120, 16384, {16, 1, 2}},
        {6144, 5120, {64, 1, 1}},
        {5120, 14336, {16, 1, 1}},
        {5120, 248320, {32, 2, 2}},
        // Bonsai 1.7B / 4B / 8B, CAT-Q 32B (host on the E-cores)
        {2048, 4096, {32, 2, 1}},
        {2048, 2048, {16, 4, 1}},
        {2048, 12288, {16, 1, 1}},
        {6144, 2048, {32, 4, 1}},
        {2048, 151680, {32, 4, 1}},
        {2560, 6144, {16, 1, 1}},
        {4096, 2560, {32, 2, 1}},
        {2560, 19456, {16, 2, 1}},
        {9728, 2560, {32, 2, 2}},
        {2560, 151680, {32, 4, 1}},
        {4096, 6144, {64, 1, 1}},
        {4096, 4096, {32, 2, 1}},
        {4096, 24576, {16, 2, 1}},
        {12288, 4096, {32, 2, 1}},
        {4096, 151680, {16, 1, 1}},
        {5120, 10240, {16, 4, 1}},
        {8192, 5120, {64, 1, 1}},
        {5120, 51200, {16, 2, 1}},
        {25600, 5120, {32, 1, 1}},
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
    if (M < 64) {
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

// Arc Pro B70 sweep of the 27B shapes, per M band (<= 16, <= 32, < 64, >= 64).
MtTile int8_tile(size_t K, size_t N, size_t M) {
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
// Programs are built once per (context, source, options) and shared by the impls created on that
// context (the FCs of a model differ only in a few shapes); every impl creates its own cl_kernel
// from them, so argument state is never shared. Each impl holds the cache, so it is released with
// the last of them.
class TernoclProgramCache {
public:
    static std::shared_ptr<TernoclProgramCache> get(const ocl_engine& engine) {
        static std::mutex m;
        static std::map<cl_context, std::weak_ptr<TernoclProgramCache>> caches;
        std::lock_guard<std::mutex> lock(m);
        for (auto it = caches.begin(); it != caches.end();) {
            it = it->second.expired() ? caches.erase(it) : std::next(it);
        }
        auto& entry = caches[engine.get_cl_context().get()];
        auto cache = entry.lock();
        if (!cache) {
            cache = std::make_shared<TernoclProgramCache>();
            entry = cache;
        }
        return cache;
    }

    cl::Program get_program(const ocl_engine& engine, const char* src, const std::string& opts) {
        const std::string key = std::to_string(reinterpret_cast<uintptr_t>(src)) + "|" + opts;
        std::lock_guard<std::mutex> lock(_m);
        auto it = _programs.find(key);
        if (it != _programs.end())
            return it->second;
        cl::Program prog(engine.get_cl_context(), std::string(src));
        try {
            prog.build(std::vector<cl::Device>{engine.get_cl_device()}, opts.c_str());
        } catch (const cl::Error&) {
            OPENVINO_THROW("[GPU] ternocl int2: kernel build failed (", opts, "):\n", prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(engine.get_cl_device()));
        }
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] built " << opts << std::endl;
        return _programs.emplace(key, prog).first->second;
    }

private:
    std::mutex _m;
    std::map<std::string, cl::Program> _programs;
};

kernel::ptr make_kernel(const ocl_engine& engine, const cl::Program& prog, const char* name) {
    return std::make_shared<ocl_kernel>(ocl_kernel_type(cl::Kernel(prog, name), engine.get_usm_helper()), name);
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

}  // namespace

struct fully_connected_ternocl_int2 : typed_primitive_impl<fully_connected> {
    using parent = typed_primitive_impl<fully_connected>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_ternocl_int2)

    // Weights and scales come from the node's inputs: the weights reorder packs the weights, and
    // prepare_quantization lays the grouped scales out as [groups, N]. Nothing here depends on the
    // node beyond kernel_impl_params, so a cached impl can serve any FC with equal params.
    const ocl_engine* _engine = nullptr;
    std::shared_ptr<TernoclProgramCache> _programs;
    // i8 [K] +-1 from the primitive's hadamard_signs (part of the impl cache key), or null.
    memory::ptr _had_signs;
    size_t _N = 0;
    size_t _K = 0;
    TernoclPostop _postop = TernoclPostop::none;
    size_t _other_dep = 0;
    size_t _scale_dep = 0;
    bool _out_f32 = false;
    size_t _had_block = 0;
    bool _integrated = false;
    // GPU_TERNOCL_INT2_INT8_PREFILL: M > 8 on the int2 x int8 DPAS GEMM (activations quantized per 128-group).
    bool _int8_prefill = false;

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
    fully_connected_ternocl_int2(ocl_engine& engine,
                                 const fully_connected& desc,
                                 size_t N,
                                 size_t K,
                                 TernoclPostop postop,
                                 size_t other_dep,
                                 size_t scale_dep,
                                 bool out_f32,
                                 bool int8)
        : parent("ternocl_int2"),
          _N(N),
          _K(K),
          _postop(postop),
          _other_dep(other_dep),
          _scale_dep(scale_dep),
          _out_f32(out_f32),
          _had_block(desc.hadamard_block),
          _int8_prefill(int8) {
        attach(engine, desc);
    }

    // Engine-bound state, rebuilt rather than serialized.
    void attach(ocl_engine& engine, const fully_connected& desc) {
        _engine = &engine;
        _integrated = engine.get_device_info().dev_type == device_type::integrated_gpu;
        _programs = TernoclProgramCache::get(engine);
        _had_signs = nullptr;
        if (desc.hadamard_block != 0 && !desc.hadamard_signs.empty()) {
            OPENVINO_ASSERT(desc.hadamard_signs.size() == _K, "[GPU] ternocl int2: hadamard signs length mismatch");
            _had_signs =
                engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(_K)}, data_types::i8, format::bfyx}, allocation_type::usm_device, false);
            _had_signs->copy_from(engine.get_service_stream(), desc.hadamard_signs.data(), true);
        }
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _N;
        ob << _K;
        ob << static_cast<int>(_postop);
        ob << _other_dep;
        ob << _scale_dep;
        ob << _out_f32;
        ob << _had_block;
        ob << _int8_prefill;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        int postop = 0;
        ib >> _N;
        ib >> _K;
        ib >> postop;
        ib >> _other_dep;
        ib >> _scale_dep;
        ib >> _out_f32;
        ib >> _had_block;
        ib >> _int8_prefill;
        _postop = static_cast<TernoclPostop>(postop);
        const auto* params = reinterpret_cast<const kernel_impl_params*>(ib.getKernelImplParams());
        OPENVINO_ASSERT(params != nullptr, "[GPU] ternocl int2: kernel_impl_params are required to load the impl");
        attach(downcast<ocl_engine>(ib.get_engine()), *params->typed_desc<fully_connected>());
    }

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
        if (M > 8 && _int8_prefill)
            descs.emplace_back((_K / kTernoclGroupSize) * ((M + 31) & ~size_t{31}), ov::element::f16);
        return descs;
    }

protected:
    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&) override {}
    void set_arguments_impl(typed_primitive_inst<fully_connected>&, kernel_arguments_data&) override {}

    std::string epi_opts() const {
        return " -DPOSTOP=" + std::to_string(static_cast<int>(_postop)) + (_out_f32 ? " -DOUT_F32" : "");
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
            l.k = make_kernel(*_engine, _programs->get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()), "int2_fp16_upcvt_gemm");
        } else {
            l.t = mt_tile(_K, _N, M, _integrated);
            opts += " -DMT_M=" + std::to_string(l.t.mt_m) + " -DMT_N=" + std::to_string(l.t.mt_n) + " -DWG_M=" + std::to_string(l.t.wg_m) +
                    " -DWG_N=" + std::to_string(l.t.wg_n) + " -cl-intel-256-GRF-per-thread";
            l.k = make_kernel(*_engine, _programs->get_program(*_engine, kTernoclUpcvtSource, opts + epi_opts()), "int2_fp16_upcvt_gemm_mt");
        }
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] K=" << _K << " N=" << _N << " M-class " << launch_class(M) << ": " << opts << epi_opts() << std::endl;
        return l;
    }

    Int8Launch& get_int8_launch(size_t M) {
        const auto cls = launch_class(M);
        OPENVINO_ASSERT(cls >= 4, "[GPU] ternocl int2: int8 launch requires M > 8");
        auto& l = _int8_launch[cls - 4];
        if (l.gemm)
            return l;
        l.t = int8_tile(_K, _N, M);
        const std::string opts = "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt -DQMODE=1 -DMT_M=" + std::to_string(l.t.mt_m) +
                                 " -DMT_N=" + std::to_string(l.t.mt_n) + " -DWG_M=" + std::to_string(l.t.wg_m) + " -DWG_N=" + std::to_string(l.t.wg_n) +
                                 " -cl-intel-256-GRF-per-thread" + epi_opts();
        const auto prog = _programs->get_program(*_engine, kTernoclInt8Source, opts);
        l.quant = make_kernel(*_engine, prog, "quant_a");
        l.gemm = make_kernel(*_engine, prog, "int2_int8_gemm_mt");
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] K=" << _K << " N=" << _N << " M-class " << launch_class(M) << " int8: " << opts << std::endl;
        return l;
    }

    event::ptr execute_int8(typed_primitive_inst<fully_connected>& instance, memory::cptr in, std::vector<event::ptr> deps, size_t M) {
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
        memory::cptr other = ternocl_postop_reads_other(_postop) ? instance.dep_memory_ptr(_other_dep) : in;
        memory::cptr bias = _postop == TernoclPostop::bias ? instance.bias_memory() : in;
        kernel_arguments_data a;
        a.inputs = {in, sa, instance.weights_memory(), instance.dep_memory_ptr(_scale_dep), other, bias};
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

        memory::cptr in = instance.input_memory_ptr(0);
        std::vector<event::ptr> deps = events;
        if (_had_block != 0) {
            // Rotated-basis checkpoint: the GEMM consumes H_1024(s * x) / 32.
            const auto& rotated = instance.get_intermediates_memories().at(0);
            if (!_fwht)
                _fwht = make_kernel(*_engine, _programs->get_program(*_engine, kTernoclFwhtSource, "-cl-std=CL3.0"), "hadamard_fwht_1024");
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
            sc[1].v.s32 = _had_signs ? 1 : 0;
            kernel_arguments_data a;
            a.inputs = {in, _had_signs ? memory::cptr(_had_signs) : in};
            a.outputs = {rotated};
            a.scalars = &sc;
            stream.set_arguments(*_fwht, d, a);
            deps = {stream.enqueue_kernel(*_fwht, d, a, deps, false)};
            in = rotated;
        }

        if (M > 8 && _int8_prefill)
            return execute_int8(instance, in, deps, M);

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
        memory::cptr other = ternocl_postop_reads_other(_postop) ? instance.dep_memory_ptr(_other_dep) : in;
        memory::cptr bias = _postop == TernoclPostop::bias ? instance.bias_memory() : in;
        kernel_arguments_data a;
        a.inputs = {in, instance.weights_memory(), instance.dep_memory_ptr(_scale_dep), other, bias};
        a.outputs = {instance.output_memory_ptr(0)};
        a.scalars = &sc;
        stream.set_arguments(*l.k, d, a);
        return stream.enqueue_kernel(*l.k, d, a, deps, instance.is_output());
    }

public:
    static std::unique_ptr<primitive_impl> create(const fully_connected_node& arg, const kernel_impl_params& impl_params) {
        auto& prog = arg.get_program();
        const bool int8 = prog.get_config().get_ternocl_int2_int8_prefill();
        const auto& desc = impl_params.typed_desc<fully_connected>();

        // At build the weights input is the u2 constant; once the reorder below is folded it is the packed one.
        const auto& wei_layout = impl_params.get_input_layout(1);
        const auto wei_shape = wei_layout.get_shape();
        const size_t N = wei_shape[0];
        const size_t K = wei_shape[1];

        size_t other_dep = 0;
        const auto postop = ternocl_int2_postop(impl_params.fused_desc, desc->bias.is_valid(), &other_dep);
        OPENVINO_ASSERT(postop.has_value(), "[GPU] ternocl int2: ", arg.id(), " has no folded epilogue for its fused chain");
        const bool out_f32 = impl_params.output_layouts[0].data_type == data_types::f32;
        // Dependencies are input, weights, [bias], scale, [zero point].
        const size_t scale_dep = desc->bias.is_valid() ? 3 : 2;

        GPU_DEBUG_TRACE << "[ternocl-int2] create " << arg.id() << " N=" << N << " K=" << K << " postop=" << static_cast<int>(*postop) << " out_f32=" << out_f32
                        << " hadamard=" << desc->hadamard_block << " int8_prefill=" << int8 << std::endl;

        auto impl =
            std::make_unique<fully_connected_ternocl_int2>(downcast<ocl_engine>(prog.get_engine()), *desc, N, K, *postop, other_dep, scale_dep, out_f32, int8);
        auto packed_layout = wei_layout;
        packed_layout.format = format::is_os_yx_isv16_ternary;
        impl->_weights_reorder_params = std::make_shared<WeightsReorderParams>(wei_layout, packed_layout);
        return impl;
    }
};

// A forced impl type (unit tests) skips validate() and takes the first OCL manager,
// so FCs this impl rejects are handed to the next OCL FC manager.
static const ImplementationManager& stock_ocl_manager(const ImplementationManager& self, shape_types shape_type) {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::fully_connected_ternocl_int2)

#endif  // OV_GPU_WITH_OCL_RT
