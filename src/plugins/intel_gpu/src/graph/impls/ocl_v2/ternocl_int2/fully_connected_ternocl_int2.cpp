// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_ternocl_int2.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "fully_connected_inst.h"
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "impls/ocl_v2/utils/kernel_generator.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "primitive_inst.h"
#include "registry/registry.hpp"
#include "ternocl_int2_kernels.inc"  // generated from TERNOCL_ROOT (CMakeLists.txt)

namespace cldnn {
namespace ocl {

namespace {

using ov::intel_gpu::ArgumentDescriptor;
using ov::intel_gpu::Arguments;
using ov::intel_gpu::DispatchDataFunc;
using ov::intel_gpu::ImplRuntimeParams;
using ov::intel_gpu::KernelData;
using ov::intel_gpu::KernelLanguage;
using ov::intel_gpu::KernelString;
using ov::intel_gpu::RuntimeParams;
using ov::intel_gpu::ScalarDescriptor;
using ov::intel_gpu::WorkGroupSizes;
using ov::intel_gpu::ocl::KernelGenerator;
using ov::intel_gpu::ocl::PrimitiveImplOCL;
using ov::intel_gpu::ocl::Stage;

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

MtTile mt_tile(size_t K, size_t N, size_t cls, bool integrated) {
    if (cls < 7) {
        // Sweep at M = 12 / 20 / 32 / 48 over the 27B shapes; the winners group
        // by output width (narrow N <= 8192, head N >= 65536).
        const size_t band = cls - 4;
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

// Arc Pro B70 sweep of the 27B shapes, per M-tiled launch class (M <= 16, <= 32, < 64, >= 64).
MtTile int8_tile(size_t K, size_t N, size_t cls) {
    const size_t band = cls - 4;
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

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// Classes 0-3: GEMV for M = 1, 2, <= 4, <= 8 (SGM = 1, 2, 4, 8).
// Classes 4-7: M-tiled GEMM for M <= 16, <= 32, < 64, >= 64.
constexpr size_t kLaunchClasses = 8;
constexpr size_t kFirstTiledClass = 4;

size_t launch_class(size_t M) {
    if (M <= 8)
        return M <= 1 ? 0 : (M == 2 ? 1 : (M <= 4 ? 2 : 3));
    return M <= 16 ? 4 : (M <= 32 ? 5 : (M < 64 ? 6 : 7));
}

struct FcShape {
    size_t M, N, K;
};

// N and K from the [N, K] weights, M from the output (0 while it is dynamic).
FcShape fc_shape(const RuntimeParams& params) {
    const auto& w = params.get_input_layout(1).get_partial_shape();
    const size_t N = static_cast<size_t>(w[0].get_length());
    const size_t K = static_cast<size_t>(w[1].get_length());
    const auto& out = params.output_layouts[0];
    return {out.is_dynamic() ? 0 : ov::shape_size(out.get_shape()) / N, N, K};
}

bool is_integrated(const RuntimeParams& params) {
    return params.get_device_info().dev_type == device_type::integrated_gpu;
}

struct Epilogue {
    TernoclPostop postop;
    size_t other_dep;  // dependency index of the eltwise operand
    bool out_f32;
};

Epilogue epilogue(const RuntimeParams& params) {
    const auto& desc = params.typed_desc<fully_connected>();
    size_t other_dep = 0;
    const auto postop = ternocl_int2_postop(params.fused_desc, desc->bias.is_valid(), &other_dep);
    OPENVINO_ASSERT(postop.has_value(), "[GPU] ternocl int2: ", desc->id, " has no folded epilogue for its fused chain");
    return {*postop, other_dep, params.output_layouts[0].data_type == data_types::f32};
}

void set_tiled_dispatch(WorkGroupSizes& wgs, const MtTile& t, const FcShape& s) {
    const size_t tn = static_cast<size_t>(t.mt_n * t.wg_n), tm = static_cast<size_t>(t.mt_m * t.wg_m);
    wgs.local = {16 * static_cast<size_t>(t.wg_n * t.wg_m), 1, 1};
    wgs.global = {ceil_div(s.N, tn) * wgs.local[0], ceil_div(s.M, tm), 1};
}

std::string tile_options(const MtTile& t) {
    return " -DMT_M=" + std::to_string(t.mt_m) + " -DMT_N=" + std::to_string(t.mt_n) + " -DWG_M=" + std::to_string(t.wg_m) +
           " -DWG_N=" + std::to_string(t.wg_n) + " -cl-intel-256-GRF-per-thread";
}

enum class TernoclKernel { upcvt, quant, int8_gemm, fwht };

// One kernel per program: TERNOCL_SELECT_KERNELS drops the other kernels of the source.
// The tile depends only on (K, N, launch class, device), so a stage serves every M of its class.
class TernoclStageGenerator : public KernelGenerator {
public:
    TernoclStageGenerator(TernoclKernel kind, size_t cls) : KernelGenerator("ternocl_int2", suffix(kind, cls)), m_kind(kind), m_cls(cls) {}

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override {
        const std::string name = kernel_name();
        std::string selector = name;
        std::transform(selector.begin(), selector.end(), selector.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

        KernelData kd;
        kd.code = std::make_shared<KernelString>();
        kd.code->language = KernelLanguage::OCLC_V2;
        kd.code->entry_point = get_entry_point(params);
        kd.code->options = get_build_options(params);
        kd.code->batch_compilation = false;
        kd.code->str =
            "#define TERNOCL_SELECT_KERNELS\n#define TERNOCL_KERNEL_" + selector + "\n#define " + name + " " + kd.code->entry_point + "\n" + source();

        kd.params.arguments = get_arguments_desc(params);
        kd.params.layerID = params.desc->id;
        kd.update_dispatch_data_func = get_dispatch_data_func();
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] " << params.desc->id << " " << name << ": " << kd.code->options << std::endl;
        return kd;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[kind = m_kind, cls = m_cls](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto s = fc_shape(params);
            auto& wgs = kd.params.workGroups;
            std::vector<int32_t> scalars;
            switch (kind) {
            case TernoclKernel::upcvt:
                if (cls < kFirstTiledClass) {
                    const auto g = gemv_tile(s.K, s.N, is_integrated(params));
                    const size_t wgn = static_cast<size_t>(g.wgn);
                    wgs.local = {wgn * static_cast<size_t>(g.ls), 1, 1};
                    wgs.global = {ceil_div(s.N, wgn) * wgs.local[0], ceil_div(s.M, size_t{1} << cls), 1};
                } else {
                    set_tiled_dispatch(wgs, mt_tile(s.K, s.N, cls, is_integrated(params)), s);
                }
                scalars = {static_cast<int32_t>(s.M), static_cast<int32_t>(s.N), static_cast<int32_t>(s.K)};
                break;
            case TernoclKernel::quant:
                // SA[g, m] = 127 / absmax of row m over group g: one sub-group per (row, group).
                wgs.local = {16, 1, 1};
                wgs.global = {(s.K / kTernoclGroupSize) * 16, s.M, 1};
                scalars = {static_cast<int32_t>(s.M), static_cast<int32_t>(s.K)};
                break;
            case TernoclKernel::int8_gemm:
                set_tiled_dispatch(wgs, int8_tile(s.K, s.N, cls), s);
                scalars = {static_cast<int32_t>(s.M), static_cast<int32_t>(s.N), static_cast<int32_t>(s.K)};
                break;
            case TernoclKernel::fwht: {
                // One work-group of B/8 items per B-wide block.
                const size_t block = params.typed_desc<fully_connected>()->hadamard_block;
                wgs.local = {block / 8, 1, 1};
                wgs.global = {s.M * (s.K / block) * (block / 8), 1, 1};
                scalars = {static_cast<int32_t>(s.K), params.typed_desc<fully_connected>()->hadamard_signs.empty() ? 0 : 1};
                break;
            }
            }
            kd.params.scalars.clear();
            for (const auto v : scalars) {
                ScalarDescriptor d;
                d.t = ScalarDescriptor::Types::INT32;
                d.v.s32 = v;
                kd.params.scalars.push_back(d);
            }
        }};
    }

protected:
    // Inputs: {A, B, S, Other, Bias, Hadamard signs}. Scratch: {[rotated A], [SA]}.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        using T = ArgumentDescriptor::Types;
        const bool rotated = params.typed_desc<fully_connected>()->hadamard_block != 0;
        const ArgumentDescriptor a = rotated ? ArgumentDescriptor{T::INTERNAL_BUFFER, 0} : ArgumentDescriptor{T::INPUT, 0};
        const ArgumentDescriptor sa{T::INTERNAL_BUFFER, rotated ? 1u : 0u};
        switch (m_kind) {
        case TernoclKernel::upcvt:  // A, B, S, C, Other, Bias, M, N, K
            return {a, {T::INPUT, 1}, {T::INPUT, 2}, {T::OUTPUT, 0}, {T::INPUT, 3}, {T::INPUT, 4}, {T::SCALAR, 0}, {T::SCALAR, 1}, {T::SCALAR, 2}};
        case TernoclKernel::quant:  // A, SA, Aq (not written in QMODE 1), M, K
            return {a, sa, a, {T::SCALAR, 0}, {T::SCALAR, 1}};
        case TernoclKernel::int8_gemm:  // A, Aq (unused), SA, B, SB, C, Other, Bias, M, N, K
            return {a, a, sa, {T::INPUT, 1}, {T::INPUT, 2}, {T::OUTPUT, 0}, {T::INPUT, 3}, {T::INPUT, 4}, {T::SCALAR, 0}, {T::SCALAR, 1}, {T::SCALAR, 2}};
        case TernoclKernel::fwht:  // x, signs, H_B(s * x) / sqrt(B), K, has_signs
            return {{T::INPUT, 0}, {T::INPUT, 5}, {T::INTERNAL_BUFFER, 0}, {T::SCALAR, 0}, {T::SCALAR, 1}};
        }
        OPENVINO_THROW("[GPU] ternocl int2: unknown kernel");
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        const auto s = fc_shape(params);
        const auto ep = epilogue(params);
        const std::string epi = " -DPOSTOP=" + std::to_string(static_cast<int>(ep.postop)) + (ep.out_f32 ? " -DOUT_F32" : "");
        std::string opts = "-cl-std=CL3.0";
        switch (m_kind) {
        case TernoclKernel::upcvt:
            if (m_cls < kFirstTiledClass) {
                const auto g = gemv_tile(s.K, s.N, is_integrated(params));
                opts += " -DSGM=" + std::to_string(size_t{1} << m_cls) + " -DNSG_N=" + std::to_string(g.wgn / 16) + " -DLS=" + std::to_string(g.ls) +
                        " -DU=" + std::to_string(g.u) + " -DPF=0";
            } else {
                opts += tile_options(mt_tile(s.K, s.N, m_cls, is_integrated(params)));
            }
            return opts + epi;
        case TernoclKernel::quant:
            return opts + " -cl-fp32-correctly-rounded-divide-sqrt -DQMODE=1";
        case TernoclKernel::int8_gemm:
            return opts + " -cl-fp32-correctly-rounded-divide-sqrt -DQMODE=1" + tile_options(int8_tile(s.K, s.N, m_cls)) + epi;
        case TernoclKernel::fwht:
            return opts + " -DHB=" + std::to_string(params.typed_desc<fully_connected>()->hadamard_block);
        }
        OPENVINO_THROW("[GPU] ternocl int2: unknown kernel");
    }

private:
    static std::string suffix(TernoclKernel kind, size_t cls) {
        switch (kind) {
        case TernoclKernel::upcvt:
            return "upcvt" + std::to_string(cls);
        case TernoclKernel::quant:
            return "quant";
        case TernoclKernel::int8_gemm:
            return "int8_" + std::to_string(cls);
        case TernoclKernel::fwht:
            return "fwht";
        }
        return {};
    }

    [[nodiscard]] std::string kernel_name() const {
        switch (m_kind) {
        case TernoclKernel::upcvt:
            return m_cls < kFirstTiledClass ? "int2_fp16_upcvt_gemm" : "int2_fp16_upcvt_gemm_mt";
        case TernoclKernel::quant:
            return "quant_a";
        case TernoclKernel::int8_gemm:
            return "int2_int8_gemm_mt";
        case TernoclKernel::fwht:
            return "hadamard_fwht";
        }
        return {};
    }

    [[nodiscard]] const char* source() const {
        switch (m_kind) {
        case TernoclKernel::upcvt:
            return kTernoclUpcvtSource;
        case TernoclKernel::quant:
        case TernoclKernel::int8_gemm:
            return kTernoclInt8Source;
        case TernoclKernel::fwht:
            return kTernoclFwhtSource;
        }
        return "";
    }

    TernoclKernel m_kind;
    size_t m_cls;
};

}  // namespace

// Stages: the up-convert GEMM per launch class, the int8 prefill (quant_a + GEMM per M-tiled class)
// and the FWHT of rotated-basis checkpoints. A dynamic-shape impl builds all classes and picks one
// per execution. A static-shape impl builds only its own class.
class fully_connected_ternocl_int2 : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::fully_connected_ternocl_int2)

    std::array<Stage::Ptr, kLaunchClasses> upcvt = make_stages<kLaunchClasses>(TernoclKernel::upcvt, 0);
    Stage::Ptr quant = make_stage<TernoclStageGenerator>(TernoclKernel::quant, 0);
    std::array<Stage::Ptr, kLaunchClasses - kFirstTiledClass> int8_gemm =
        make_stages<kLaunchClasses - kFirstTiledClass>(TernoclKernel::int8_gemm, kFirstTiledClass);
    Stage::Ptr fwht = make_stage<TernoclStageGenerator>(TernoclKernel::fwht, 0);

    fully_connected_ternocl_int2() : PrimitiveImplOCL("ternocl_int2") {}
    fully_connected_ternocl_int2(const program_node& node, const RuntimeParams& params) : fully_connected_ternocl_int2() {
        const auto& desc = params.typed_desc<fully_connected>();
        // The weights input is the u2 constant at build time and the packed one once this reorder is folded.
        const auto& wei_layout = params.get_input_layout(1);
        auto packed_layout = wei_layout;
        packed_layout.format = format::is_os_yx_isv16_ternary;
        _weights_reorder_params = std::make_shared<WeightsReorderParams>(wei_layout, packed_layout);
        const bool int8 = node.get_program().get_config().get_ternocl_int2_int8_prefill();
        auto add = [&](Stage::Ptr& stage) {
            add_stage(stage, params);
            OPENVINO_ASSERT(has_stage(stage), "[GPU] ternocl int2: failed to generate a kernel for ", node.id());
        };
        if (desc->hadamard_block != 0)
            add(fwht);
        if (params.is_dynamic()) {
            for (size_t cls = 0; cls < (int8 ? kFirstTiledClass : kLaunchClasses); ++cls)
                add(upcvt[cls]);
            if (int8) {
                add(quant);
                for (auto& stage : int8_gemm)
                    add(stage);
            }
        } else {
            const size_t cls = launch_class(fc_shape(params).M);
            if (int8 && cls >= kFirstTiledClass) {
                add(quant);
                add(int8_gemm[cls - kFirstTiledClass]);
            } else {
                add(upcvt[cls]);
            }
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<fully_connected_ternocl_int2>(this);
        copy->_had_signs = _had_signs;
        return copy;
    }

    // Scratch comes from the network memory pool, so it is shared across primitives
    // and resized with M: [had] f16 [M, K] rotated activation, [int8] f16 [K/128, LDSA(M)] scales.
    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        if (params.output_layouts[0].is_dynamic())
            return {};
        const auto s = fc_shape(params);
        std::vector<BufferDescriptor> descs;
        if (has_stage(fwht))
            descs.emplace_back(s.M * s.K, ov::element::f16);
        if (runs_int8(s.M))
            descs.emplace_back((s.K / kTernoclGroupSize) * ((s.M + 31) & ~size_t{31}), ov::element::f16);
        return descs;
    }

    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const kernel_impl_params& params) const override {
        if (!is_dynamic())
            return _order;
        const size_t M = fc_shape(params).M;
        const size_t cls = launch_class(M);
        std::vector<size_t> order;
        if (has_stage(fwht))
            order.push_back(stage_index(fwht));
        if (runs_int8(M)) {
            order.push_back(stage_index(quant));
            order.push_back(stage_index(int8_gemm[cls - kFirstTiledClass]));
        } else {
            order.push_back(stage_index(upcvt[cls]));
        }
        return order;
    }

    // Runs the stages back to back and returns the last stage's event.
    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        if (instance.can_be_optimized())
            return instance.get_network().get_stream().aggregate_events(events, events.size() > 1, instance.is_output());
        update_rt_params(instance);
        std::vector<event::ptr> deps = events;
        if (has_stage(fwht) && !_had_signs) {
            // Not uploaded at impl creation: impls are created in parallel, and the service stream
            // is not thread-safe on Level Zero.
            const auto& signs = instance.get_impl_params()->typed_desc<fully_connected>()->hadamard_signs;
            if (!signs.empty()) {
                OPENVINO_ASSERT(signs.size() == fc_shape(*instance.get_impl_params()).K, "[GPU] ternocl int2: hadamard signs length mismatch");
                _had_signs = instance.get_network().get_engine().allocate_memory(
                    layout{ov::PartialShape{static_cast<int64_t>(signs.size())}, data_types::i8, format::bfyx},
                    allocation_type::usm_device,
                    false);
                deps.push_back(_had_signs->copy_from(instance.get_network().get_stream(), signs.data(), false));
                for (auto& stage : _stages)
                    stage->kd.need_args_update = true;
            }
        }
        event::ptr ev;
        for (const auto i : get_stages_execution_order(*instance.get_impl_params())) {
            ev = execute_stage(deps, instance, *_stages[i]);
            deps = {ev};
        }
        return ev;
    }

    [[nodiscard]] kernel_arguments_data get_arguments(const primitive_inst& instance) const override {
        const auto& fc = static_cast<const fully_connected_inst&>(instance);
        const auto& params = *instance.get_impl_params();
        const auto ep = epilogue(params);
        const auto in = instance.input_memory_ptr(0);
        // Dependencies are input, weights, [bias], scale, [zero point].
        const size_t scale_dep = params.typed_desc<fully_connected>()->bias.is_valid() ? 3 : 2;

        kernel_arguments_data args;
        // Unused Other / Bias / signs slots are never read, but need a valid buffer.
        args.inputs = {in,
                       fc.weights_memory(),
                       instance.dep_memory_ptr(scale_dep),
                       ternocl_postop_reads_other(ep.postop) ? instance.dep_memory_ptr(ep.other_dep) : in,
                       ep.postop == TernoclPostop::bias ? fc.bias_memory() : in,
                       _had_signs ? _had_signs : in};
        args.outputs = {instance.output_memory_ptr(0)};
        const auto& scratch = instance.get_intermediates_memories();
        args.intermediates = {scratch.begin(), scratch.end()};
        return args;
    }

private:
    // hadamard_signs as i8 [K], uploaded at the first execution.
    memory::ptr _had_signs;

    template <size_t Count>
    std::array<Stage::Ptr, Count> make_stages(TernoclKernel kind, size_t first_class) {
        std::array<Stage::Ptr, Count> stages;
        for (size_t i = 0; i < Count; ++i)
            stages[i] = make_stage<TernoclStageGenerator>(kind, first_class + i);
        return stages;
    }

    [[nodiscard]] size_t stage_index(const Stage::Ptr& stage) const {
        return static_cast<size_t>(std::find(_stages.begin(), _stages.end(), stage.get()) - _stages.begin());
    }

    // The int8 prefill runs when its stages are built and M > 8.
    [[nodiscard]] bool runs_int8(size_t M) const {
        return has_stage(quant) && (!is_dynamic() || M > 8);
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

    const auto& desc = params.typed_desc<fully_connected>();
    GPU_DEBUG_TRACE << "[ternocl-int2] create " << node.id() << " N=" << fc_shape(params).N << " K=" << fc_shape(params).K
                    << " postop=" << static_cast<int>(epilogue(params).postop) << " out_f32=" << epilogue(params).out_f32
                    << " hadamard=" << desc->hadamard_block << " int8_prefill=" << node.get_program().get_config().get_ternocl_int2_int8_prefill()
                    << (params.is_dynamic() ? " dynamic" : " static") << std::endl;

    return std::make_unique<fully_connected_ternocl_int2>(node, params);
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
