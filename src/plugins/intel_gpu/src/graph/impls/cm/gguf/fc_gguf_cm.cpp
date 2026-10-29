// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// fc_gguf_cm.cpp — CM (C-for-Metal) dpas GEMM implementation for GGUF FC.
//
// Two-phase operation:
//   Prefill (M > threshold): gemm_q{4k,5k,6k,40,41,80}_full.cm — systolic dpas GEMM
//     with fp16/fp32 output (accumulation in fp32 registers, with output type and
//     store width selected by JIT).  Activation input is always fp16 on the wire:
//     when the runtime activation is fp32, convert_act_f32_to_f16.cm first converts
//     it into a scratch fp16 buffer (see FCGGUFCmImpl::execute_cm_gemm), so the big
//     GEMM kernels themselves never need a second input ABI.
//   Decode  (M <= threshold): fc_gguf_q{4k,5k,6k}_sg.cl or fc_gguf_q{4_0,4_1,8_0}_sg.cl
//     (OCL sub-group-block-read shuffle GEMV, same kernel as in FCGGUFOptImpl).
//     This OCL kernel JIT-selects its activation type directly, so it natively
//     supports fp16 and fp32 activation without any conversion pre-pass.

#include "fc_gguf_cm.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "../../ocl_v2/gguf/fc_gguf_shuffle_gen.hpp"
#include "../../ocl_v2/utils/jitter.hpp"
#include "../../ocl_v2/utils/kernel_generator.hpp"
#include "../primitive_cm_base.hpp"
#include "../utils/kernels_db.hpp"  // cm::SourcesDB::get_kernel_template() for .cm files
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::cm {

using namespace ov::intel_gpu::ocl;

namespace {

// -------------------------------------------------------------------------
// Shared utilities (mirrored from fc_gguf_opt.cpp anonymous namespace)
// -------------------------------------------------------------------------

// GGUF_GEMV_SG_SIZE, gguf_shuffle_choose_ksplit, and the FCGGUFQ*SgGenerator
// shuffle GEMV classes come from the shared header (fc_gguf_shuffle_gen.hpp)
// via the using-directive above.
constexpr int GGUF_CM_OPG = GGUF_GEMV_SG_SIZE;  // outputs per subgroup in the .cm kernels
constexpr int GGUF_CM_TOKENS_PER_TILE = 8;      // dpas RepeatCount (fixed by the ISA)

// f32 -> f16 activation conversion pre-pass (see convert_act_f32_to_f16.cm).
// CONV_ACT_BLOCK must divide 256 (the CM GEMM K-tiling granularity, enforced
// in FCGGUFCm::support_shapes), so total_elems = M*K is always an exact
// multiple of it -- no tail handling needed in the kernel.
constexpr int CONV_ACT_BLOCK = 64;  // elements converted per thread
constexpr int CONV_ACT_LOCAL = 16;  // threads per work-group

// Flattened activation rows BM = product of all activation dims except the last (K).
size_t derive_bm_cm(const ov::Shape& shape_a) {
    size_t bm = 1;
    for (size_t i = 0; i + 1 < shape_a.size(); ++i) {
        bm *= shape_a[i];
    }
    return bm;
}

// Prefill threshold: M above this uses the CM GEMM.
// Overridable via OV_GPU_GGUF_PREFILL_THRESHOLD (same env var as OCL impl).
size_t cm_prefill_threshold() {
    if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_THRESHOLD")) {
        const long v = std::atol(env);
        if (v >= 0)
            return static_cast<size_t>(v);
    }
    return 32;
}

// Per-node CM GEMM tuned configuration (architecture-dependent).
struct CmGemmConfig {
    int row_groups;    // N-dimension blocking (ROW_GROUPS)
    int token_groups;  // token dimension register blocking (TOKEN_GROUPS)
    int token_local;   // threads per work-group (TOKEN_LOCAL) — CORRECTNESS requirement
    int row_blocks;    // ROW_BLOCKS (additional N-block split within the work-group)
    int slm_jblk;      // sub-blocks published per COOP_SLM phase (or -1 = kernel default)
    int pf_a;          // PF_A prefetch flag (or -1 = kernel default)
    int a_outer;       // A_OUTER loop order flag
};

CmGemmConfig pick_cm_gemm_config(const RuntimeParams& params, size_t K) {
    const auto arch = params.get_device_info().arch;
    const bool is_xe3 = (arch >= gpu_arch::xe3);
    (void)K;  // no PTL long-K promotion any more, see the comment below

    if (is_xe3) {
        // PTL (Arc B390 iGPU, Xe3): ADOPTED config from the standalone
        // cm_gguf_gemm_kernel/ ab_lowbit_config.py PTL round-2 tuning.
        // ROW_GROUPS=4/ROW_BLOCKS=1/TOKEN_LOCAL=16 measured 1.01-1.63x
        // faster than this older RG=2/RB=2/TL=32 tile, across ALL SIX
        // quants and every shape (K 1024..16384, token_len 64..8192,
        // interleaved min-of-N -- see cm_kernel_dev.md repo memory / the
        // "PTL round 2" notes). This branch had gone stale: it still
        // shipped the PRE-round-2 tile, which is why re-measuring
        // gemm_q41_full on the real PTL box (ab-style interleaved A/B)
        // showed the old branch 1.07-1.43x SLOWER than the adopted one at
        // K/N=4096/4096..12288/4096, token_len=1024 -- a config mismatch,
        // NOT register spill (both variants build to <=121/128 registers,
        // no spill, with or without post-ops; verified via
        // -mCM_printregusage on the actual Arc B390).
        // Do NOT re-add the old RG=2 tile's "K>=10240 -> TOKEN_GROUPS=4"
        // long-K promotion on top of this: RG*TG would become 8 KB of
        // accumulators and spill (measured 0.19-0.27x). RG=4 already wins
        // at long K directly (1.54-1.63x at K=12288/16384).
        return CmGemmConfig{/*row_groups=*/4,
                            /*token_groups=*/2,
                            /*token_local=*/16,
                            /*row_blocks=*/1,
                            /*slm_jblk=*/4,
                            /*pf_a=*/1,
                            /*a_outer=*/1};
    }
    // BMG (Arc B580, Xe2) — default, tuned for 160 XVE @ 2.9 GHz.
    return CmGemmConfig{/*row_groups=*/2,
                        /*token_groups=*/4,
                        /*token_local=*/8,
                        /*row_blocks=*/1,
                        /*slm_jblk=*/-1,
                        /*pf_a=*/-1,
                        /*a_outer=*/1};
}

// ==========================================================================
// Shared CM kernel-data builder.
//
// The ocl::KernelGenerator base always builds an OCLC_V2 kernel_string loaded
// from the OCL source DB.  CM kernels need language=CM and the CM SourcesDB;
// every CM-native generator in this file (the big GEMM kernels and the small
// f32->f16 activation-conversion kernel) shares that boilerplate here.
// ==========================================================================
class CmKernelGeneratorBase : public KernelGenerator {
public:
    explicit CmKernelGeneratorBase(const char* kernel_name) : KernelGenerator(kernel_name) {}

protected:
    // Override get_kernel_data to produce a CM (not OCLC_V2) kernel descriptor.
    // The base ocl::KernelGenerator::get_kernel_data() always sets language=OCLC_V2
    // and loads templates from the OCL source DB.  For CM kernels we must set
    // language=CM and load from the CM SourcesDB so the kernel compiles correctly.
    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override {
        const auto jit = get_jit_constants(params);

        KernelData kd;
        kd.code = std::make_shared<KernelString>();
        kd.code->language = KernelLanguage::CM;
        kd.code->entry_point = get_entry_point(params);
        kd.code->jit = "";
        kd.code->undefs = "";
        kd.code->options = get_build_options(params);
        kd.code->batch_compilation = false;  // Each .cm file has its own #include<cm/cm.h>; do NOT batch
        kd.code->has_microkernels = false;

        // Prepend JIT #define macros, then the .cm kernel source, then #undef.
        // Some jit constants are function-like macros (name contains a
        // parameter list, e.g. "CM_APPLY_POST_OP_0(V, BASE, N)"); #undef only
        // accepts the bare macro identifier, so strip everything from the
        // first '(' onward for the #undef line.
        std::string src;
        for (const auto& c : jit) {
            src += "#define " + c.name + " " + c.value + "\n";
        }
        src += std::string(ov::intel_gpu::cm::SourcesDB::get_kernel_template(get_kernel_name()));
        for (const auto& c : jit) {
            const auto paren = c.name.find('(');
            src += "#undef " + c.name.substr(0, paren) + "\n";
        }
        kd.code->str = std::move(src);

        kd.params.arguments = get_arguments_desc(params);
        kd.params.layerID = params.desc->id;
        kd.update_dispatch_data_func = get_dispatch_data_func();
        kd.need_args_update = true;
        kd.need_dispatch_data_update = true;

        return kd;
    }
};

// ==========================================================================
// CM GEMM kernel generator base.
//
// One subclass per GGUF format so each gets its own kernel binary / cache slot.
// The generator is shape-agnostic with respect to M (passed as scalar at run
// time) and static for K / N (baked into scalars at dispatch time).
//
// Activation is always fp16 by the time this kernel runs: an fp32 activation
// is first converted to a scratch fp16 buffer by convert_act_f32_to_f16.cm
// (see FCGGUFCmImpl::execute_cm_gemm), so this kernel never needs a second
// input ABI.
// ==========================================================================
class FCGGUFCmGemmGenerator : public CmKernelGeneratorBase {
public:
    explicit FCGGUFCmGemmGenerator(const char* kernel_name) : CmKernelGeneratorBase(kernel_name) {}

protected:
    // Build options: -cmc + architecture-specific tile macros.
    // Do NOT call the ocl::KernelGenerator base class get_build_options() here;
    // it returns OCL-specific flags that are incompatible with the CM compiler.
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t K = in1.get_shape()[1];
        const CmGemmConfig cfg = pick_cm_gemm_config(params, K);

        std::string opts = " -cmc";
        opts += " -DROW_GROUPS=" + std::to_string(cfg.row_groups);
        opts += " -DTOKEN_GROUPS=" + std::to_string(cfg.token_groups);
        opts += " -DTOKEN_LOCAL=" + std::to_string(cfg.token_local);
        opts += " -DROW_BLOCKS=" + std::to_string(cfg.row_blocks);
        opts += " -DA_OUTER=" + std::to_string(cfg.a_outer);
        if (params.get_output_layout(0).data_type == data_types::f32)
            opts += " -DCM_OUTPUT_F32=1";
        if (cfg.slm_jblk >= 0)
            opts += " -DSLM_JBLK=" + std::to_string(cfg.slm_jblk);
        if (cfg.pf_a >= 0)
            opts += " -DPF_A=" + std::to_string(cfg.pf_a);
        return opts;
    }

    // The entry point MUST match the literal function name in the .cm source file.
    // Unlike OCL kernels (which use a KERNEL_NAME macro), our CM GEMM kernels use
    // fixed function names.  K and N are passed as runtime scalars, so the same
    // binary serves all shapes — no per-shape suffix needed.
    [[nodiscard]] std::string get_entry_point(const RuntimeParams&) const override {
        return get_kernel_name();  // e.g. "gemm_q41_full" — matches void in the .cm file
    }

    // Generate CM-native post-ops JIT constants from params.fused_desc.
    // We do NOT use the OCL FUSED_OPS / FUSED_OPS_RESULT system because OCL
    // macros contain OCL-specific type conversions (convert_half, __global, etc.)
    // that don't compile in CM.  Instead we emit:
    //   CM_POST_OPS_NUM        — number of post-ops
    //   CM_POST_OPS_DECLS      — extra kernel signature parameters (outer-dep inputs)
    //   CM_APPLY_POST_OP_N(V, BASE, N) — function-like macro, VECTORIZED: it
    //       mutates the WHOLE output vector `V` (width N, base output element
    //       index BASE) in a couple of SIMD instructions.
    //
    // IMPORTANT: this must stay vectorized. An earlier version applied post-ops
    // with a scalar per-lane statement (`v`/`out_idx`) wrapped in a
    // `#pragma unroll for (int _j = 0; _j < N; _j++)` loop at the epilogue.
    // Since the epilogue itself is already unrolled over
    // TOKEN_GROUPS*TOKENS_PER_TILE (=32) token stores, that produced up to
    // ~1024 unrolled scalar float ops PER THREAD, PLUS N scalar
    // (non-coalesced) single-element loads of the fused eltwise operand
    // instead of one contiguous vector load -- this caused a large measured
    // performance regression (bloated code size / register pressure / scalar
    // memory traffic), not classic dpas-loop register spill.
    // Reorder fused ops are layout-only and intentionally skipped.
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        if (!params.has_fused_primitives())
            return {};

        // Collect actionable (non-reorder) fused ops and their apply statements.
        struct PostOp {
            std::string apply_stmt;  // CM code; references vector `V` (float, width N) and uint `BASE`
            bool has_input;          // true if references cm_post_inN pointer
            size_t input_idx;        // index of the outer-dep input pointer (0, 1, ...)
        };
        std::vector<PostOp> ops;
        size_t n_inputs = 0;
        // input_dtypes[input_idx] = actual runtime dtype of that outer-dep memory
        // (fused eltwise operands are NOT always fp16 -- e.g. a residual Add
        // feeding an fp32-output FC node has an fp32 outer dep).
        std::vector<data_types> input_dtypes;

        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<cldnn::reorder>())
                continue;

            PostOp op{};
            op.has_input = fd.has_outer_dep();
            op.input_idx = op.has_input ? n_inputs++ : 0;
            if (op.has_input) {
                input_dtypes.push_back(params.input_layouts[fd.outer_dep_start_idx].data_type);
            }

            if (fd.is_type<cldnn::activation>()) {
                const auto act = fd.typed_desc<cldnn::activation>();
                switch (act->activation_function) {
                case cldnn::activation_func::swish:
                    // swish(x) = x / (1 + exp(-x))
                    // CM has no expf(); use cm_exp(x*log2e) = e^x  (base-2 hardware
                    // exp, vector form -- cm_exp deduces width from its argument).
                    op.apply_stmt = "V = V / (1.0f + cm_exp(-V * 1.4426950408889634f));";
                    break;
                case cldnn::activation_func::relu:
                    op.apply_stmt = "V = cm_max<float>(V, 0.0f);";
                    break;
                case cldnn::activation_func::gelu:
                case cldnn::activation_func::gelu_tanh:
                    // gelu(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
                    // tanh via cm_exp: tanh(a) = (e^2a - 1) / (e^2a + 1)
                    op.apply_stmt = "{ vector<float, N> _ga = 0.7978845608028654f * "
                                    "(V + 0.044715f * V * V * V); "
                                    "vector<float, N> _ge2a = cm_exp(2.0f * _ga * 1.4426950408889634f); "
                                    "V = V * 0.5f * (1.0f + (_ge2a - 1.0f) * cm_inv(_ge2a + 1.0f)); }";
                    break;
                default:
                    op.apply_stmt = "";  // unknown activation — pass-through
                    break;
                }
            } else if (fd.is_type<cldnn::eltwise>() && op.has_input) {
                const auto elt = fd.typed_desc<cldnn::eltwise>();
                const std::string ptr = "cm_post_in" + std::to_string(op.input_idx);
                // Fused operand is contiguous (same layout/strides as `outputs`),
                // so load it as ONE vector instead of N scalar element loads.
                // The load's element type is baked in statically per the
                // operand's known runtime dtype (see input_dtypes above).
                const bool is_f32 =
                    op.input_idx < input_dtypes.size() && input_dtypes[op.input_idx] == data_types::f32;
                std::string load;
                if (is_f32) {
                    load = "vector<float, N> _pin = cm_ptr_load<float, N>((float*)" + ptr + ", (BASE) * 4u);";
                } else {
                    load = "vector<half, N> _pinh = cm_ptr_load<uint, N/2>((uint*)" + ptr +
                           ", (BASE) * 2u).format<half>(); vector<float, N> _pin; _pin = _pinh;";
                }
                switch (elt->mode) {
                case cldnn::eltwise_mode::sum:
                    op.apply_stmt = "{ " + load + " V += _pin; }";
                    break;
                case cldnn::eltwise_mode::prod:
                    op.apply_stmt = "{ " + load + " V *= _pin; }";
                    break;
                default:
                    op.apply_stmt = "";  // unsupported mode — skip
                    break;
                }
            }

            if (!op.apply_stmt.empty())
                ops.push_back(std::move(op));
        }

        if (ops.empty())
            return {};

        JitConstants jit;
        jit.add(make_jit_constant("CM_POST_OPS_NUM", static_cast<int>(ops.size())));

        for (size_t i = 0; i < ops.size(); i++) {
            // Function-like macro: CM_APPLY_POST_OP_i(V, BASE, N) mutates the
            // WHOLE vector V (width N, base output element index BASE).
            jit.add(make_jit_constant("CM_APPLY_POST_OP_" + std::to_string(i) + "(V, BASE, N)",
                                       std::string(ops[i].apply_stmt)));
        }

        // Build the kernel signature extension for outer-dep pointer arguments.
        if (n_inputs > 0) {
            std::string sig;
            for (size_t i = 0; i < n_inputs; i++) {
                // Declare each fused-op pointer with its actual runtime element
                // type; assuming fp16 unconditionally silently misreads any
                // fp32 fused operand (half the bytes, garbage values).
                const char* elem_type = (input_dtypes[i] == data_types::f32) ? "float" : "half";
                sig += ", ";
                sig += elem_type;
                sig += "* cm_post_in" + std::to_string(i) + " [[type(\"svmptr_t\")]]";  // SVM pointer for GPU USM
            }
            jit.add(make_jit_constant("CM_POST_OPS_DECLS", std::string(sig)));
        }

        return jit;
    }

    // Argument binding: INPUT(0)=activation, INPUT(1)=weight, OUTPUT(0)=output,
    // SCALAR(0)=M(token_len), SCALAR(1)=K(input_len), SCALAR(2)=N(output_len),
    // then INPUT_OF_FUSED_PRIMITIVE(j) for each non-reorder fused op with an outer dep.
    //
    // NOTE: INPUT_OF_FUSED_PRIMITIVE uses an index into the runtime's fused_op_inputs array,
    // which is indexed 0..get_fused_mem_count()-1 (sequential, includes ALL fused outer deps).
    // outer_dep_start_idx is the absolute index in the node's full dep list; we must subtract
    // the fused_mem_offset (= min outer_dep_start_idx across all fused deps) to get the slot.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});   // activation
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});   // weight (shuffled)
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});  // fp16/fp32 output
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // token_len (M)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // input_len (K)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // output_len (N)

        if (!params.has_fused_primitives())
            return args;

        // Compute fused_mem_offset = minimum outer_dep_start_idx across ALL fused ops with outer deps.
        // This is the index where fused dep memories BEGIN in the node's full deps list.
        int fused_mem_offset = -1;
        for (const auto& fd : params.fused_desc) {
            if (fd.outer_dep_start_idx >= 0) {
                if (fused_mem_offset < 0 || fd.outer_dep_start_idx < fused_mem_offset)
                    fused_mem_offset = fd.outer_dep_start_idx;
            }
        }
        if (fused_mem_offset < 0)
            return args;  // no outer deps at all

        // Add outer-dep inputs for non-reorder fused ops.
        // The slot index j = outer_dep_start_idx - fused_mem_offset gives the position in
        // the runtime's fused_op_inputs[] array (which holds all outer dep memories in order).
        for (const auto& fd : params.fused_desc) {
            if (fd.is_type<cldnn::reorder>())
                continue;  // layout-only, skip
            if (fd.has_outer_dep()) {
                const auto slot = static_cast<uint32_t>(fd.outer_dep_start_idx - fused_mem_offset);
                args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, slot});
            }
        }
        return args;
    }

    // Compute global/local work sizes and scalar values from runtime shapes.
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& in0 = params.get_input_layout(0);
            const auto& in1 = params.get_input_layout(1);
            const size_t N = in1.get_shape()[0];
            const size_t K = in1.get_shape()[1];
            const size_t M = derive_bm_cm(in0.get_shape());

            const CmGemmConfig cfg = pick_cm_gemm_config(params, K);

            const size_t tokens_per_thread = GGUF_CM_TOKENS_PER_TILE * static_cast<size_t>(cfg.token_groups);
            const size_t token_slots = static_cast<size_t>(cfg.token_local) / static_cast<size_t>(cfg.row_blocks);
            const size_t ntiles = (M + tokens_per_thread - 1) / tokens_per_thread;
            // ngroups = N / (OPG * ROW_GROUPS * ROW_BLOCKS)
            const size_t ngroups = N / (GGUF_CM_OPG * static_cast<size_t>(cfg.row_groups) * static_cast<size_t>(cfg.row_blocks));
            const size_t gsize_y = ((ntiles + token_slots - 1) / token_slots) * static_cast<size_t>(cfg.token_local);

            auto& wgs = kd.params.workGroups;
            wgs.global = {ngroups, gsize_y, 1};
            wgs.local = {1, static_cast<size_t>(cfg.token_local), 1};

            // Scalar values: M (uint32), K (uint32), N (uint32).
            kd.params.scalars.clear();
            auto push_u32 = [&](size_t v) {
                scalar_desc sd{};
                sd.t = scalar_desc::Types::UINT32;
                sd.v.u32 = static_cast<uint32_t>(v);
                kd.params.scalars.push_back(sd);
            };
            push_u32(M);
            push_u32(K);
            push_u32(N);
        }};
    }
};

// Per-format CM GEMM generator subclasses (each maps to its own .cm file).
class FCGGUFCmGemmQ4kGenerator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ4kGenerator() : FCGGUFCmGemmGenerator("gemm_q4k_full") {}
};
class FCGGUFCmGemmQ5kGenerator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ5kGenerator() : FCGGUFCmGemmGenerator("gemm_q5k_full") {}
};
class FCGGUFCmGemmQ6kGenerator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ6kGenerator() : FCGGUFCmGemmGenerator("gemm_q6k_full") {}
};
class FCGGUFCmGemmQ40Generator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ40Generator() : FCGGUFCmGemmGenerator("gemm_q40_full") {}
};
class FCGGUFCmGemmQ41Generator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ41Generator() : FCGGUFCmGemmGenerator("gemm_q41_full") {}
};
class FCGGUFCmGemmQ80Generator : public FCGGUFCmGemmGenerator {
public:
    FCGGUFCmGemmQ80Generator() : FCGGUFCmGemmGenerator("gemm_q80_full") {}
};

// ==========================================================================
// f32 -> f16 activation conversion kernel generator (convert_act_f32_to_f16.cm).
//
// Explicit-args only: dispatched via FCGGUFCmImpl::execute_stage_explicit(),
// so get_arguments_desc()/get_dispatch_data_func() are left trivial (mirrors
// the pattern used by FCGGUFShuffleGenerator in fc_gguf_shuffle_gen.hpp).
// ==========================================================================
class FCGGUFCmActConvertGenerator : public CmKernelGeneratorBase {
public:
    FCGGUFCmActConvertGenerator() : CmKernelGeneratorBase("convert_act_f32_to_f16") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams&) const override {
        return " -cmc -DCONV_BLOCK=" + std::to_string(CONV_ACT_BLOCK) + " -DCONV_LOCAL=" + std::to_string(CONV_ACT_LOCAL);
    }
    [[nodiscard]] std::string get_entry_point(const RuntimeParams&) const override {
        return get_kernel_name();
    }
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// ==========================================================================
// FCGGUFCmImpl — the primitive implementation
// ==========================================================================
class FCGGUFCmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::FCGGUFCmImpl)

    // ---- CM GEMM stages (prefill) ----------------------------------------
    Stage::Ptr q4k_gemm_stage = make_stage<FCGGUFCmGemmQ4kGenerator>();
    Stage::Ptr q5k_gemm_stage = make_stage<FCGGUFCmGemmQ5kGenerator>();
    Stage::Ptr q6k_gemm_stage = make_stage<FCGGUFCmGemmQ6kGenerator>();
    Stage::Ptr q40_gemm_stage = make_stage<FCGGUFCmGemmQ40Generator>();
    Stage::Ptr q41_gemm_stage = make_stage<FCGGUFCmGemmQ41Generator>();
    Stage::Ptr q80_gemm_stage = make_stage<FCGGUFCmGemmQ80Generator>();

    // ---- OCL shuffle GEMV stages (decode) — from fc_gguf_shuffle_gen.hpp ---
    Stage::Ptr q4k_sg_stage = make_stage<FCGGUFQ4kSgGenerator>();
    Stage::Ptr q5k_sg_stage = make_stage<FCGGUFQ5kSgGenerator>();
    Stage::Ptr q6k_sg_stage = make_stage<FCGGUFQ6kSgGenerator>();
    Stage::Ptr q40_sg_stage = make_stage<FCGGUFQ40SgGenerator>();
    Stage::Ptr q41_sg_stage = make_stage<FCGGUFQ41SgGenerator>();
    Stage::Ptr q80_sg_stage = make_stage<FCGGUFQ80SgGenerator>();

    // ---- f32 -> f16 activation conversion stage (prefill, fp32-input only) ---
    Stage::Ptr act_f32_to_f16_stage = make_stage<FCGGUFCmActConvertGenerator>();

    element::Type_t m_weight_type = element::Type_t::f32;  // placeholder; set in (node,params) ctor
    size_t m_prefill_threshold = 32;

    // Scratch buffer holding the fp16-converted activation for the fp32-input
    // path (grow-only, lazily (re)allocated in ensure_act_f16_scratch()).
    cldnn::memory::ptr m_act_f16_scratch;
    size_t m_act_f16_scratch_elems = 0;

    FCGGUFCmImpl() : PrimitiveImplCM(FCGGUFCm::get_type_info_static()) {
        m_prefill_threshold = cm_prefill_threshold();
    }

    FCGGUFCmImpl(const program_node& node, const RuntimeParams& params) : FCGGUFCmImpl() {
        m_weight_type = static_cast<element::Type_t>(params.input_layouts[1].data_type);

        // Add the CM GEMM stage for this format.
        switch (m_weight_type) {
        case element::Type_t::gguf_q4_k:
            add_stage(q4k_gemm_stage, params);
            add_stage(q4k_sg_stage, params);
            break;
        case element::Type_t::gguf_q5_k:
            add_stage(q5k_gemm_stage, params);
            add_stage(q5k_sg_stage, params);
            break;
        case element::Type_t::gguf_q6_k:
            add_stage(q6k_gemm_stage, params);
            add_stage(q6k_sg_stage, params);
            break;
        case element::Type_t::gguf_q4_0:
            add_stage(q40_gemm_stage, params);
            add_stage(q40_sg_stage, params);
            break;
        case element::Type_t::gguf_q4_1:
            add_stage(q41_gemm_stage, params);
            add_stage(q41_sg_stage, params);
            break;
        case element::Type_t::gguf_q8_0:
            add_stage(q80_gemm_stage, params);
            add_stage(q80_sg_stage, params);
            break;
        default:
            OPENVINO_THROW("[GPU] FCGGUFCmImpl: unsupported weight type ", element::Type(m_weight_type).get_type_name());
        }

        // Weight-format independent: needed whenever the runtime activation
        // dtype turns out to be fp32 (see execute_cm_gemm()).
        add_stage(act_f32_to_f16_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<FCGGUFCmImpl>(this);
        auto* c = static_cast<FCGGUFCmImpl*>(copy.get());
        c->m_weight_type = m_weight_type;
        c->m_prefill_threshold = m_prefill_threshold;
        return copy;
    }

    // Bind activation (INPUT 0), weight (INPUT 1), and output for the CM GEMM and
    // OCL GEMV stages that use the standard dispatch path.
    //
    // When the runtime activation is fp32, the CM GEMM stage reads from
    // m_act_f16_scratch (already converted+filled by execute_cm_gemm() before
    // this is called) instead of the raw fp32 dep memory; the OCL GEMV decode
    // path never calls get_arguments() (it uses execute_stage_explicit()), so
    // it is unaffected and keeps consuming the raw fp32 buffer directly.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);

        const auto& in0 = instance.get_impl_params()->get_input_layout(0);
        if (in0.data_type == data_types::f32 && m_act_f16_scratch) {
            data.inputs.push_back(m_act_f16_scratch);  // fp16-converted activation
        } else {
            data.inputs.push_back(instance.dep_memory_ptr(0));  // activation
        }
        data.inputs.push_back(fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1));

        // Fused-op inputs for the FUSED_OPS decode path.
        if (instance.has_fused_primitives()) {
            const size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; ++i)
                data.fused_op_inputs.push_back(instance.fused_memory(i));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); ++i)
            data.outputs.push_back(instance.output_memory_ptr(i));
        data.shape_info = instance.shape_info_memory_ptr();
        return data;
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        const auto& params = *instance.get_impl_params();
        const auto& in0 = params.get_input_layout(0);

        // Determine whether to use CM GEMM (prefill) or OCL GEMV (decode).
        const bool shape_known = !in0.is_dynamic();
        const size_t M = shape_known ? derive_bm_cm(in0.get_shape()) : 0;

        // Use CM GEMM when M exceeds the prefill threshold.
        // Fused ops (swish, mul, add) are applied inline at output-write time
        // via CM_APPLY_POST_OP_N macros generated by get_jit_constants().
        const bool use_cm_gemm = shape_known && (M > m_prefill_threshold);

        if (use_cm_gemm) {
            return execute_cm_gemm(events, instance, M);
        }

        // Decode / fallback: OCL sub-group-block-read shuffle GEMV.
        // This handles fused ops via FUSED_OPS and is correct for all M values.
        return execute_shuffle_gemv_decode(events, instance);
    }

private:
    // Bring the base execute_stage overloads into scope.
    using PrimitiveImplCM::execute_stage;

    // Explicit-args stage dispatch helper (same pattern as FCGGUFOptImpl).
    cldnn::event::ptr execute_stage_explicit(const std::vector<cldnn::event::ptr>& events,
                                             cldnn::primitive_inst& instance,
                                             Stage& stage,
                                             const std::vector<cldnn::memory::ptr>& inputs,
                                             const std::vector<cldnn::memory::ptr>& outputs,
                                             const std::vector<size_t>& global,
                                             const std::vector<size_t>& local,
                                             const std::vector<cldnn::memory::ptr>& fused_inputs = {},
                                             bool needs_shape_info = false,
                                             const std::vector<scalar_desc>& scalars = {}) const {
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;
        if (needs_shape_info) {
            desc.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.shape_info = instance.shape_info_memory_ptr();
        }
        for (uint32_t i = 0; i < inputs.size(); ++i) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }
        for (uint32_t i = 0; i < outputs.size(); ++i) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
        }
        for (uint32_t i = 0; i < fused_inputs.size(); ++i) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, i});
            args.fused_op_inputs.push_back(fused_inputs[i]);
        }
        if (!scalars.empty()) {
            for (uint32_t i = 0; i < scalars.size(); ++i)
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, i});
            desc.scalars = scalars;
            args.scalars = &desc.scalars;
        }
        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;
        kernel_dump_info.add_entry_point(stage.kernel->get_id());
        return stream.enqueue_kernel(*stage.kernel,
                                     desc,
                                     {},
                                     events,
                                     /*needs_completion_event=*/false);
    }

    // Select the GEMM stage for the current weight type.
    Stage& get_gemm_stage() {
        switch (m_weight_type) {
        case element::Type_t::gguf_q4_k:
            return *q4k_gemm_stage;
        case element::Type_t::gguf_q5_k:
            return *q5k_gemm_stage;
        case element::Type_t::gguf_q6_k:
            return *q6k_gemm_stage;
        case element::Type_t::gguf_q4_0:
            return *q40_gemm_stage;
        case element::Type_t::gguf_q4_1:
            return *q41_gemm_stage;
        case element::Type_t::gguf_q8_0:
            return *q80_gemm_stage;
        default:
            OPENVINO_THROW("[GPU] FCGGUFCmImpl: no GEMM stage for weight type ", element::Type(m_weight_type).get_type_name());
        }
    }

    // Select the OCL shuffle GEMV stage for the current weight type.
    Stage& get_sg_stage() {
        switch (m_weight_type) {
        case element::Type_t::gguf_q4_k:
            return *q4k_sg_stage;
        case element::Type_t::gguf_q5_k:
            return *q5k_sg_stage;
        case element::Type_t::gguf_q6_k:
            return *q6k_sg_stage;
        case element::Type_t::gguf_q4_0:
            return *q40_sg_stage;
        case element::Type_t::gguf_q4_1:
            return *q41_sg_stage;
        case element::Type_t::gguf_q8_0:
            return *q80_sg_stage;
        default:
            OPENVINO_THROW("[GPU] FCGGUFCmImpl: no shuffle SG stage for weight type ", element::Type(m_weight_type).get_type_name());
        }
    }

    // ---- Prefill path: CM GEMM with optional in-kernel post-ops ---------
    cldnn::event::ptr execute_cm_gemm(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance, size_t M) {
        Stage& gemm = get_gemm_stage();
        gemm.kd.need_dispatch_data_update = true;
        gemm.kd.need_args_update = true;

        const auto& params = *instance.get_impl_params();
        const auto& in0 = params.get_input_layout(0);
        if (in0.data_type != data_types::f32) {
            return execute_stage(events, instance, gemm);
        }

        // fp32 activation: run a tiny CM kernel that converts it into a
        // scratch fp16 buffer first; get_arguments() (above) then binds that
        // scratch buffer instead of the raw fp32 memory for the GEMM stage.
        const auto& in1 = params.get_input_layout(1);
        const size_t K = in1.get_shape()[1];
        cldnn::event::ptr conv_ev = execute_act_f32_to_f16(events, instance, M * K);
        return execute_stage({conv_ev}, instance, gemm);
    }

    // Grow-only (re)allocation of the fp16 activation scratch buffer.
    void ensure_act_f16_scratch(cldnn::primitive_inst& instance, size_t total_elems) {
        if (m_act_f16_scratch && m_act_f16_scratch_elems >= total_elems)
            return;

        auto& engine = instance.get_network().get_engine();
        auto& stream = instance.get_network().get_stream();
        if (m_act_f16_scratch) {
            // The old (smaller) buffer may still be read by an in-flight GEMM
            // kernel from a previous call; finish the queue before freeing it.
            stream.finish();
        }
        const cldnn::layout scratch_layout(ov::Shape{total_elems}, data_types::f16, cldnn::format::bfyx);
        const auto alloc_type = engine.get_preferred_memory_allocation_type();
        m_act_f16_scratch = engine.allocate_memory(scratch_layout, alloc_type, /*reset=*/false);
        m_act_f16_scratch_elems = total_elems;
    }

    // ---- fp32 -> fp16 activation conversion pre-pass (see convert_act_f32_to_f16.cm) ----
    cldnn::event::ptr execute_act_f32_to_f16(const std::vector<cldnn::event::ptr>& events,
                                             cldnn::primitive_inst& instance,
                                             size_t total_elems) {
        ensure_act_f16_scratch(instance, total_elems);

        Stage& conv = *act_f32_to_f16_stage;
        conv.kd.need_dispatch_data_update = true;
        conv.kd.need_args_update = true;

        const size_t threads = (total_elems + CONV_ACT_BLOCK - 1) / CONV_ACT_BLOCK;
        const size_t global = ((threads + CONV_ACT_LOCAL - 1) / CONV_ACT_LOCAL) * CONV_ACT_LOCAL;

        scalar_desc total_elems_scalar{};
        total_elems_scalar.t = scalar_desc::Types::UINT32;
        total_elems_scalar.v.u32 = static_cast<uint32_t>(total_elems);

        return execute_stage_explicit(events,
                                      instance,
                                      conv,
                                      /*inputs=*/{instance.dep_memory_ptr(0)},
                                      /*outputs=*/{m_act_f16_scratch},
                                      /*global=*/{global, 1, 1},
                                      /*local=*/{static_cast<size_t>(CONV_ACT_LOCAL), 1, 1},
                                      /*fused_inputs=*/{},
                                      /*needs_shape_info=*/false,
                                      /*scalars=*/{total_elems_scalar});
    }

    // ---- Decode path: OCL sub-group-block-read shuffle GEMV ----------------
    cldnn::event::ptr execute_shuffle_gemv_decode(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) {
        const auto& params = *instance.get_impl_params();
        const auto& in0 = params.get_input_layout(0);
        const auto& in1 = params.get_input_layout(1);
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        const size_t M = in0.is_dynamic() ? 1 : derive_bm_cm(in0.get_shape());
        const size_t ksplit = static_cast<size_t>(gguf_shuffle_choose_ksplit(N, K));
        const size_t row_groups = (N + GGUF_GEMV_SG_SIZE - 1) / GGUF_GEMV_SG_SIZE;

        const auto* fc_inst = dynamic_cast<cldnn::fully_connected_inst*>(&instance);
        auto act = instance.dep_memory_ptr(0);
        auto weight = fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1);
        auto out = instance.output_memory_ptr(0);

        // Collect fused-op input memories (for SwiGLU / residual fused onto this FC).
        std::vector<cldnn::memory::ptr> fused_inputs;
        if (instance.has_fused_primitives()) {
            const size_t count = instance.get_fused_mem_count();
            fused_inputs.reserve(count);
            for (size_t i = 0; i < count; ++i)
                fused_inputs.push_back(instance.fused_memory(i));
        }

        Stage& sg = get_sg_stage();
        const bool needs_si = is_dynamic();
        return execute_stage_explicit(events,
                                      instance,
                                      sg,
                                      /*inputs=*/{act, weight},
                                      /*outputs=*/{out},
                                      /*global=*/{row_groups * GGUF_GEMV_SG_SIZE, M, ksplit},
                                      /*local=*/{GGUF_GEMV_SG_SIZE, 1, ksplit},
                                      fused_inputs,
                                      needs_si);
    }
};

}  // namespace

// ==========================================================================
// FCGGUFCm::create_impl
// ==========================================================================
[[nodiscard]] std::unique_ptr<primitive_impl> FCGGUFCm::create_impl(const program_node& node, const RuntimeParams& params) const {
    return std::make_unique<FCGGUFCmImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::FCGGUFCmImpl)
