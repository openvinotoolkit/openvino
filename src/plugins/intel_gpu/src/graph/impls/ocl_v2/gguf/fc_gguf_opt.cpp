// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fc_gguf_opt.hpp"

#include <string>

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "openvino/core/type/element_type.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl.hpp>
#    include <oneapi/dnnl/dnnl_ocl.hpp>

#    include "impls/onednn/utils.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

namespace ov::intel_gpu::ocl {
namespace {

// JIT flag selecting the per-format block decoder (shared by the GEMV and transcode kernels).
const char* gguf_type_jit_flag(element::Type_t t) {
    switch (t) {
    case element::Type_t::gguf_q4_0:
        return "GGUF_IS_Q4_0";
    case element::Type_t::gguf_q8_0:
        return "GGUF_IS_Q8_0";
    case element::Type_t::gguf_q4_k:
        return "GGUF_IS_Q4_K";
    case element::Type_t::gguf_q5_k:
        return "GGUF_IS_Q5_K";
    case element::Type_t::gguf_q6_k:
        return "GGUF_IS_Q6_K";
    default:
        OPENVINO_THROW("[GPU] FCGGUFOpt: no kernel for GGUF element type ", element::Type(t).get_type_name());
    }
}

// Flattened activation rows BM = product of all activation dims except the last (K).
size_t derive_bm(const ov::Shape& shape_a) {
    size_t bm = 1;
    for (size_t i = 0; i + 1 < shape_a.size(); ++i) {
        bm *= shape_a[i];
    }
    return bm;
}

// --------------------------------------------------------------------------------------------------
// GGUF -> OneDNN-WOQ transcode mapping (compute-bound path, SUMMARY §3.3.2 / SPEC §4.3).
// Each baseline GGUF format is requantised (symmetric, per REQUANT_GROUP) into the smallest OneDNN
// low-bit weight domain that preserves its precision tier: 4-bit families -> i4, 5/6/8-bit -> i8.
// REQUANT_GROUP is fixed at 32 (divides every baseline block_elem: 32 and 256).
// --------------------------------------------------------------------------------------------------
constexpr int GGUF_REQUANT_GROUP = 32;

struct GgufTranscodeTarget {
    bool to_i4;   // true -> i4 weight, false -> i8 weight
    int qmax;     // symmetric quant max (7 for i4, 127 for i8)
};

GgufTranscodeTarget transcode_target(element::Type_t t) {
    switch (t) {
    case element::Type_t::gguf_q4_0:
    case element::Type_t::gguf_q4_k:
        return {true, 7};
    case element::Type_t::gguf_q5_k:
    case element::Type_t::gguf_q6_k:
    case element::Type_t::gguf_q8_0:
        return {false, 127};
    default:
        OPENVINO_THROW("[GPU] FCGGUFOpt: no transcode target for ", element::Type(t).get_type_name());
    }
}

// =================================================================================================
// Memory-bound (decode) GEMV kernel generator — unchanged native path, handles any M.
// =================================================================================================

// Subgroup width for the K-split GEMV: one subgroup (this many lanes) cooperatively computes one
// output, the reduction blocks of a row striped across its lanes. 16 matches the BMG/Xe2 native
// SIMD width and divides every shape's blocks-per-row (K is a multiple of 256 -> >= 16 blocks).
constexpr int GGUF_GEMV_SG_SIZE = 16;

#ifdef ENABLE_ONEDNN_FOR_GPU
// Q6_K int8-activation dp4a decode tuning (see fc_gguf_dp4a.cl). Both the kernel generator and the
// impl read this so the JIT geometry and the runtime dispatch stay in lock-step.
//   - NROW: output rows one subgroup owns (multi-row register blocking; 4 is the measured sweet spot
//           on the DRAM-bound down_proj shape, amortising the per-block activation re-read). The
//           weight is read at its native block stride (no repack / no second copy), so the GEMV's
//           unrolled per-row reads are row-guarded for shapes whose N is not a NROW multiple.
int gguf_q6k_nrow() {
    if (const char* env = std::getenv("OV_GPU_GGUF_Q6K_NROW")) {
        const long v = std::atol(env);
        if (v >= 1) {
            return static_cast<int>(v);
        }
    }
    return 4;
}
#endif  // ENABLE_ONEDNN_FOR_GPU

class FCGGUFOptGenerator : public KernelGenerator {
public:
    FCGGUFOptGenerator() : KernelGenerator("fc_gguf_opt") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        std::string name = get_kernel_name() + "_" + element::Type(in1.data_type).get_type_name() + "_K" +
                           std::to_string(K) + "_N" + std::to_string(N);
        if (params.is_dynamic()) {
            return name + "__sa";  // shape-agnostic (M from shape_info)
        }
        return name + "_M" + std::to_string(derive_bm(params.input_layouts[0].get_shape()));
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto& in0 = params.input_layouts[0];  // activation [BM, K]
        const auto& in1 = params.input_layouts[1];  // gguf weight [N, K] (always static)
        const auto& shape_w = in1.get_shape();

        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add(make_type_jit_constants("OUTPUT", params.output_layouts[0].data_type));

        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const element::Type wt(in1.data_type);

        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("GGUF_BLOCK_ELEM", static_cast<int>(wt.block_elem_count())),
            make_jit_constant("GGUF_BLOCK_BYTES", static_cast<int>(wt.block_byte_size())),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            if (params.is_dynamic()) {
                return;
            }
            const auto& in1 = params.input_layouts[1];
            const size_t N = in1.get_shape()[0];
            const size_t BM = derive_bm(params.input_layouts[0].get_shape());

            // K-split: one subgroup (GGUF_GEMV_SG_SIZE lanes) per output. global[0] = N * SG_SIZE,
            // local[0] = SG_SIZE keeps exactly one subgroup per work-group (max work-groups -> best
            // occupancy, critical for the small-N k/v projections).
            auto& wgs = kd.params.workGroups;
            wgs.global = {N * GGUF_GEMV_SG_SIZE, BM, 1};
            wgs.local = {GGUF_GEMV_SG_SIZE, 1, 1};
        }};
    }
};

#ifdef ENABLE_ONEDNN_FOR_GPU
// =================================================================================================
// Transcode kernel generator (GGUF block -> i4/i8 weight + f16 per-group scale scratchpad).
// Shape-independent of M: keyed by the static weight only. Args are bound explicitly by the impl,
// so get_arguments_desc returns an empty descriptor (filled at dispatch time).
// =================================================================================================
class FCGGUFTranscodeGenerator : public KernelGenerator {
public:
    FCGGUFTranscodeGenerator() : KernelGenerator("fc_gguf_transcode") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        return get_kernel_name() + "_" + element::Type(in1.data_type).get_type_name() + "_K" + std::to_string(K) +
               "_N" + std::to_string(N);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto& in1 = params.input_layouts[1];
        const auto& shape_w = in1.get_shape();
        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const element::Type wt(in1.data_type);
        const auto tgt = transcode_target(in1.data_type);

        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("GGUF_BLOCK_ELEM", static_cast<int>(wt.block_elem_count())),
            make_jit_constant("GGUF_BLOCK_BYTES", static_cast<int>(wt.block_byte_size())),
            make_jit_constant("REQUANT_GROUP", GGUF_REQUANT_GROUP),
            make_jit_constant("TRANSCODE_TO_I4", tgt.to_i4 ? 1 : 0),
            make_jit_constant("QMAX", tgt.qmax),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });
        return jit;
    }

    // Args (raw weight in; packed weight + scale out) are supplied explicitly by the impl at dispatch.
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// =================================================================================================
// Activation int8 pre-quant generator (Q5_K / Q6_K dp4a decode path). Explicit-args, keyed by the
// static activation dtype + K, N; dispatched with the concrete M at execute time (no shape-info arg).
// =================================================================================================
class FCGGUFPrequantGenerator : public KernelGenerator {
public:
    FCGGUFPrequantGenerator() : KernelGenerator("fc_gguf_prequant") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in0 = params.input_layouts[0];
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        return get_kernel_name() + "_" + element::Type(in0.data_type).get_type_name() + "_K" +
               std::to_string(K) + "_N" + std::to_string(N);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto& in0 = params.input_layouts[0];
        const auto& in1 = params.input_layouts[1];
        const size_t K = in1.get_shape()[1];
        jit.add(make_type_jit_constants("INPUT0", in0.data_type));
        jit.add({make_jit_constant("K_SIZE", static_cast<int>(K))});
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};

// =================================================================================================
// Q5_K / Q6_K SWAR dp4a GEMV generator (int8-activation decode path). Explicit-args; output dtype +
// static K, N + block geometry baked in. SG_SIZE matches the K-split GEMV; Q6_K adds NROW multi-row
// register blocking, Q5_K is single-row (NROW=1).
// =================================================================================================
class FCGGUFDp4aGenerator : public KernelGenerator {
public:
    FCGGUFDp4aGenerator() : KernelGenerator("fc_gguf_dp4a") {}

protected:
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in1 = params.input_layouts[1];
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        return get_kernel_name() + "_" + element::Type(in1.data_type).get_type_name() + "_K" +
               std::to_string(K) + "_N" + std::to_string(N);
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = make_base_jit_constants(params);
        const auto& in1 = params.input_layouts[1];
        const auto& shape_w = in1.get_shape();
        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const element::Type wt(in1.data_type);
        const bool is_q6k = (in1.data_type == element::Type_t::gguf_q6_k);
        // Q6_K uses multi-row register blocking (one subgroup owns NROW output rows); Q5_K is
        // single-row (NROW=1). Both read the weight at its native per-block stride (no repack).
        const int nrow = is_q6k ? gguf_q6k_nrow() : 1;
        jit.add(make_type_jit_constants("OUTPUT", params.output_layouts[0].data_type));
        jit.add({
            make_jit_constant("K_SIZE", static_cast<int>(K)),
            make_jit_constant("N_SIZE", static_cast<int>(N)),
            make_jit_constant("GGUF_BLOCK_ELEM", static_cast<int>(wt.block_elem_count())),
            make_jit_constant("GGUF_BLOCK_BYTES", static_cast<int>(wt.block_byte_size())),
            make_jit_constant("SG_SIZE", GGUF_GEMV_SG_SIZE),
            make_jit_constant("NROW", nrow),
            make_jit_constant(gguf_type_jit_flag(in1.data_type), 1),
        });
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams&) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }
};
#endif  // ENABLE_ONEDNN_FOR_GPU

class FCGGUFOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FCGGUFOptImpl)

    // Memory-bound GEMV stage (decode / small M) — always present.
    Stage::Ptr gguf_stage = make_stage<FCGGUFOptGenerator>();
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Compute-bound transcode stage (prefill / large M) — feeds a direct dnnl::matmul.
    Stage::Ptr transcode_stage = make_stage<FCGGUFTranscodeGenerator>();
    // Q5_K / Q6_K int8-activation dp4a decode path (small M): activation prequant + SWAR dp4a GEMV.
    Stage::Ptr prequant_stage = make_stage<FCGGUFPrequantGenerator>();
    Stage::Ptr dp4a_stage = make_stage<FCGGUFDp4aGenerator>();
#endif

    // Activation rows above which the transcode + OneDNN WOQ GEMM path is used (SUMMARY §5,
    // M_MEM_BOUND_THRESHOLD). Overridable via OV_GPU_GGUF_PREFILL_THRESHOLD for tuning.
    size_t m_prefill_threshold = 32;

    // Q5_K dp4a int8-activation decode path: on by default for Q5_K weights, disabled by setting
    // OV_GPU_GGUF_Q5K_DP4A=0. m_q5k_dp4a records whether this node actually instantiated the path.
    bool m_use_q5k_dp4a = true;
    bool m_q5k_dp4a = false;

    // Q6_K dp4a int8-activation decode path: on by default for Q6_K weights, disabled by setting
    // OV_GPU_GGUF_Q6K_DP4A=0. m_q6k_dp4a records whether this node actually instantiated the path.
    // The weight is read at its native block stride (single copy, no repack), with NROW multi-row
    // register blocking amortising the per-block activation re-read.
    bool m_use_q6k_dp4a = true;
    bool m_q6k_dp4a = false;

    FCGGUFOptImpl() : PrimitiveImplOCL(FCGGUFOpt::get_type_info_static()) {
        if (const char* env = std::getenv("OV_GPU_GGUF_PREFILL_THRESHOLD")) {
            const long v = std::atol(env);
            if (v >= 0) {
                m_prefill_threshold = static_cast<size_t>(v);
            }
        }
        if (const char* env = std::getenv("OV_GPU_GGUF_Q5K_DP4A")) {
            m_use_q5k_dp4a = (std::atol(env) != 0);
        }
        if (const char* env = std::getenv("OV_GPU_GGUF_Q6K_DP4A")) {
            m_use_q6k_dp4a = (std::atol(env) != 0);
        }
    }
    FCGGUFOptImpl(const program_node& node, const RuntimeParams& params) : FCGGUFOptImpl() {
        add_stage(gguf_stage, params);
#ifdef ENABLE_ONEDNN_FOR_GPU
        add_stage(transcode_stage, params);
        if (m_use_q5k_dp4a && params.input_layouts[1].data_type == element::Type_t::gguf_q5_k) {
            m_q5k_dp4a = true;
            add_stage(prequant_stage, params);
            add_stage(dp4a_stage, params);
        }
        if (m_use_q6k_dp4a && params.input_layouts[1].data_type == element::Type_t::gguf_q6_k) {
            m_q6k_dp4a = true;
            add_stage(prequant_stage, params);
            add_stage(dp4a_stage, params);
        }
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<FCGGUFOptImpl>(this);
        auto* c = static_cast<FCGGUFOptImpl*>(copy.get());
        c->m_prefill_threshold = m_prefill_threshold;
        c->m_use_q5k_dp4a = m_use_q5k_dp4a;
        c->m_q5k_dp4a = m_q5k_dp4a;
        c->m_use_q6k_dp4a = m_use_q6k_dp4a;
        c->m_q6k_dp4a = m_q6k_dp4a;
        return copy;
    }

    // Bind activation (INPUT0) + weight (INPUT1) + output for the GEMV stage. The empty scale/ZP FC
    // dependencies are intentionally not referenced by that kernel's descriptor.
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);

        data.inputs.push_back(instance.dep_memory_ptr(0));  // activation
        if (fc_inst) {
            data.inputs.push_back(fc_inst->weights_memory());
        } else {
            data.inputs.push_back(instance.dep_memory_ptr(1));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); ++i) {
            data.outputs.push_back(instance.output_memory_ptr(i));
        }
        data.shape_info = instance.shape_info_memory_ptr();
        return data;
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Scratchpad: transcoded low-bit weight [N, K] + f16 per-group scale [K/group, N]. Both sized
    // only by the static K, N (never by M), so they are allocated once and reused across executes.
    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto& in1 = params.get_input_layout(1);
        if (in1.is_dynamic() || !element::is_gguf_block(in1.data_type)) {
            return {};
        }
        const auto& shape_w = in1.get_shape();
        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const size_t num_groups = K / GGUF_REQUANT_GROUP;
        const auto tgt = transcode_target(in1.data_type);

        std::vector<BufferDescriptor> bufs;
        // [0] packed weight: i4 ([N,K] -> bytes_count = N*K/2) or i8 ([N,K]).
        const auto w_dt = tgt.to_i4 ? data_types::i4 : data_types::i8;
        bufs.emplace_back(cldnn::layout(ov::Shape{N, K}, w_dt, cldnn::format::bfyx), /*lockable=*/false);
        // [1] per-group scale: f16 [K/group, N].
        bufs.emplace_back(cldnn::layout(ov::Shape{num_groups, N}, data_types::f16, cldnn::format::bfyx),
                          /*lockable=*/false);
        // [2],[3] Q5_K dp4a scratch: int8 activation [Mmax, K] + per-32 f32 scale [Mmax, K/32], sized
        // for the decode cap Mmax = m_prefill_threshold (dp4a only runs for M <= threshold), so they
        // are M-independent and allocated once.
        if (m_q5k_dp4a || m_q6k_dp4a) {
            const size_t Mmax = std::max<size_t>(m_prefill_threshold, 1);
            bufs.emplace_back(cldnn::layout(ov::Shape{Mmax, K}, data_types::i8, cldnn::format::bfyx),
                              /*lockable=*/false);
            bufs.emplace_back(cldnn::layout(ov::Shape{Mmax, K / GGUF_REQUANT_GROUP}, data_types::f32,
                                            cldnn::format::bfyx),
                              /*lockable=*/false);
        }
        return bufs;
    }
#endif

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events,
                              cldnn::primitive_inst& instance) override {
        // Refresh per-stage need_args_update / need_dispatch_data_update from the current execution
        // flags (SHAPE_CHANGED, ARG_UPDATE_REQUIRED, ...). The base execute() does this; since we
        // override execute() and dispatch the GEMV stage directly, we must do it too. Without it the
        // shape-agnostic GEMV kernel keeps the global_work_size computed for the first (prefill) shape
        // and re-runs decode (M=1) with the prefill row count, writing past the M=1 output buffer
        // (CL_OUT_OF_RESOURCES / out-of-bounds).
        update_rt_params(instance);
#ifdef ENABLE_ONEDNN_FOR_GPU
        const auto& params = *instance.get_impl_params();
        const auto& in0 = params.get_input_layout(0);
        const auto& in1 = params.get_input_layout(1);
        if (!in0.is_dynamic() && !in1.is_dynamic()) {
            const size_t M = derive_bm(in0.get_shape());
            if (M > m_prefill_threshold) {
                return execute_transcode_plus_onednn_woq(events, instance, M);
            }
            if (m_q5k_dp4a || m_q6k_dp4a) {
                return execute_prequant_plus_dp4a(events, instance, M);
            }
        }
#endif
        // Memory-bound / small-M (or OneDNN disabled): run only the GEMV stage.
        return execute_stage(events, instance, gguf_stage);
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
private:
    // Bring the base auto-args execute_stage overloads into scope (the explicit-args overload below
    // would otherwise hide them by name).
    using PrimitiveImplOCL::execute_stage;

    // Explicit-args stage dispatch (the base execute_stage binds args via get_arguments(), which only
    // knows the FC deps). The transcode kernel reads the raw weight and writes the two scratchpads,
    // so its inputs/outputs/gws are supplied directly here (mirrors moe_3gemm_swiglu_opt's helper).
    cldnn::event::ptr execute_stage(const std::vector<cldnn::event::ptr>& events,
                                    cldnn::primitive_inst& instance,
                                    Stage& stage,
                                    std::vector<cldnn::memory::ptr> inputs,
                                    std::vector<cldnn::memory::ptr> outputs,
                                    const std::vector<size_t>& global,
                                    const std::vector<size_t>& local) const {
        cldnn::stream& stream = instance.get_network().get_stream();
        cldnn::kernel_arguments_data args;
        cldnn::kernel_arguments_desc desc;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, i});
            args.inputs.push_back(inputs[i]);
        }
        for (uint32_t i = 0; i < outputs.size(); i++) {
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, i});
            args.outputs.push_back(outputs[i]);
        }
        stream.set_arguments(*stage.kernel, desc, args);
        desc.workGroups.global = global;
        desc.workGroups.local = local;
        kernel_dump_info.add_entry_point(stage.kernel->get_id());
        return stream.enqueue_kernel(*stage.kernel, desc, {}, events, /*needs_completion_event=*/false);
    }

    // Cached direct dnnl::matmul WOQ primitive keyed by (gguf type, M, K, N). K/N are static per node
    // so in practice the cache holds one entry per distinct prefill M.
    struct GgufMatmul {
        dnnl::matmul prim;
        dnnl::matmul::primitive_desc pd;
        dnnl::memory::desc src_md;
        dnnl::memory::desc wei_md;
        dnnl::memory::desc dst_md;
        dnnl::memory::desc scale_md;
    };
    struct MmKey {
        int et;
        int m;
        int k;
        int n;
        bool operator==(const MmKey& o) const {
            return et == o.et && m == o.m && k == o.k && n == o.n;
        }
    };
    struct MmKeyHash {
        size_t operator()(const MmKey& k) const {
            size_t h = std::hash<int>()(k.et);
            h = h * 31 + std::hash<int>()(k.m);
            h = h * 31 + std::hash<int>()(k.k);
            h = h * 31 + std::hash<int>()(k.n);
            return h;
        }
    };
    mutable cldnn::LruCache<MmKey, std::shared_ptr<GgufMatmul>, MmKeyHash> m_matmul_cache{64};

    GgufMatmul& get_matmul(element::Type_t et,
                           int M,
                           int K,
                           int N,
                           dnnl::memory::data_type src_dt,
                           dnnl::memory::data_type dst_dt,
                           dnnl::engine& eng) {
        MmKey key{static_cast<int>(et), M, K, N};
        if (m_matmul_cache.has(key)) {
            return *m_matmul_cache.get(key);
        }
        const auto tgt = transcode_target(et);
        const auto w_dt = tgt.to_i4 ? dnnl::memory::data_type::s4 : dnnl::memory::data_type::s8;

        auto k = std::make_shared<GgufMatmul>();
        // src/dst dtype must match the FC activation/output layouts exactly (binding an f16 output
        // buffer as f32, or vice-versa, mis-reads every element). The WOQ decompression still runs in
        // f16 via fpmath_mode below.
        k->src_md = dnnl::memory::desc({M, K}, src_dt, dnnl::memory::format_tag::ab);
        k->dst_md = dnnl::memory::desc({M, N}, dst_dt, dnnl::memory::format_tag::ab);
        // Fixed weight layout [K, N] as `ba` -> physical [N, K] (matches the transcode kernel's write).
        k->wei_md = dnnl::memory::desc({K, N}, w_dt, dnnl::memory::format_tag::ba);
        // Per-K-group x per-N f16 scale.
        k->scale_md = dnnl::memory::desc({K / GGUF_REQUANT_GROUP, N},
                                         dnnl::memory::data_type::f16,
                                         dnnl::memory::format_tag::ab);

        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {GGUF_REQUANT_GROUP, 1}, dnnl::memory::data_type::f16);

        k->pd = dnnl::matmul::primitive_desc(eng, k->src_md, k->wei_md, k->dst_md, attr);
        // Bind the scratchpad through the pd's actual descriptors (dnnl may choose an internal weight
        // layout that differs from the requested ba tag); using the requested md can mis-read the bytes.
        k->wei_md = k->pd.weights_desc();
        k->src_md = k->pd.src_desc();
        k->dst_md = k->pd.dst_desc();
        k->prim = dnnl::matmul(k->pd);
        m_matmul_cache.add(key, k);
        return *m_matmul_cache.get(key);
    }

    cldnn::event::ptr execute_transcode_plus_onednn_woq(const std::vector<cldnn::event::ptr>& events,
                                                        cldnn::primitive_inst& instance,
                                                        size_t M) {
        const auto& params = *instance.get_impl_params();
        const auto& in1 = params.get_input_layout(1);
        const auto& shape_w = in1.get_shape();
        const int N = static_cast<int>(shape_w[0]);
        const int K = static_cast<int>(shape_w[1]);
        const auto et = static_cast<element::Type_t>(in1.data_type);
        const int block_elem = static_cast<int>(element::Type(et).block_elem_count());
        const int blocks_per_row = K / block_elem;

        auto& stream = instance.get_network().get_stream();
        auto* fc_inst = dynamic_cast<cldnn::fully_connected_inst*>(&instance);

        const auto& intermediates = instance.get_intermediates_memories();
        OPENVINO_ASSERT(intermediates.size() >= 2,
                        "[GPU] FCGGUFOpt: transcode scratchpad not allocated (expected >= 2 internal buffers).");
        auto w_scratch = intermediates[0];  // packed i4/i8 weight [N, K]
        auto s_scratch = intermediates[1];  // f16 scale [K/group, N]

        auto weight_mem = fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1);

        // Stage 1: transcode raw GGUF blocks -> {packed low-bit weight, f16 per-group scale}.
        //   One work-item per (n, GGUF block): the block is decoded once and all REQUANT groups inside
        //   it are requantized together, removing the (block_elem / REQUANT_GROUP)x redundant decode of
        //   the old per-group decomposition. N is the subgroup lane axis (local = SG), padded up to a
        //   full subgroup; the kernel guards the padded tail with n >= N_SIZE.
        const size_t n_global =
            ((static_cast<size_t>(N) + GGUF_GEMV_SG_SIZE - 1) / GGUF_GEMV_SG_SIZE) * GGUF_GEMV_SG_SIZE;
        auto transcode_ev = execute_stage(events,
                                          instance,
                                          *transcode_stage,
                                          /*inputs=*/{weight_mem},
                                          /*outputs=*/{w_scratch, s_scratch},
                                          /*global=*/{n_global, static_cast<size_t>(blocks_per_row), 1},
                                          /*local=*/{GGUF_GEMV_SG_SIZE, 1, 1});

        // Stage 2: direct dnnl::matmul WOQ consuming the scratchpad. The OneDNN stream shares the same
        // in-order OCL queue, so submission order serialises it after the transcode kernel; pass the
        // transcode event as the dependency for the (later) returned event ordering.
        auto& dnn_stream = stream.get_onednn_stream();
        auto& dnn_engine = instance.get_network().get_engine().get_onednn_engine();
        const auto src_dt = onednn::convert_data_type(params.get_input_layout(0).data_type);
        const auto dst_dt = onednn::convert_data_type(params.get_output_layout(0).data_type);
        auto& mm = get_matmul(et, static_cast<int>(M), K, N, src_dt, dst_dt, dnn_engine);

        auto src = instance.dep_memory_ptr(0)->get_onednn_memory(mm.src_md);
        auto dst = instance.output_memory_ptr(0)->get_onednn_memory(mm.dst_md);
        auto wei = w_scratch->get_onednn_memory(mm.wei_md);
        auto scale = s_scratch->get_onednn_memory(mm.scale_md);

        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, src},
            {DNNL_ARG_WEIGHTS, wei},
            {DNNL_ARG_DST, dst},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale},
        };
        mm.prim.execute(dnn_stream, args);

        // The matmul (last op) returns no cldnn event; on the shared in-order queue, downstream
        // consumers observe its output. Return the transcode event for dependency tracking; if the
        // output is a network output, force completion so host reads see the matmul result.
        if (instance.needs_completion_event()) {
            stream.finish();
        }
        return transcode_ev;
    }

    // Q5_K / Q6_K decode: one-shot int8 activation prequant -> SWAR dp4a GEMV. Both stages are
    // dispatched with the concrete M (decode M is tiny), so no shape-info arg is needed. The int8
    // activation + per-32 scale scratch are the last two internal buffers. The weight is read at its
    // native block stride (single copy, no repack). Q6_K owns NROW output rows per subgroup, so the
    // GEMV grid spans ceil(N / NROW) subgroups (NROW=1 for Q5_K -> one subgroup per row, as before);
    // the kernel row-guards its unrolled per-row reads so a non-NROW-multiple N stays in-bounds.
    cldnn::event::ptr execute_prequant_plus_dp4a(const std::vector<cldnn::event::ptr>& events,
                                                 cldnn::primitive_inst& instance,
                                                 size_t M) {
        const auto& params = *instance.get_impl_params();
        const auto& in1 = params.get_input_layout(1);
        const auto& shape_w = in1.get_shape();
        const size_t N = shape_w[0];
        const size_t K = shape_w[1];
        const size_t nrow =
            (in1.data_type == element::Type_t::gguf_q6_k) ? static_cast<size_t>(gguf_q6k_nrow()) : 1;
        const size_t row_groups = (N + nrow - 1) / nrow;

        auto* fc_inst = dynamic_cast<cldnn::fully_connected_inst*>(&instance);
        const auto& intermediates = instance.get_intermediates_memories();
        OPENVINO_ASSERT(intermediates.size() >= 2,
                        "[GPU] FCGGUFOpt: dp4a scratch not allocated (expected >= 2 internal buffers).");
        auto aq_scratch = intermediates[intermediates.size() - 2];   // int8 activation [Mmax, K]
        auto asc_scratch = intermediates[intermediates.size() - 1];  // f32 per-32 scale [Mmax, K/32]

        auto act = instance.dep_memory_ptr(0);
        auto weight = fc_inst ? fc_inst->weights_memory() : instance.dep_memory_ptr(1);
        auto out = instance.output_memory_ptr(0);

        // Stage 1: prequant. One work-item per (group, row): global = [K/32, M, 1].
        auto pq_ev = execute_stage(events,
                                   instance,
                                   *prequant_stage,
                                   /*inputs=*/{act},
                                   /*outputs=*/{aq_scratch, asc_scratch},
                                   /*global=*/{K / GGUF_REQUANT_GROUP, M, 1},
                                   /*local=*/{1, 1, 1});

        // Stage 2: SWAR dp4a GEMV. One subgroup owns NROW output rows: global = [ceil(N/NROW)*SG, M, 1].
        return execute_stage({pq_ev},
                             instance,
                             *dp4a_stage,
                             /*inputs=*/{aq_scratch, asc_scratch, weight},
                             /*outputs=*/{out},
                             /*global=*/{row_groups * GGUF_GEMV_SG_SIZE, M, 1},
                             /*local=*/{GGUF_GEMV_SG_SIZE, 1, 1});
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
};

}  // namespace

std::unique_ptr<primitive_impl> FCGGUFOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FCGGUFOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCGGUFOptImpl)
