// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "gather_matmul_gen_micro.hpp"
#include "gathermatmul_sort_gen.hpp"
#include "gathermatmul_gather_gen.hpp"
#include "gathermatmul_batched_gemm_gen.hpp"
#include "gathermatmul_scatter_gen.hpp"
// clang-format on

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "gather_matmul_impl.hpp"
#include "gather_matmul_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <algorithm>
#    include <cstdlib>
#    include <memory>
#    include <oneapi/dnnl/dnnl.hpp>
#    include <unordered_map>

#    include "intel_gpu/primitives/swiglu.hpp"
#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

namespace ov::intel_gpu::ocl {
namespace {
#ifdef ENABLE_ONEDNN_FOR_GPU
inline bool is_prefill_stage(const RuntimeParams& params) {
    const auto& input_shape = params.input_layouts[0].get_partial_shape();
    const auto n_tokens = input_shape[1];
    if (n_tokens.is_dynamic())
        return false;
    return n_tokens.get_length() > 1;
}

static constexpr int32_t BATCHED_PREFILL_THRESHOLD = 16;

inline bool use_batched_prefill(const GatherMatmulRuntimeParams* rtp) {
    return rtp->n_tokens > BATCHED_PREFILL_THRESHOLD;
}

inline bool has_fused_swiglu(const kernel_impl_params& params) {
    for (const auto& fd : params.fused_desc) {
        if (fd.is_type<swiglu>())
            return true;
    }
    return false;
}

// Local to avoid an ocl_v2 → onednn link dependency.
inline dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
    case cldnn::data_types::f32:
        return dnnl::memory::data_type::f32;
    case cldnn::data_types::f16:
        return dnnl::memory::data_type::f16;
    case cldnn::data_types::i8:
        return dnnl::memory::data_type::s8;
    case cldnn::data_types::u8:
        return dnnl::memory::data_type::u8;
    case cldnn::data_types::i32:
        return dnnl::memory::data_type::s32;
    case cldnn::data_types::i4:
        return dnnl::memory::data_type::s4;
    case cldnn::data_types::u4:
        return dnnl::memory::data_type::u4;
    default:
        throw std::invalid_argument("[GPU] gather_matmul: unsupported cldnn->onednn type conversion");
    }
}

// OCL micro + batched_gemm kernels are u4/i4-only; everything else must take the onednn-grouped path.
inline bool weights_need_onednn_grouped(const kernel_impl_params& params) {
    const auto& weights_layout = params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT];
    const auto dt = weights_layout.data_type;
    return !(dt == cldnn::data_types::u4 || dt == cldnn::data_types::i4);
}
#endif

enum GatherMatmulInternalBufferIdx {
    GATHERED_A = 0,        // f16, n_tokens * top_k * K
    GROUP_EXPERT_IDS = 1,  // i32, n_all_experts * top_k
    GROUP_SLOT_IDS = 2,    // i32, n_all_experts * top_k
    GROUP_OFFSETS = 3,     // i32, n_all_experts * top_k
    GROUP_SIZES = 4,       // i32, n_all_experts * top_k
    TOKEN_MAP = 5,         // i32, n_tokens * top_k
    NUM_GROUPS = 6,        // i32, 1
    // Cumulative per-expert end-offsets consumed by dnnl::memory::desc::grouped().
    EXPERT_OFFSETS = 7,  // i32, n_all_experts
    // Conditionally allocated: dominates internal memory at typical MoE shapes.
    PACKED_OUT = 8,  // f16, n_tokens * top_k * N
};

class GatherMatmulOCLImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherMatmulOCLImpl)
#ifdef ENABLE_ONEDNN_FOR_GPU
    static constexpr bool prefill = true;

    // Per-token stages
    Stage::Ptr regular_micro_single_token = make_stage<GatherMatmulMicroGenerator>(!prefill);
    Stage::Ptr regular_micro_multi_tokens = make_stage<GatherMatmulMicroGenerator>(prefill);

    // Batched OCL prefill stages
    Stage::Ptr batched_sort = make_stage<GatherMatmulSortGenerator>();
    Stage::Ptr batched_gather = make_stage<GatherMatmulGatherGenerator>();
    Stage::Ptr batched_gemm = make_stage<GatherMatmulBatchedGemmGenerator>();

    // Onednn-grouped prefill stages (sort packs expert-major; scatter unpacks).
    Stage::Ptr onednn_sort = make_stage<GatherMatmulOnednnSortGenerator>();
    Stage::Ptr onednn_scatter = make_stage<GatherMatmulScatterGenerator>();
#endif

    explicit GatherMatmulOCLImpl() : PrimitiveImplOCL(GatherMatmulImpl::get_type_info_static()) {
#ifdef ENABLE_ONEDNN_FOR_GPU
        // GATHER_MATMUL_USE_ONEDNN_PREFILL gates the dnnl-grouped large-prefill path (default off).
        _use_onednn_grouped = std::getenv("GATHER_MATMUL_USE_ONEDNN_PREFILL") != nullptr;
#endif
    }
    explicit GatherMatmulOCLImpl(const RuntimeParams& impl_param) : GatherMatmulOCLImpl() {
        auto params = impl_param;
        GPU_DEBUG_TRACE_DETAIL << "GatherMatmul create stages for dynamic = " << params.is_dynamic() << "\n";

#ifdef ENABLE_ONEDNN_FOR_GPU
        // Non-u4/i4 weights have no functional OCL path; force onednn-grouped.
        const bool need_onednn_for_weights = weights_need_onednn_grouped(params);
        if (need_onednn_for_weights) {
            _use_onednn_grouped = true;
        }

        GPU_DEBUG_TRACE_DETAIL << "GatherMatmul :: use_onednn_grouped=" << _use_onednn_grouped << " need_onednn_for_weights=" << need_onednn_for_weights
                               << std::endl;

        add_stage(regular_micro_multi_tokens, params);
        add_stage(regular_micro_single_token, params);

        // Always registered: serve as fallback when onednn declines (e.g. fused swiglu).
        add_stage(batched_sort, params);
        add_stage(batched_gather, params);
        add_stage(batched_gemm, params);

        if (_use_onednn_grouped) {
            add_stage(onednn_sort, params);
            add_stage(onednn_scatter, params);
        }
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatherMatmulOCLImpl>(this);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        std::vector<BufferDescriptor> descs;

        const auto& input_shape = params.input_layouts[gather_matmul::BGMInputIdx::INPUT].get_shape();
        const auto& weight_shape = params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT].get_shape();
        const auto& indices_shape = params.input_layouts[gather_matmul::BGMInputIdx::INDICES].get_shape();

        size_t n_tokens = input_shape[1];
        size_t top_k = indices_shape[1];
        size_t k = weight_shape.size() == 4 ? weight_shape[2] * weight_shape[3] : weight_shape[2];
        size_t n_all_experts = weight_shape[0];
        size_t max_groups = n_all_experts * top_k;

        // 0: gathered_A — f16, n_tokens * top_k * K
        descs.emplace_back(n_tokens * top_k * k, ov::element::f16);
        // 1: group_expert_ids — i32, max_groups
        descs.emplace_back(max_groups, ov::element::i32);
        // 2: group_slot_ids — i32, max_groups
        descs.emplace_back(max_groups, ov::element::i32);
        // 3: group_offsets — i32, max_groups
        descs.emplace_back(max_groups, ov::element::i32);
        // 4: group_sizes — i32, max_groups
        descs.emplace_back(max_groups, ov::element::i32);
        // 5: token_map — i32, n_tokens * top_k
        // Also used as scratch during sort (needs max_groups entries), ensure it's large enough
        descs.emplace_back(std::max(n_tokens * top_k, max_groups), ov::element::i32);
        // 6: num_groups — i32, 1
        descs.emplace_back(static_cast<size_t>(1), ov::element::i32);
        // 7: expert_offsets — cumulative exclusive ends per expert.
        descs.emplace_back(n_all_experts, ov::element::i32);

#ifdef ENABLE_ONEDNN_FOR_GPU
        // OR: env-var doesn't survive clone(), but execute() dispatches on dtype directly.
        if (_use_onednn_grouped || weights_need_onednn_grouped(params)) {
            // 8: packed dnnl::matmul output [total_tokens, N], unfused N.
            size_t n_val = weight_shape[1];
            descs.emplace_back(n_tokens * top_k * n_val, ov::element::f16);
        }
#endif

        return descs;
    }

    void update_rt_params(const primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<GatherMatmulRuntimeParams>();
        }
        update_stages_flags(instance);
        auto rtp = static_cast<GatherMatmulRuntimeParams*>(m_rt_params.get());
        const auto& input_shape = instance.get_input_layout(gather_matmul::BGMInputIdx::INPUT).get_shape();
        const auto& indices_shape = instance.get_input_layout(gather_matmul::BGMInputIdx::INDICES).get_shape();
        rtp->n_activated_experts = static_cast<int32_t>(input_shape[0]);
        rtp->n_tokens = static_cast<int32_t>(input_shape[1]);
        rtp->top_k = static_cast<int32_t>(indices_shape[1]);
        GPU_DEBUG_TRACE_DETAIL << "GatherMatmul :: n_activated_experts=" << rtp->n_activated_experts << " n_tokens=" << rtp->n_tokens << " top_k=" << rtp->top_k
                               << "\n";
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        PrimitiveImplOCL::update(inst, impl_params);
        inst.update_shape_info_tensor(impl_params);
        update_rt_params(inst);
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
#ifdef ENABLE_ONEDNN_FOR_GPU
        const auto& params = *instance.get_impl_params();
        bool is_prefill = is_prefill_stage(params);
        update_rt_params(instance);
        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(m_rt_params.get());

        // Non-u4/i4 weights: dnnl is the only correctness-preserving path; route everything here.
        if (weights_need_onednn_grouped(params) && !has_fused_swiglu(params) && has_stage(onednn_sort) && has_stage(onednn_scatter) &&
            has_stage(batched_gather)) {
            GPU_DEBUG_TRACE_DETAIL << "GatherMatmul Execute onednn-grouped (non-u4/i4 weights, n_tokens=" << rtp->n_tokens << ")" << std::endl;
            return execute_onednn_grouped(events, instance);
        }

        if (is_prefill) {
            // Onednn-grouped large-prefill for u4/i4 — fused-swiglu still falls back to OCL batched_gemm.
            if (_use_onednn_grouped && rtp->n_tokens > BATCHED_PREFILL_THRESHOLD && !has_fused_swiglu(params) && has_stage(onednn_sort) &&
                has_stage(onednn_scatter) && has_stage(batched_gather)) {
                GPU_DEBUG_TRACE_DETAIL << "GatherMatmul Execute onednn-grouped prefill (n_tokens=" << rtp->n_tokens << ")" << std::endl;
                return execute_onednn_grouped(events, instance);
            }

            if (use_batched_prefill(rtp) && has_stage(batched_sort) && has_stage(batched_gather) && has_stage(batched_gemm)) {
                GPU_DEBUG_TRACE_DETAIL << "GatherMatmul Execute batched prefill pipeline (n_tokens=" << rtp->n_tokens << ")" << std::endl;
                auto sort_event = execute_stage(events, instance, batched_sort);
                auto gather_event = execute_stage({sort_event}, instance, batched_gather);
                return execute_stage({gather_event}, instance, batched_gemm);
            } else if (has_stage(regular_micro_multi_tokens)) {
                GPU_DEBUG_TRACE_DETAIL << "GatherMatmul Execute prefill micro_multi_tokens stage (n_tokens=" << rtp->n_tokens << ")" << std::endl;
                return execute_stage(events, instance, regular_micro_multi_tokens);
            } else {
                OPENVINO_THROW("GatherMatmul Prefill stage is not available");
            }
        } else {
            return execute_stage(events, instance, regular_micro_single_token);
        }
#else
        OPENVINO_THROW("gather_matmul is only supported on systolic platforms.");
#endif
        return nullptr;
    }

private:
#ifdef ENABLE_ONEDNN_FOR_GPU
    // One entry per total_tokens; carries the wrappers we need at execute() time.
    struct OnednnGroupedKernel {
        dnnl::matmul::primitive_desc pd;
        dnnl::matmul prim;
        dnnl::memory::desc scale_md;  // valid iff is_quantized
        dnnl::memory::desc zp_md;     // valid iff is_quantized && has_zp
        dnnl::memory::desc bias_md;   // valid iff has_bias
        bool is_quantized = false;
        bool has_zp = false;
        bool has_bias = false;
        int32_t weight_group_size = -1;  // -1 ⇒ per-OC
    };

    bool _use_onednn_grouped = false;
    cldnn::LruCache<int, std::shared_ptr<OnednnGroupedKernel>> _grouped_kernels{32};

    OnednnGroupedKernel& get_grouped_kernel(int total_tokens, primitive_inst& instance) {
        if (_grouped_kernels.has(total_tokens)) {
            return *_grouped_kernels.get(total_tokens);
        }

        const auto& impl_params = *instance.get_impl_params();
        auto desc = impl_params.typed_desc<gather_matmul>();
        auto& engine = instance.get_network().get_engine();
        auto& dnnl_engine = engine.get_onednn_engine();

        const auto& input_layout = impl_params.input_layouts[gather_matmul::BGMInputIdx::INPUT];
        const auto& weights_layout = impl_params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT];
        const auto& output_layout = impl_params.output_layouts[0];
        const auto& wshape = weights_layout.get_shape();

        auto a_dt = convert_data_type(input_layout.data_type);
        auto out_dt = convert_data_type(output_layout.data_type);
        auto w_dt = convert_data_type(weights_layout.data_type);

        const dnnl::memory::dim num_experts = static_cast<dnnl::memory::dim>(wshape[0]);
        const dnnl::memory::dim N = static_cast<dnnl::memory::dim>(wshape[1]);
        const dnnl::memory::dim K = wshape.size() == 4 ? static_cast<dnnl::memory::dim>(wshape[2] * wshape[3]) : static_cast<dnnl::memory::dim>(wshape[2]);

        static const std::vector<cldnn::data_types> quantized_types = {cldnn::data_types::u4,
                                                                       cldnn::data_types::i4,
                                                                       cldnn::data_types::u8,
                                                                       cldnn::data_types::i8};
        bool is_quantized = std::any_of(quantized_types.begin(), quantized_types.end(), [&](cldnn::data_types t) {
            return t == weights_layout.data_type;
        });

        int32_t weight_group_size = -1;
        if (is_quantized) {
            const auto& scale_layout = impl_params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT_SCALE];
            const auto& sshape = scale_layout.get_shape();
            // Scales: [E, N, G] (rank-3) or [E, N] (rank-2).
            size_t num_groups = (sshape.size() >= 3) ? sshape[2] : 1;
            OPENVINO_ASSERT(num_groups > 0 && (static_cast<size_t>(K) % num_groups) == 0,
                            "[GPU] gather_matmul: scale num_groups=",
                            num_groups,
                            " does not divide K=",
                            K);
            weight_group_size = (num_groups == 1) ? -1 : static_cast<int32_t>(K / num_groups);
            constexpr int32_t dnnl_decompression_group_alignment = 32;
            OPENVINO_ASSERT(weight_group_size <= 0 || (weight_group_size % dnnl_decompression_group_alignment) == 0,
                            "[GPU] gather_matmul: weight_group_size must be a multiple of ",
                            dnnl_decompression_group_alignment,
                            ", got ",
                            weight_group_size);
        }

        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

        if (is_quantized) {
            const bool has_groups = (weight_group_size > 0);
            if (has_groups) {
                // per-expert(0) × per-K-group(1) × per-N-channel(2)
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {weight_group_size, 1}, dnnl::memory::data_type::f16);
                if (desc->has_zp) {
                    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1) | (1 << 2), {weight_group_size, 1}, w_dt);
                }
            } else {
                // per-expert(0) × per-N-channel(2), no K-grouping
                attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, dnnl::memory::data_type::f16);
                if (desc->has_zp) {
                    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2), {}, w_dt);
                }
            }
        }

        // Grouped src/dst: rows grouped by expert along axis-0 (s32 end-offsets).
        auto src_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, K}, a_dt, 0, num_experts, dnnl::memory::data_type::s32);
        auto dst_md = dnnl::memory::desc::grouped(dnnl::memory::dims{total_tokens, N}, out_dt, 0, num_experts, dnnl::memory::data_type::s32);
        // Logical [E, K, N], physical layout acb ⇒ stored as [E, N, K].
        auto w_md = dnnl::memory::desc(dnnl::memory::dims{num_experts, K, N}, w_dt, dnnl::memory::format_tag::acb);

        auto gk = std::make_shared<OnednnGroupedKernel>();
        gk->is_quantized = is_quantized;
        gk->has_zp = is_quantized && desc->has_zp;
        gk->has_bias = desc->has_bias;
        gk->weight_group_size = weight_group_size;

        if (desc->has_bias) {
            const auto& bias_layout = impl_params.input_layouts[gather_matmul::BGMInputIdx::BIAS];
            auto b_dt = convert_data_type(bias_layout.data_type);
            // Logical [E, 1, N] flattens to dnnl {E, N}.
            gk->bias_md = dnnl::memory::desc(dnnl::memory::dims{num_experts, N}, b_dt, dnnl::memory::format_tag::ab);
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, gk->bias_md, dst_md, attr);
        } else {
            gk->pd = dnnl::matmul::primitive_desc(dnnl_engine, src_md, w_md, dst_md, attr);
        }

        if (is_quantized) {
            const auto& scale_layout = impl_params.input_layouts[gather_matmul::BGMInputIdx::WEIGHT_SCALE];
            auto s_dt = convert_data_type(scale_layout.data_type);
            const bool has_groups = (weight_group_size > 0);
            if (has_groups) {
                const dnnl::memory::dim num_k_groups = K / weight_group_size;
                gk->scale_md = dnnl::memory::desc({num_experts, num_k_groups, N}, s_dt, dnnl::memory::format_tag::abc);
                if (gk->has_zp) {
                    gk->zp_md = dnnl::memory::desc({num_experts, num_k_groups, N}, w_dt, dnnl::memory::format_tag::abc);
                }
            } else {
                gk->scale_md = dnnl::memory::desc({num_experts, N}, s_dt, dnnl::memory::format_tag::ab);
                if (gk->has_zp) {
                    gk->zp_md = dnnl::memory::desc({num_experts, N}, w_dt, dnnl::memory::format_tag::ab);
                }
            }
        }

        gk->prim = dnnl::matmul(gk->pd);

        _grouped_kernels.add(total_tokens, gk);
        return *_grouped_kernels.get(total_tokens);
    }

    // Pipeline: onednn_sort → batched_gather → dnnl::matmul → onednn_scatter (in-order queue serializes).
    event::ptr execute_onednn_grouped(const std::vector<event::ptr>& events, primitive_inst& instance) {
        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(m_rt_params.get());
        const int total_tokens = rtp->n_tokens * rtp->top_k;

        auto sort_event = execute_stage(events, instance, onednn_sort);
        auto gather_event = execute_stage({sort_event}, instance, batched_gather);

        auto& net = instance.get_network();
        auto& stream = net.get_stream();
        auto& dnn_stream = stream.get_onednn_stream();

        auto& gk = get_grouped_kernel(total_tokens, instance);
        const auto& impl_params = *instance.get_impl_params();
        auto desc = impl_params.typed_desc<gather_matmul>();

        const auto& intermediates = instance.get_intermediates_memories();
        OPENVINO_ASSERT(intermediates.size() > PACKED_OUT,
                        "[GPU] gather_matmul: onednn-grouped path requires internal buffer ",
                        PACKED_OUT,
                        " but only ",
                        intermediates.size(),
                        " buffers are allocated");

        auto& gathered_a_mem = *intermediates[GATHERED_A];
        auto& packed_out_mem = *intermediates[PACKED_OUT];
        auto& expert_offsets_mem = *intermediates[EXPERT_OFFSETS];

        auto src_dnn = gathered_a_mem.get_onednn_grouped_memory(gk.pd.src_desc(), expert_offsets_mem);
        auto dst_dnn = packed_out_mem.get_onednn_grouped_memory(gk.pd.dst_desc(), expert_offsets_mem);
        auto& weights_mem = *instance.input_memory_ptr(gather_matmul::BGMInputIdx::WEIGHT);
        auto w_dnn = weights_mem.get_onednn_memory(gk.pd.weights_desc());

        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, src_dnn},
            {DNNL_ARG_WEIGHTS, w_dnn},
            {DNNL_ARG_DST, dst_dnn},
        };

        if (gk.is_quantized) {
            auto& scale_mem = *instance.input_memory_ptr(gather_matmul::BGMInputIdx::WEIGHT_SCALE);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem.get_onednn_memory(gk.scale_md)});
            if (gk.has_zp) {
                auto& zp_mem = *instance.input_memory_ptr(gather_matmul::BGMInputIdx::WEIGHT_ZP);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_mem.get_onednn_memory(gk.zp_md)});
            }
        }

        if (gk.has_bias) {
            auto& bias_mem = *instance.input_memory_ptr(gather_matmul::BGMInputIdx::BIAS);
            // Use the saved bias_md: pd.weights_desc() no-arg form shadows the base bias variant.
            args.insert({DNNL_ARG_BIAS, bias_mem.get_onednn_memory(gk.bias_md)});
        }

        gk.prim.execute(dnn_stream, args);

        return execute_stage({gather_event}, instance, onednn_scatter);
    }
#endif
};
}  // namespace

std::unique_ptr<primitive_impl> GatherMatmulImpl::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gather_matmul>());
    return std::make_unique<GatherMatmulOCLImpl>(params);
}
}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_matmul)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherMatmulOCLImpl)
