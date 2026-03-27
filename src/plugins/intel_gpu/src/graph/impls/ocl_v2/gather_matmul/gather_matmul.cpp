// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "gather_matmul_gen_micro.hpp"
#include "gathermatmul_sort_gen.hpp"
#include "gathermatmul_gather_gen.hpp"
#include "gathermatmul_batched_gemm_gen.hpp"
// clang-format on

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "gather_matmul_impl.hpp"
#include "gather_matmul_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"

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

// Threshold: use batched path only when n_tokens exceeds this value
static constexpr int32_t BATCHED_PREFILL_THRESHOLD = 16;

inline bool use_batched_prefill(const GatherMatmulRuntimeParams* rtp) {
    return rtp->n_tokens > BATCHED_PREFILL_THRESHOLD;
}
#endif

// Internal buffer indices for the batched prefill pipeline
enum GatherMatmulInternalBufferIdx {
    GATHERED_A = 0,        // f16, n_tokens * top_k * K
    GROUP_EXPERT_IDS = 1,  // i32, n_all_experts * top_k
    GROUP_SLOT_IDS = 2,    // i32, n_all_experts * top_k
    GROUP_OFFSETS = 3,     // i32, n_all_experts * top_k
    GROUP_SIZES = 4,       // i32, n_all_experts * top_k
    TOKEN_MAP = 5,         // i32, n_tokens * top_k
    NUM_GROUPS = 6,        // i32, 1
};

class GatherMatmulOCLImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherMatmulOCLImpl)
#ifdef ENABLE_ONEDNN_FOR_GPU
    static constexpr bool prefill = true;

    // Existing per-token stages
    Stage::Ptr regular_micro_single_token = make_stage<GatherMatmulMicroGenerator>(!prefill);
    Stage::Ptr regular_micro_multi_tokens = make_stage<GatherMatmulMicroGenerator>(prefill);

    // Batched prefill stages
    Stage::Ptr batched_sort = make_stage<GatherMatmulSortGenerator>();
    Stage::Ptr batched_gather = make_stage<GatherMatmulGatherGenerator>();
    Stage::Ptr batched_gemm = make_stage<GatherMatmulBatchedGemmGenerator>();
#endif

    explicit GatherMatmulOCLImpl() : PrimitiveImplOCL(GatherMatmulImpl::get_type_info_static()) {}
    explicit GatherMatmulOCLImpl(const RuntimeParams& impl_param) : GatherMatmulOCLImpl() {
        auto params = impl_param;
        GPU_DEBUG_TRACE_DETAIL << "GatherMatmul create stages for dynamic = " << params.is_dynamic() << "\n";

#ifdef ENABLE_ONEDNN_FOR_GPU
        // Per-token stages (generate + small prefill fallback)
        add_stage(regular_micro_multi_tokens, params);
        add_stage(regular_micro_single_token, params);

        // Batched prefill stages
        add_stage(batched_sort, params);
        add_stage(batched_gather, params);
        add_stage(batched_gemm, params);
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
        rtp->n_activated_experts = input_shape[0];
        rtp->n_tokens = input_shape[1];
        rtp->top_k = indices_shape[1];
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

        if (is_prefill) {
            auto* rtp = static_cast<GatherMatmulRuntimeParams*>(m_rt_params.get());

            if (use_batched_prefill(rtp) && has_stage(batched_sort) && has_stage(batched_gather) && has_stage(batched_gemm)) {
                GPU_DEBUG_TRACE_DETAIL << "GatherMatmul Execute batched prefill pipeline (n_tokens=" << rtp->n_tokens << ")" << std::endl;

                // Stage 1: Sort tokens by expert within each slot
                auto sort_event = execute_stage(events, instance, batched_sort);

                // Stage 2: Gather activations into contiguous buffer
                auto gather_event = execute_stage({sort_event}, instance, batched_gather);

                // Stage 3: Batched GEMM with scattered output
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
};
}  // namespace

std::unique_ptr<primitive_impl> GatherMatmulImpl::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gather_matmul>());
    return std::make_unique<GatherMatmulOCLImpl>(params);
}
}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_matmul)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherMatmulOCLImpl)
