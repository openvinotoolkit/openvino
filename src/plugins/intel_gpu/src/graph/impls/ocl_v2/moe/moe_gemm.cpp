// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "moe_gemm_gen_micro.hpp"
// clang-format on

#include "moe_gemm.hpp"

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "moe_gemm_base.hpp"
#include "moe_gemm_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"

namespace ov::intel_gpu::ocl {
namespace {
#ifdef ENABLE_ONEDNN_FOR_GPU
inline bool is_prefill_stage(const RuntimeParams& params) {
    const auto target_seq_len = params.input_layouts[0].get_partial_shape()[0];
    const auto num_offsets = params.input_layouts[3].get_partial_shape()[0];
    if (num_offsets.is_dynamic())
        return false;
    if (target_seq_len.is_dynamic())
        return false;
    return (target_seq_len.get_length() / num_offsets.get_length()) > 1;
}
#endif

class MoEGemmImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MoEGemmImpl)
#ifdef ENABLE_ONEDNN_FOR_GPU
    static constexpr bool prefill = true;

    Stage::Ptr regular_micro_single_token = make_stage<MoEGemmMicroGenerator>(!prefill);
    Stage::Ptr regular_micro_multi_tokens = make_stage<MoEGemmMicroGenerator>(prefill);
#endif

    explicit MoEGemmImpl() : PrimitiveImplOCL(MoEGemm::get_type_info_static()) {}
    explicit MoEGemmImpl(const RuntimeParams& impl_param) : MoEGemmImpl() {
        auto params = impl_param;
        GPU_DEBUG_TRACE_DETAIL << "create stages for dynamic = " << params.is_dynamic() << "\n";

#ifdef ENABLE_ONEDNN_FOR_GPU
        add_stage(regular_micro_multi_tokens, params);
        add_stage(regular_micro_single_token, params);
#endif
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<MoEGemmImpl>(this);
    }

    void update_rt_params(const primitive_inst& instance) override {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<MoEGemmRuntimeParams>();
        }
        update_stages_flags(instance);
        auto rtp = static_cast<MoEGemmRuntimeParams*>(m_rt_params.get());
        rtp->num_actually_used_experts = instance.get_input_layout(moe_gemm::MoEGemmInputIdx::EXPERTS_IDS).get_shape()[0];
        GPU_DEBUG_TRACE_DETAIL << "moe_gemm :: num_actually_used_experts = " << rtp->num_actually_used_experts << "\n";
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
            if (has_stage(regular_micro_multi_tokens)) {
                GPU_DEBUG_TRACE_DETAIL << "Execute prefill micro_multi_tokens stage" << std::endl;
                return execute_stage(events, instance, regular_micro_multi_tokens);
            } else {
                OPENVINO_THROW("Prefill stage is not available");
            }
        } else {
            return execute_stage(events, instance, regular_micro_single_token);
        }
#else
        OPENVINO_THROW("moe_gemm is only supported on systolic platforms.");
#endif
        return nullptr;
    }
};
}  // namespace

std::unique_ptr<primitive_impl> MoEGemm::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<moe_gemm>());
    return std::make_unique<MoEGemmImpl>(params);
}
}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MoEGemmImpl)
