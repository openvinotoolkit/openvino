// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "sdpa_gen_micro.hpp"
// clang-format on

#include "sdpa_opt.hpp"

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "../../cm/paged_attention_gen.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "primitive_inst.h"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_gen_opt.hpp"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

class SDPAOptImpl : public SDPAImplBase {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SDPAOptImpl)
    static constexpr bool indirect = true;
    static constexpr bool prefill = true;

    Stage::Ptr indirect_single_token = make_stage<SDPAOptGeneratorSingleToken>(indirect);
    Stage::Ptr regular_single_token = make_stage<SDPAOptGeneratorSingleToken>(!indirect);

    Stage::Ptr indirect_multi_tokens = make_stage<SDPAOptGeneratorMultiToken>(indirect);
    Stage::Ptr regular_multi_tokens = make_stage<SDPAOptGeneratorMultiToken>(!indirect);

    Stage::Ptr indirect_finalization = make_stage<SDPAOptGeneratorFinalization>(indirect);
    Stage::Ptr regular_finalization = make_stage<SDPAOptGeneratorFinalization>(!indirect);

#ifdef ENABLE_ONEDNN_FOR_GPU
    Stage::Ptr regular_micro_single_token = make_stage<SDPAMicroGenerator>(!prefill);
    Stage::Ptr regular_micro_multi_tokens = make_stage<SDPAMicroGenerator>(prefill);
#endif
    Stage::Ptr sdpa_prefill_cm = make_stage<cm::PagedAttentionSDPAGeneratorMultiToken>();

    bool supports_cm_sdpa(const kernel_impl_params& params) const {
        auto& engine = params.get_program().get_engine();
        // 0 - unknown, 1 - supported, 2 - not supported
        static char supports_cm = 0;

        if (supports_cm == 0) {
            auto query_result = cldnn::check_cm_jit_support(engine, params.get_program().get_config());
            if (params.get_device_info().arch < gpu_arch::xe_hpg || !query_result) {
                supports_cm = 2;
            } else {
                supports_cm = 1;
            }
        }
        return supports_cm == 1;
    }

    SDPAOptImpl() : SDPAImplBase(SDPAOpt::get_type_info_static()) {}
    explicit SDPAOptImpl(const RuntimeParams& impl_param) : SDPAOptImpl() {
        auto params = SDPABase::requires_shape_canonicalization(impl_param) ? SDPABase::static_canonicalize_shapes(impl_param) : impl_param;
        GPU_DEBUG_TRACE_DETAIL << "create stages for dynamic = " << params.is_dynamic() << "\n";
        const bool use_cm_sdpa = true; //supports_cm_sdpa(params);
        if (params.is_dynamic()) {
            GPU_DEBUG_TRACE_DETAIL << "add stages for dynamic ...\n";
            add_stage(regular_single_token, params);
            add_stage(indirect_single_token, params);
            add_stage(regular_multi_tokens, params);
            add_stage(indirect_multi_tokens, params);
            add_stage(regular_finalization, params);
            add_stage(indirect_finalization, params);

            GPU_DEBUG_TRACE_DETAIL << "supports_cm_sdpa = " << use_cm_sdpa << "\n";
            if (use_cm_sdpa) {
                GPU_DEBUG_TRACE_DETAIL << "add stage for cm_sdpa dynamic with prefill_stage \n";
                add_stage(sdpa_prefill_cm, params);
            }
#ifdef ENABLE_ONEDNN_FOR_GPU
            if (SDPAOpt::supports_micro_sdpa(params) && !use_cm_sdpa) {
                GPU_DEBUG_TRACE_DETAIL << "add stage for micro_sdpa  dynamic ...\n";
                add_stage(regular_micro_multi_tokens, params);
                add_stage(regular_micro_single_token, params);
            }
#endif
            GPU_DEBUG_TRACE_DETAIL << "add stage for dynamic done \n";
        } else {
            auto is_indirect = params.typed_desc<scaled_dot_product_attention>()->indirect_axis != -1;
            GPU_DEBUG_TRACE_DETAIL << "add stage for non-dynamic, is_indirect = " << is_indirect << "\n";
            if (is_prefill_stage(params) || unaligned_head_size(params)) {
                GPU_DEBUG_TRACE_DETAIL << "add stage for cm_sdpa non-dynamic with prefill_stage \n";
                if (use_cm_sdpa)
                    add_stage(sdpa_prefill_cm, params);
                if (is_indirect) {
                    GPU_DEBUG_TRACE_DETAIL << "add stage for indirect non-dynamic with prefill_stage \n";
                    add_stage(indirect_multi_tokens, params);
#ifdef ENABLE_ONEDNN_FOR_GPU
                } else if (SDPAOpt::supports_micro_sdpa(params)) {
                    GPU_DEBUG_TRACE_DETAIL << "add stage for micro_sdpa non-dynamic with prefill_stage \n";
                    add_stage(regular_micro_multi_tokens, params);
                    // Sometimes micro kernel will fail due to "Insufficient registers in requested bundle",
                    // In this case, fallback to opt kernel.
                    if (!has_stage(regular_micro_multi_tokens)) {
                        GPU_DEBUG_TRACE_DETAIL << "fail to create micro kernel, fallback to regular_multi_tokens for prefill \n";
                        add_stage(regular_multi_tokens, params);
                    }
#endif
                } else {
                    GPU_DEBUG_TRACE_DETAIL << "add stage regular_multi_tokens kernels with prefill_stage \n";
                    add_stage(regular_multi_tokens, params);
                }
            } else {
                GPU_DEBUG_TRACE_DETAIL << "add stage single_tokens \n";
#ifdef ENABLE_ONEDNN_FOR_GPU
                const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
                bool is_ARL_H = (gfx_ver.major == 12 && gfx_ver.minor == 74);
                bool can_use_micro_sdpa = SDPAOpt::supports_micro_sdpa(params) && !is_ARL_H && !is_indirect;
                if (can_use_micro_sdpa) {
                    add_stage(regular_micro_single_token, params);
                }
#endif
                add_stage(is_indirect ? indirect_single_token : regular_single_token, params);
                if (get_partitions_num(params, SDPAStage::SINGLE_TOKEN) > 1) {
                    add_stage(is_indirect ? indirect_finalization : regular_finalization, params);
                }
            }
        }
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        auto new_params = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
        bool is_prefill = is_prefill_stage(new_params);
        bool is_indirect = need_indirect_load(static_cast<scaled_dot_product_attention_inst&>(instance));
        GPU_DEBUG_TRACE_DETAIL << "execute indirect = " << is_indirect << ", prefill = " << is_prefill << "\n";
        update_rt_params(instance);

        if(has_stage(sdpa_prefill_cm) && is_prefill) {
            GPU_DEBUG_TRACE_DETAIL << "execute sdpa_prefill_cm \n";
            return execute_stage(events, instance, sdpa_prefill_cm);
        }
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (has_stage(regular_micro_multi_tokens) && is_prefill && !is_indirect) {
            GPU_DEBUG_TRACE_DETAIL << "execute regular_micro_multi_tokens for prefill \n";
            return execute_stage(events, instance, regular_micro_multi_tokens);
        }
#endif
        // TODO: Unaligned head size is currently supported by only multi tokens kernel.
        // So far this case was observed only from the non-lm models such as vision embedding model.
        // If we need to optimize unaligned head size SDPA for 2nd+ token phase of LM model,
        // we'll need to fix single_token kernel to support unaligned head size.
        if (is_prefill || unaligned_head_size(new_params)) {
            GPU_DEBUG_TRACE_DETAIL << "execute multi_tokens for prefill with indirect = " << is_indirect << "\n";
            return execute_stage(events, instance, is_indirect ? indirect_multi_tokens : regular_multi_tokens);
        }
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (has_stage(regular_micro_single_token) && !is_indirect) {
            return execute_stage(events, instance, regular_micro_single_token);
        }
#endif
        const auto num_of_partitions = get_partitions_num(new_params, SDPAStage::SINGLE_TOKEN);
        GPU_DEBUG_TRACE_DETAIL << "execute single_tokens with indirect = " << is_indirect << "\n";
        auto ev = execute_stage(events, instance, is_indirect ? indirect_single_token : regular_single_token);
        if (num_of_partitions > 1) {
            GPU_DEBUG_TRACE_DETAIL << "execute single_tokens_finalization (" << num_of_partitions << ") with indirect = " << is_indirect << "\n";
            ev = execute_stage({ev}, instance, is_indirect ? indirect_finalization : regular_finalization);
        }
        return ev;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        std::vector<BufferDescriptor> internal_buffers;
        auto desc = params.typed_desc<scaled_dot_product_attention>();
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto params_canonicalization = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
        const auto head_size = get_head_size(params_canonicalization.get_input_layout(0), extended_input_q_transpose_order);

        const auto num_of_partitions = get_partitions_num(params_canonicalization, SDPAStage::SINGLE_TOKEN);
        const auto is_prefill = is_prefill_stage(params_canonicalization);
        const size_t buf_elements_count = (num_of_partitions == 1 || is_prefill) ? 1 : params.output_layouts[0].count() / head_size * num_of_partitions;
        const size_t tmp_out_elements_count = (num_of_partitions == 1 || is_prefill) ? 2 : params.output_layouts[0].count() * num_of_partitions;

        internal_buffers.emplace_back(buf_elements_count, ov::element::f32);
        internal_buffers.emplace_back(buf_elements_count, ov::element::f32);
        internal_buffers.emplace_back(tmp_out_elements_count, params.output_layouts[0].data_type);

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SDPAOptImpl>(this);
    }
};

bool SDPAOpt::supports_micro_sdpa(const RuntimeParams& params) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& engine = params.get_program().get_engine();
    const auto& device_info = engine.get_device_info();

    if (device_info.supports_immad) {
        const auto supports_microkernels = cldnn::query_microkernels_supported(engine, params.get_program().get_config());
        if (device_info.arch < gpu_arch::xe_hpg || !supports_microkernels) {
            return false;
        }
    } else {
        return false;
    }

    const auto& q_layout = params.get_input_layout(0);
    const auto& k_layout = params.get_input_layout(1);
    const auto& v_layout = params.get_input_layout(2);
    auto desc = params.typed_desc<scaled_dot_product_attention>();

    // Will check it later to decide whether support micro kernel
    // if (desc->indirect_axis != -1) {
    //     // Micro kernel does not support indirect axis
    //     return false;
    // }

    auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
    auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
    auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);

    ov::Dimension Q_num_heads_dim = get_num_heads(q_layout, extended_input_q_transpose_order);
    ov::Dimension K_num_heads_dim = get_num_heads(k_layout, extended_input_k_transpose_order);
    ov::Dimension V_num_heads_dim = get_num_heads(v_layout, extended_input_v_transpose_order);

    if (extended_input_q_transpose_order[3] != 3 || extended_input_k_transpose_order[3] != 3 || extended_input_v_transpose_order[3] != 3) {
        return false;
    }

    if (Q_num_heads_dim.is_dynamic() || K_num_heads_dim.is_dynamic() || V_num_heads_dim.is_dynamic() || K_num_heads_dim != V_num_heads_dim) {
        return false;
    }

    auto K_head_size = get_head_size(k_layout, extended_input_k_transpose_order);
    auto V_head_size = get_head_size(v_layout, extended_input_v_transpose_order);
    if (K_head_size != V_head_size || K_head_size > 256 || V_head_size > 256) {
        return false;
    }

    auto data_inputs_num = get_data_inputs_num(*desc);
    // TODO: To support sdpa_micro kernel with non-const scalar mask / scale inputs
    const auto mask_idx = 3lu;
    if (!desc->attn_mask_val.has_value() && data_inputs_num > mask_idx && !params.get_input_layout(mask_idx).is_dynamic() &&
        params.get_input_layout(mask_idx).count() == 1) {
        return false;
    }
    if (q_layout.get_partial_shape()[mask_idx].get_length() > 256) {
        return false;
    }

    // Known limitation: In vision encoding model of qwen-vl, when the shape of sdpa is 3D and num_heads is 1,
    // there is an accuracy issue with sdpa_micro kernel. Therefore, it is currently restricted to execute with sdpa_opt kernel.
    const bool is_output_rank_4d = desc->output_transpose_order.size() == 4;
    if (!is_output_rank_4d)
        return false;

    return true;
#else
    return false;
#endif
}

std::unique_ptr<primitive_impl> SDPAOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    try {
        return std::make_unique<SDPAOptImpl>(params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create SDPAOptImpl: ", e.what());
    }
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SDPAOptImpl)