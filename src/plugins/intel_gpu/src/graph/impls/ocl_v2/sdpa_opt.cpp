// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "sdpa_opt.hpp"
#include "sdpa_gen_micro.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "kv_cache_inst.h"
// #include "micro_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_gen_opt.hpp"
#include "utils/kernel_generator.hpp"

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

    Stage::Ptr regular_micro = make_stage<SDPAMicroGenerator>(prefill);

    SDPAOptImpl() : SDPAImplBase(SDPAOpt::get_type_info_static()) {}
    explicit SDPAOptImpl(const RuntimeParams& params) : SDPAOptImpl() {
        if (params.is_dynamic()) {
            add_stage(regular_single_token, params);
            add_stage(indirect_single_token, params);

            add_stage(regular_multi_tokens, params);
            add_stage(indirect_multi_tokens, params);

            add_stage(regular_finalization, params);
            add_stage(indirect_finalization, params);

            if (SDPAOpt::supports_micro_sdpa(params)) {
                add_stage(regular_micro, params);
            }
        } else {
            auto is_indirect = params.typed_desc<scaled_dot_product_attention>()->indirect_axis != -1;
            if (is_prefill_stage(params)) {
                if (indirect) {
                    add_stage(indirect_multi_tokens, params);
                } else if (SDPAOpt::supports_micro_sdpa(params)) {
                    add_stage(regular_micro, params);
                } else {
                    add_stage(regular_multi_tokens, params);
                }
            } else {
                const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
                bool is_ARL_H = (gfx_ver.major == 12 && gfx_ver.minor == 74);
                if (!SDPAOpt::supports_micro_sdpa(params) || is_ARL_H) {
                    add_stage(is_indirect ? indirect_single_token : regular_single_token, params);

                    if (get_partitions_num(params, SDPAStage::SINGLE_TOKEN) > 1) {
                        add_stage(is_indirect ? indirect_finalization : regular_finalization, params);
                    }
                } else {
                    add_stage(regular_micro, params);
                }
            }
        }
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        bool is_prefill = is_prefill_stage(params);
        bool is_indirect = need_indirect_load(static_cast<scaled_dot_product_attention_inst&>(instance));
        const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
        bool is_ARL_H = (gfx_ver.major == 12 && gfx_ver.minor == 74);
        bool run_micro_sdpa = has_stage(regular_micro) && (is_prefill || !is_ARL_H) && !is_indirect;

        update_rt_params(instance);

        if (run_micro_sdpa) {
            return execute_stage(events, instance, regular_micro);
        }
        if (is_prefill) {
            return execute_stage(events, instance, is_indirect ? indirect_multi_tokens : regular_multi_tokens);
        }
        const auto num_of_partitions = get_partitions_num(params, SDPAStage::SINGLE_TOKEN);

        auto ev = execute_stage(events, instance, is_indirect ? indirect_single_token : regular_single_token);
        if (num_of_partitions > 1) {
            ev = execute_stage({ev}, instance, is_indirect ? indirect_finalization : regular_finalization);
        }
        return ev;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        std::vector<BufferDescriptor> internal_buffers;

        auto desc = params.typed_desc<scaled_dot_product_attention>();

        const auto& q_l = params.input_layouts[0];

        const auto head_size = extract_channel(get_transposed_channel(ChannelName::X, desc->input_q_transpose_order), q_l);

        const auto num_of_partitions = get_partitions_num(params, SDPAStage::SINGLE_TOKEN);
        const auto is_prefill = is_prefill_stage(params);

        const size_t buf_elements_count = (num_of_partitions == 1 || is_prefill) ? 1 : params.output_layouts[0].count() / head_size * num_of_partitions;
        const size_t tmp_out_elements_count = (num_of_partitions == 1 || is_prefill) ? 1 : params.output_layouts[0].count() * num_of_partitions;

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
    auto& engine = params.get_program().get_engine();
    const auto& device_info = engine.get_device_info();

    const auto supports_microkernels = cldnn::query_microkernels_supported(engine, params.get_program().get_config());
    if (device_info.arch < gpu_arch::xe_hpg || !supports_microkernels) {
        return false;
    }

    const auto& q_layout = params.get_input_layout(0);
    const auto& k_layout = params.get_input_layout(1);
    const auto& v_layout = params.get_input_layout(2);

    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (desc->is_causal) {
        return false;
    }

    auto Q_num_heads_dim = get_num_heads(q_layout, desc->input_q_transpose_order);
    auto K_num_heads_dim = get_num_heads(k_layout, desc->input_k_transpose_order);
    auto V_num_heads_dim = get_num_heads(v_layout, desc->input_v_transpose_order);

    if (desc->input_q_transpose_order[3] != 3 || desc->input_k_transpose_order[3] != 3 || desc->input_v_transpose_order[3] != 3) {
        return false;
    }

    if (Q_num_heads_dim.is_dynamic() || K_num_heads_dim.is_dynamic() || V_num_heads_dim.is_dynamic() || K_num_heads_dim != V_num_heads_dim) {
        return false;
    }

    if (q_layout.get_partial_shape()[3].get_length() > 256) {
        return false;
    }

    auto data_inputs_num = get_data_inputs_num(*desc);

    // Do not use sdpa_micro kernel with a scalar-value mask
    return data_inputs_num <= 3 || params.get_input_layout(3).is_dynamic() || params.get_input_layout(3).count() != 1;
}

std::unique_ptr<primitive_impl> SDPAOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPAOptImpl>(params);
}

}  // namespace ov::intel_gpu::ocl
