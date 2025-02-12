// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "sdpa_opt.hpp"
#include "sdpa_base.hpp"
#include "utils/jitter.hpp"
#include "utils/kernel_base.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "primitive_ocl_base.hpp"
#include "kv_cache_inst.h"
#include "scaled_dot_product_attention_inst.h"

#include "micro_utils.hpp"
#include "sdpa_gen_micro.hpp"
#include "sdpa_gen_opt.hpp"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

class SDPAOptImpl : public SDPAImplBase {
public:
    static constexpr const size_t INDIRECT_STAGE = 10;
    static constexpr const size_t REGULAR_STAGE = 20;

    SDPAOptImpl(const kernel_impl_params& params) : SDPAImplBase(std::string(SDPAOpt::get_type_info_static().name)) {
        if (params.is_dynamic()) {
            add_stage<SDPAOptGeneratorSingleToken, REGULAR_STAGE + KernelsTypes::SINGLE_TOKEN>(params, false);
            add_stage<SDPAOptGeneratorSingleToken, INDIRECT_STAGE + KernelsTypes::SINGLE_TOKEN>(params, true);

            add_stage<SDPAOptGeneratorMultiToken, REGULAR_STAGE + KernelsTypes::MULTI_TOKENS>(params, false);
            add_stage<SDPAOptGeneratorMultiToken, INDIRECT_STAGE + KernelsTypes::MULTI_TOKENS>(params, true);

            add_stage<SDPAOptGeneratorFinalization, REGULAR_STAGE + KernelsTypes::FINALIZATION>(params, false);

            if (SDPAOpt::supports_micro_sdpa(params))
                add_stage<SDPAMicroGenerator, REGULAR_STAGE + KernelsTypes::MICRO>(params, false);
        } else {
            auto indirect = params.typed_desc<scaled_dot_product_attention>()->indirect_axis != -1;
            if (is_prefill_stage(params)) {
                if (indirect)
                    add_stage<SDPAOptGeneratorMultiToken, INDIRECT_STAGE + KernelsTypes::MULTI_TOKENS>(params, false);
                else if (SDPAOpt::supports_micro_sdpa(params))
                    add_stage<SDPAMicroGenerator, REGULAR_STAGE + KernelsTypes::MICRO>(params, false);
                else
                    add_stage<SDPAOptGeneratorMultiToken, REGULAR_STAGE + KernelsTypes::MULTI_TOKENS>(params, false);
            } else {
                const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
                bool is_ARL_H = (gfx_ver.major == 12 && gfx_ver.minor == 74);
                if (!SDPAOpt::supports_micro_sdpa(params) || is_ARL_H) {
                    if (indirect)
                        add_stage<SDPAOptGeneratorSingleToken, INDIRECT_STAGE + KernelsTypes::SINGLE_TOKEN>(params, false);
                    else
                        add_stage<SDPAOptGeneratorSingleToken, REGULAR_STAGE + KernelsTypes::SINGLE_TOKEN>(params, true);

                    if (get_partitions_num(params, KernelsTypes::SINGLE_TOKEN) > 1)
                        add_stage<SDPAOptGeneratorFinalization, REGULAR_STAGE + KernelsTypes::FINALIZATION>(params, false);
                } else {
                    add_stage<SDPAMicroGenerator, REGULAR_STAGE + KernelsTypes::MICRO>(params, false);
                }
            }
        }
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        bool is_prefill = is_prefill_stage(params);
        auto stage_type = need_indirect_load(static_cast<scaled_dot_product_attention_inst&>(instance)) ? INDIRECT_STAGE : REGULAR_STAGE;
        const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
        bool is_ARL_H = (gfx_ver.major == 12 && gfx_ver.minor == 74);
        bool run_micro_sdpa = has_stage(REGULAR_STAGE + KernelsTypes::MICRO) && (is_prefill || !is_ARL_H) && stage_type == REGULAR_STAGE;


        if (run_micro_sdpa) {
            return execute_stage(events, instance, REGULAR_STAGE + KernelsTypes::MICRO);
        } else if (is_prefill) {
            return execute_stage(events, instance, stage_type + KernelsTypes::MULTI_TOKENS);
        } else {
            const auto num_of_partitions = get_partitions_num(params, KernelsTypes::SINGLE_TOKEN);

            auto ev = execute_stage(events, instance, stage_type + KernelsTypes::SINGLE_TOKEN);
            if (num_of_partitions > 1) {
                ev = execute_stage({ev}, instance, stage_type + KernelsTypes::FINALIZATION);
            }
            return ev;
        }
    }

    std::vector<layout> get_internal_buffer_layouts(const kernel_impl_params& params) const override {
        std::vector<layout> bufs;

        const auto& q_l = params.input_layouts[0];
        // if (!params.is_dynamic()) {
        //     const auto& k_l = params.input_layouts[1];

        //     const auto& q_shape = q_l.get_shape();
        //     const auto& k_shape = k_l.get_shape();
        //     const size_t buf_size = q_l.count() / q_shape[3] * k_shape[2];

        //     bufs = { layout{ov::PartialShape{static_cast<int64_t>(buf_size)}, q_l.data_type, format::bfyx } };
        // } else {
            auto buf = layout{ layout{ov::PartialShape{4}, q_l.data_type, format::bfyx } };
            bufs = { buf, buf, buf };
        // }

        return bufs;
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<SDPAOptImpl>(*this);
    }
};

bool SDPAOpt::supports_micro_sdpa(const kernel_impl_params& params) {
    auto& engine = params.get_program().get_engine();
    const auto& device_info = engine.get_device_info();

    const auto supports_microkernels = cldnn::query_microkernels_supported(engine, params.get_program().get_config());
    if (device_info.arch < gpu_arch::xe_hpg || !supports_microkernels)
        return false;


    const auto& q_layout = params.get_input_layout(0);
    const auto& k_layout = params.get_input_layout(1);
    const auto& v_layout = params.get_input_layout(2);

    auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (desc->is_causal)
        return false;

    auto Q_num_heads_dim = get_num_heads(q_layout, desc->input_q_transpose_order);
    auto K_num_heads_dim = get_num_heads(k_layout, desc->input_k_transpose_order);
    auto V_num_heads_dim = get_num_heads(v_layout, desc->input_v_transpose_order);

    if (desc->input_q_transpose_order[3] != 3 || desc->input_k_transpose_order[3] != 3 || desc->input_v_transpose_order[3] != 3)
        return false;

    if (Q_num_heads_dim.is_dynamic() || K_num_heads_dim.is_dynamic() || V_num_heads_dim.is_dynamic() || K_num_heads_dim != V_num_heads_dim)
        return false;

    if (q_layout.get_partial_shape()[3].get_length() > 256)
        return false;

    auto data_inputs_num = get_data_inputs_num(*desc);

    // Do not use sdpa_micro kernel with a scalar-value mask
    if (data_inputs_num > 3 && !params.get_input_layout(3).is_dynamic() && params.get_input_layout(3).count() == 1)
        return false;

    return true;
}

std::unique_ptr<primitive_impl> SDPAOpt::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPAOptImpl>(params);
}

}  // namespace ov::intel_gpu::ocl
