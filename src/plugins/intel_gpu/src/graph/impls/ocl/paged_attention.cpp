// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "multi_stage_primitive.hpp"

#include "paged_attention_inst.h"
#include "paged_attention/paged_attention_kernel_selector.hpp"
#include "paged_attention/kv_cache_update_kernel_ref.hpp"
#include "paged_attention/sdpa_kernel_ref.hpp"

namespace cldnn {
namespace ocl {

struct paged_attention_impl : multi_stage_primitive<paged_attention> {
    using parent = multi_stage_primitive<paged_attention>;
    using parent::parent;
    using kv_cache_update_kernel_selector_t = kernel_selector::kv_cache_update_kernel_selector;
    using kv_cache_update_kernel_params_t = kernel_selector::kv_cache_update_params;

    using sdpa_kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using sdpa_kernel_params_t = kernel_selector::sdpa_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::paged_attention_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<paged_attention_impl>(*this);
    }

    paged_attention_impl() = default;

    paged_attention_impl(const std::vector<kernel_selector::kernel_data>& kd) : parent(kd) {
        this->can_reuse_memory = true;
    }

    void set_arguments_impl(paged_attention_inst& instance) override {}
    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage) const override { return kernel_arguments_data(); }

    enum Stage {
        KV_CACHE_UPDATE,
        SDPA
    };

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kv_cache_update_kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[Stage::KV_CACHE_UPDATE].kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::KV_CACHE_UPDATE]);

            auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
            auto bt_kernel_impl = sdpa_kernel_selector.GetImplementation(_kernels_data[Stage::SDPA].kernelName);
            bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::SDPA]);
        }
    }

    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage, size_t kernel_idx) const {
        kernel_arguments_data args;
        if (stage == Stage::KV_CACHE_UPDATE || (stage == Stage::SDPA && kernel_idx == 0))
            args.shape_info = instance.shape_info_memory_ptr();

        if (stage == Stage::KV_CACHE_UPDATE) {
            args.inputs = {  instance.input_memory_ptr(1),  /* key */
                             instance.input_memory_ptr(2),  /* value */
                             instance.input_memory_ptr(6)   /* slot_mapping */};
            args.outputs = { instance.input_memory_ptr(3),  /* key_cache */
                             instance.input_memory_ptr(4)   /* value_cache */ };
        } else if (stage == Stage::SDPA) {
            if (kernel_idx == 0) {
                args.inputs = { instance.input_memory_ptr(0), /* query */
                                instance.input_memory_ptr(3), /* key_cache */
                                instance.input_memory_ptr(4), /* value_cache */
                                instance.input_memory_ptr(7), /* max_context_len */
                                instance.input_memory_ptr(8), /* context_lens */
                                instance.input_memory_ptr(9), /* block_tables */
                                instance.input_memory_ptr(10) /* scale */ };
            } else {
                args.inputs = { instance.input_memory_ptr(8), /* context_lens */ };
            }
            args.outputs = { instance.output_memory_ptr(0) };
        }

        return args;
    }

    void execute_stage(const std::vector<event::ptr>& events, paged_attention_inst& instance, std::vector<event::ptr>& all_events, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            auto time0 = std::chrono::high_resolution_clock::now();
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;

            auto args = get_arguments(instance, stage, kd_idx);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
            }

            auto time1 = std::chrono::high_resolution_clock::now();
            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);
            auto time2 = std::chrono::high_resolution_clock::now();

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            auto time3 = std::chrono::high_resolution_clock::now();
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }

            auto time_res0 = std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count();
            auto time_res1 = std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count();
            auto time_res2 = std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count();
            GPU_DEBUG_TRACE_DETAIL << "Time execute_stage inside = " << time_res0 << "  " << time_res1 << " " << time_res2 << "\n";

            all_events.push_back(ev);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, paged_attention_inst& instance) override {
        std::vector<event::ptr> res_events;
        execute_stage(events, instance, res_events, Stage::KV_CACHE_UPDATE);

        std::vector<event::ptr> dep_events(res_events.begin(), res_events.end());
        execute_stage(dep_events, instance, res_events, Stage::SDPA);

        return aggregate_events(res_events, instance.get_network().get_stream(), res_events.size() > 1);
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        const auto query_layout = impl_param.get_input_layout(0);
        const auto key_cache_layout = impl_param.get_input_layout(3);
        const auto value_cache_layout = impl_param.get_input_layout(4);

        const auto desc = impl_param.typed_desc<paged_attention>();
        config.head_size = desc->head_size;
        config.heads_num = desc->heads_num;
        config.kv_heads_num = desc->kv_heads_num;
        config.block_size = desc->block_size;
        config.x_block_size = desc->x_block_size;
        config.max_context_len = 1;

        if (!impl_param.is_dynamic()) {
            auto query_shape = impl_param.get_input_layout(0).get_shape();
            auto key_cache_shape = impl_param.get_input_layout(3).get_shape();
            auto value_cache_shape = impl_param.get_input_layout(4).get_shape();

            auto actual_head_size = value_cache_shape[2];
            auto actual_heads_num = query_shape[2] / actual_head_size;
            auto actual_kv_heads_num = value_cache_shape[1];
            auto actual_block_size = value_cache_shape[3];
            auto actual_x_block_size = key_cache_shape[4];

            bool valid_params = config.head_size == actual_head_size &&
                                config.heads_num == actual_heads_num &&
                                config.kv_heads_num == actual_kv_heads_num &&
                                config.block_size == actual_block_size &&
                                config.x_block_size == actual_x_block_size;

            OPENVINO_ASSERT(valid_params, "[GPU] Got unexpected parameters for PA operation. ",
                            "Currently they need to be specified explicitly (this should be fixed soon by PA model conversion improvement). ",
                            "Please use the following environment variables for proper PA configuration: ",
                            "PA_HEAD_SIZE=", actual_head_size, " ",
                            "PA_HEADS_NUM=", actual_heads_num, " ",
                            "PA_KV_HEADS_NUM=", actual_kv_heads_num, " ",
                            "PA_BLOCK_SIZE=", actual_block_size, " ",
                            "PA_X_BLOCK_SIZE=", actual_x_block_size);
        }

        const size_t simd_size = 16;
        OPENVINO_ASSERT(config.head_size % simd_size == 0, "[GPU] Head size is expected to be divisible by 16");

        return config;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        kv_cache_update_kernel_params_t params;
        set_params(impl_param, params);

        auto query = impl_param.get_input_layout(0);
        auto key = impl_param.get_input_layout(1);
        auto value = impl_param.get_input_layout(2);
        auto key_cache = impl_param.get_input_layout(3);
        auto value_cache = impl_param.get_input_layout(4);
        auto slot_mapping = impl_param.get_input_layout(6);

        params.is_shape_agnostic = is_dynamic;
        params.stage_id = 0;
        params.inputs.resize(3);
        params.outputs.resize(2);
        params.inputs[0] = convert_data_tensor(key);
        params.inputs[1] = convert_data_tensor(value);
        params.inputs[2] = convert_data_tensor(slot_mapping);
        params.outputs[0] = convert_data_tensor(key_cache);
        params.outputs[1] = convert_data_tensor(value_cache);
        params.layerID = impl_param.desc->id;

        params.configuration = get_sdpa_configuration(impl_param);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(1)},
            {1, in_offsets_map.at(2)},
            {2, in_offsets_map.at(6)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
            {1, in_offsets_map.at(4)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static sdpa_kernel_params_t get_sdpa_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        const auto inputs_count = 7;
        const auto query_layout = impl_param.get_input_layout(0);
        const auto key_cache_layout = impl_param.get_input_layout(3);
        const auto value_cache_layout = impl_param.get_input_layout(4);
        const auto max_context_len_layout = impl_param.get_input_layout(7);
        const auto context_lens_layout = impl_param.get_input_layout(8);
        const auto block_tables_layout = impl_param.get_input_layout(9);
        const auto scale_layout = impl_param.get_input_layout(10);

        params.inputs.resize(inputs_count);
        params.inputs[1] = convert_data_tensor(key_cache_layout);
        params.inputs[2] = convert_data_tensor(value_cache_layout);
        params.inputs[3] = convert_data_tensor(max_context_len_layout);
        params.inputs[4] = convert_data_tensor(context_lens_layout);
        params.inputs[5] = convert_data_tensor(block_tables_layout);
        params.inputs[6] = convert_data_tensor(scale_layout);

        params.configuration = get_sdpa_configuration(impl_param);
        if (!is_dynamic) {
            auto& constant_mem = impl_param.memory_deps;

            const auto max_context_len_mem = constant_mem.at(7);
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len_mem, impl_param.get_stream());

            const auto is_prompt_stage_mem = constant_mem.at(5);
            mem_lock<uint8_t, mem_lock_type::read> is_prompt_stage_mem_lock(is_prompt_stage_mem, impl_param.get_stream());
            bool is_prompt_stage = is_prompt_stage_mem_lock[0];

            if (is_prompt_stage) {
                // Use number of slots for KV cache as a maximum context length for the first iteration
                auto slot_mapping = impl_param.get_input_layout(6);
                params.configuration.max_context_len = slot_mapping.get_shape()[1];
            } else {
                const auto max_context_len_mem = constant_mem.at(7);
                mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len_mem, impl_param.get_stream());
                params.configuration.max_context_len = max_context_len_mem_lock[0];
            }
        }

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(3)},
            {2, in_offsets_map.at(4)},
            {3, in_offsets_map.at(7)},
            {4, in_offsets_map.at(8)},
            {5, in_offsets_map.at(9)},
            {6, in_offsets_map.at(10)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(kv_cache_update_kernel_selector.get_best_kernel(kv_cache_update_kernel_params));

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params));

        return cldnn::make_unique<paged_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::KV_CACHE_UPDATE].update_dispatch_data_func)(kv_cache_update_kernel_params, _kernels_data[Stage::KV_CACHE_UPDATE]);

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::SDPA].update_dispatch_data_func)(sdpa_kernel_params, _kernels_data[Stage::SDPA]);
    }
};

namespace detail {

attach_paged_attention_impl::attach_paged_attention_impl() {
    auto types = { data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<paged_attention>::add(impl_types::ocl,
                                             shape_types::dynamic_shape,
                                             paged_attention_impl::create,
                                             types,
                                             formats);

    implementation_map<paged_attention>::add(impl_types::ocl,
                                             shape_types::static_shape,
                                             paged_attention_impl::create,
                                             types,
                                             formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::paged_attention_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
