// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "multi_stage_primitive.hpp"
#include "paged_attention_inst.h"

#include "sdpa/sdpa_kernel_base.h"
#include "sdpa/sdpa_kernel_selector.h"
#include "sdpa/pa_kv_cache_update_kernel_ref.h"
#include "sdpa/pa_sdpa_kernel_opt.h"

namespace cldnn {
namespace ocl {

struct paged_attention_impl : multi_stage_primitive<paged_attention> {
    using parent = multi_stage_primitive<paged_attention>;
    using parent::parent;

    using sdpa_kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using sdpa_kernel_params_t = kernel_selector::sdpa_params;

    using pa_sdpa_kernel_selector_t = kernel_selector::pa_sdpa_kernel_selector;
    using pa_sdpa_kernel_params_t = kernel_selector::pa_sdpa_params;

    using kv_cache_update_kernel_selector_t = kernel_selector::kv_cache_update_kernel_selector;
    using kv_cache_update_kernel_params_t = kernel_selector::kv_cache_update_params;

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
        SDPA,
        PA_SDPA,
    };

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            OPENVINO_ASSERT(false, "Unimplemented load call");
            // auto& kernel_selector = kv_cache_update_kernel_selector_t::Instance();
            // auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[Stage::KV_CACHE_UPDATE].kernelName);
            // kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::KV_CACHE_UPDATE]);

            auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
            auto bt_kernel_impl = sdpa_kernel_selector.GetImplementation(_kernels_data[Stage::SDPA].kernelName);
            bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::SDPA]);
        }
    }

    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage, size_t kernel_idx) const {
        kernel_arguments_data args;
        if (stage == Stage::KV_CACHE_UPDATE || stage == Stage::SDPA || (stage == Stage::PA_SDPA && kernel_idx == 0))
            args.shape_info = instance.shape_info_memory_ptr();

        if (stage == Stage::KV_CACHE_UPDATE) {
            args.inputs = {  instance.input_memory_ptr(1),  /* key */
                             instance.input_memory_ptr(2),  /* value */
                             instance.input_memory_ptr(6),  /* subsequence_begins */
                             instance.input_memory_ptr(7),  /* block_indices */
                             instance.input_memory_ptr(5),  /* past_lens */
                             instance.input_memory_ptr(8),  /* block_indices_begins */ };

            args.outputs = { instance.input_memory_ptr(3),  /* key_cache */
                             instance.input_memory_ptr(4)   /* value_cache */ };
        } else if (stage == Stage::SDPA) {
            args.inputs = { instance.input_memory_ptr(0), /* query */
                            instance.input_memory_ptr(1), /* key */
                            instance.input_memory_ptr(2), /* value */
                            instance.input_memory_ptr(6), /* subsequence_begins */ };

            args.outputs = { instance.output_memory_ptr(0) };
        } else if (stage == Stage::PA_SDPA) {
            if (kernel_idx == 0) {
                args.inputs = { instance.input_memory_ptr(0), /* query */
                                instance.input_memory_ptr(3), /* key_cache */
                                instance.input_memory_ptr(4), /* value_cache */
                                instance.input_memory_ptr(5), /* past_lens */
                                instance.input_memory_ptr(6), /* subsequence_begins */
                                instance.input_memory_ptr(7), /* block_indices */
                                instance.input_memory_ptr(8), /* block_indices_begins */ };
            } else {
                args.inputs = { instance.input_memory_ptr(5), /* past_lens */ };

                args.outputs = { instance.output_memory_ptr(0) };
            }

            args.outputs = { instance.output_memory_ptr(0) };
        }

        return args;
    }

    std::set<size_t> get_lockable_internal_buffers() const override {
        return { 3, 4, 5 };
    };

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

        // TODO: optimize get intermediate buffer function with respect to stage

        std::vector<event::ptr> dep_events(res_events.begin(), res_events.end());
        if (is_prefill_stage(*instance.get_impl_params())) {
            execute_stage(dep_events, instance, res_events, Stage::SDPA);
        } else {
            execute_stage(dep_events, instance, res_events, Stage::PA_SDPA);
        }

        return aggregate_events(res_events, instance.get_network().get_stream(), res_events.size() > 1);
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        const auto key_cache_shape = impl_param.get_input_layout(3).get_partial_shape();
        if (key_cache_shape[2].is_static())
            config.head_size = key_cache_shape[2].get_length();

        if (key_cache_shape[1].is_static())
            config.heads_num = key_cache_shape[1].get_length();

        config.is_causal = true;

        const auto desc = impl_param.typed_desc<paged_attention>();
        if (desc->scale_val.has_value()) {
            config.has_scale_val = true;
            config.scale_val = desc->scale_val.value();
        }

        return config;
    }

    static sdpa_kernel_params_t get_sdpa_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        const auto query_layout = impl_param.get_input_layout(0);
        const auto key_layout = impl_param.get_input_layout(1);
        const auto value_layout = impl_param.get_input_layout(2);
        const auto subsequence_begins_layout = impl_param.get_input_layout(6);

        const auto inputs_count = 4;
        params.inputs.resize(inputs_count);
        params.inputs[0] = convert_data_tensor(query_layout);
        params.inputs[1] = convert_data_tensor(key_layout);
        params.inputs[2] = convert_data_tensor(value_layout);
        params.inputs[3] = convert_data_tensor(subsequence_begins_layout);
        params.conf = get_sdpa_configuration(impl_param);

        params.is_paged_attention = true;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(1)},
            {2, in_offsets_map.at(2)},
            {3, in_offsets_map.at(6)},
            {4, in_offsets_map.at(9)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        if (is_prefill_stage(impl_param)) {
            const auto& input_mem = impl_param.memory_deps;

            auto align_seq_len = [&](cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read>& subsequence_begins, size_t target_seq_len_block_size = 16) {
                // TODO: can be optimized if vLLM's block_size (key_cache[3]) == target_seq_len_block_size
                // Then aligned_seq_len = block_indices_shape[0] * target_seq_len_block_size
                int64_t aligned_seq_len = 0;
                for (size_t i = 0; i < subsequence_begins.size() - 1; i++) {
                    auto prompt_length = subsequence_begins[i + 1] - subsequence_begins[i];
                    aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
                    GPU_DEBUG_TRACE_DETAIL << "Res: " << i << ", " << prompt_length << " " << target_seq_len_block_size << " " << align_to(prompt_length, target_seq_len_block_size) << "\n";
                }

                GPU_DEBUG_TRACE_DETAIL << "Aligned seq_len = " << aligned_seq_len << " size=" << subsequence_begins.size() << "\n";
                return aligned_seq_len;
            };
            GPU_DEBUG_TRACE_DETAIL << "update_dispatch_data1\n";

            const auto subsequence_begins_mem = input_mem.at(6);
            mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *impl_param.strm);
            GPU_DEBUG_TRACE_DETAIL << "update_dispatch_data2\n";

            params.paged_attention_aligned_seq_len = align_seq_len(subsequence_begins_mem_lock);
        }
        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        kv_cache_update_kernel_params_t params;
        set_params(impl_param, params);

        auto query = impl_param.get_input_layout(0);
        auto key = impl_param.get_input_layout(1);
        auto value = impl_param.get_input_layout(2);
        auto key_cache = impl_param.get_input_layout(3);
        auto value_cache = impl_param.get_input_layout(4);
        auto past_lens = impl_param.get_input_layout(5);
        auto subsequence_begins = impl_param.get_input_layout(6);
        auto block_indices = impl_param.get_input_layout(7);
        auto block_indices_begins = impl_param.get_input_layout(8);

        params.is_shape_agnostic = is_dynamic;
        params.stage_id = 0; // TODO: can be removed
        params.inputs.resize(6);
        params.outputs.resize(2);
        params.inputs[0] = convert_data_tensor(key);
        params.inputs[1] = convert_data_tensor(value);
        params.inputs[2] = convert_data_tensor(subsequence_begins);
        params.inputs[3] = convert_data_tensor(block_indices);
        params.inputs[4] = convert_data_tensor(past_lens);
        params.inputs[5] = convert_data_tensor(block_indices_begins);
        params.outputs[0] = convert_data_tensor(key_cache);
        params.outputs[1] = convert_data_tensor(value_cache);
        params.layerID = impl_param.desc->id; // TODO: can be removed

        GPU_DEBUG_TRACE_DETAIL << "get_kv_cache_update_kernel_params 2\n";

        params.conf = get_sdpa_configuration(impl_param);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(1)},
            {1, in_offsets_map.at(2)},
            {2, in_offsets_map.at(6)},
            {3, in_offsets_map.at(7)},
            {4, in_offsets_map.at(5)},
            {5, in_offsets_map.at(8)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
            {1, in_offsets_map.at(4)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }


    static pa_sdpa_kernel_params_t get_pa_sdpa_params(const kernel_impl_params& impl_param, bool is_dynamic = false) {
        pa_sdpa_kernel_params_t params;
        set_params(impl_param, params);

        auto query = impl_param.get_input_layout(0);
        auto key_cache = impl_param.get_input_layout(3);
        auto value_cache = impl_param.get_input_layout(4);
        auto past_lens = impl_param.get_input_layout(5);
        auto subsequence_begins = impl_param.get_input_layout(6);
        auto block_indices = impl_param.get_input_layout(7);
        auto block_indices_begins = impl_param.get_input_layout(8);

        auto output = impl_param.get_output_layout(0);

        params.is_shape_agnostic = is_dynamic;
        params.stage_id = 0; // TODO: can be removed
        params.inputs.resize(7);
        params.outputs.resize(1);
        params.inputs[0] = convert_data_tensor(query);
        params.inputs[1] = convert_data_tensor(key_cache);
        params.inputs[2] = convert_data_tensor(value_cache);
        params.inputs[3] = convert_data_tensor(past_lens);
        params.inputs[4] = convert_data_tensor(subsequence_begins);
        params.inputs[5] = convert_data_tensor(block_indices);
        params.inputs[6] = convert_data_tensor(block_indices_begins);
        params.outputs[0] = convert_data_tensor(output);
        params.layerID = impl_param.desc->id; // TODO: can be removed

        GPU_DEBUG_TRACE_DETAIL << "get_pa_sdpa_params 2\n";

        params.conf = get_sdpa_configuration(impl_param);

        if (!is_prefill_stage(impl_param) && !is_dynamic) {
            const auto& input_mem = impl_param.memory_deps;
            const auto max_context_len = input_mem.at(12);
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *impl_param.strm);
            params.max_context_len = max_context_len_mem_lock[0];
        }

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;

        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(3)},
            {2, in_offsets_map.at(4)},
            {3, in_offsets_map.at(5)},
            {4, in_offsets_map.at(6)},
            {5, in_offsets_map.at(7)},
            {6, in_offsets_map.at(8)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static bool is_prefill_stage(const kernel_impl_params& impl_param) {
        auto query_shape = impl_param.get_input_layout(0).get_partial_shape();
        auto past_lens_shape = impl_param.get_input_layout(5).get_partial_shape();

        if (query_shape.is_static() && past_lens_shape.is_static()) {
            GPU_DEBUG_TRACE_DETAIL << "Prefill stage: " << (query_shape[0].get_length() != past_lens_shape[0].get_length()) << " " << query_shape[0].get_length() << " " << past_lens_shape[0].get_length() << "\n";
            return query_shape[0].get_length() != past_lens_shape[0].get_length();
        }

        GPU_DEBUG_TRACE_DETAIL << "Prefill stage: false\n";
        return false;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;

        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(kv_cache_update_kernel_selector.get_best_kernel(kv_cache_update_kernel_params));

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params));

        auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, impl_param.is_dynamic());
        auto& pa_sdpa_kernel_selector = pa_sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(pa_sdpa_kernel_selector.get_best_kernel(pa_sdpa_kernel_params));

        GPU_DEBUG_TRACE_DETAIL << "PA Created with " << kernels_data.size() << " kds\n";

        return cldnn::make_unique<paged_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        GPU_DEBUG_TRACE_DETAIL << "update_dispatch_data\n";

        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, impl_param.is_dynamic());
        (_kernels_data[Stage::KV_CACHE_UPDATE].update_dispatch_data_func)(kv_cache_update_kernel_params, _kernels_data[Stage::KV_CACHE_UPDATE]);

        if (is_prefill_stage(impl_param)) {
            auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[Stage::SDPA].update_dispatch_data_func)(sdpa_kernel_params, _kernels_data[Stage::SDPA]);
        } else {
            auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[Stage::PA_SDPA].update_dispatch_data_func)(pa_sdpa_kernel_params, _kernels_data[Stage::PA_SDPA]);
        }
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
