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

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
            auto kv_cache_update_kernel_impl = kv_cache_update_kernel_selector.GetImplementation(_kernels_data[Stage::KV_CACHE_UPDATE].kernelName);
            kv_cache_update_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::KV_CACHE_UPDATE]);

            auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
            auto sdpa_kernel_impl = sdpa_kernel_selector.GetImplementation(_kernels_data[Stage::SDPA].kernelName);
            sdpa_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::SDPA]);

            auto& pa_sdpa_kernel_selector = pa_sdpa_kernel_selector_t::Instance();
            auto pa_sdpa_kernel_impl = pa_sdpa_kernel_selector.GetImplementation(_kernels_data[Stage::PA_SDPA].kernelName);
            pa_sdpa_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::PA_SDPA]);
        }
    }

    std::vector<layout> get_internal_buffer_layouts_impl() const override {
        auto add_internal_buffers = [](std::vector<layout>& layouts, const kernel_selector::KernelData& kd) {
            if (kd.internalBufferSizes.empty())
                return;

            auto dtype = from_data_type(kd.internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            for (auto size : kd.internalBufferSizes) {
                layout inbuf_layout = {dtype, format::bfyx, // simple linear format (flattern to x channel)
                                       {1, 1, 1, (tensor::value_type)(size / bpp)}};
                layouts.push_back(inbuf_layout);
            }
        };

        std::vector<layout> layouts;
        add_internal_buffers(layouts, _kernels_data[Stage::KV_CACHE_UPDATE]);
        add_internal_buffers(layouts, _kernels_data[Stage::PA_SDPA]);

        return layouts;
    }

    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage, size_t kernel_idx, bool is_mixed_mode) const {
        const auto desc = instance.get_node().as<paged_attention>().get_primitive();

        kernel_arguments_data args;
        if (stage == Stage::KV_CACHE_UPDATE || stage == Stage::SDPA)
            args.shape_info = instance.shape_info_memory_ptr();

        if (stage == Stage::KV_CACHE_UPDATE) {
            args.inputs = {  instance.key_memory_ptr(),
                             instance.value_memory_ptr(),
                             instance.past_lens_memory_ptr(),
                             instance.block_indices_memory_ptr(),
                             instance.block_indices_begins_memory_ptr(),
                             instance.subsequence_begins_memory_ptr() };

            args.outputs = { instance.key_cache_memory_ptr(),
                             instance.value_cache_memory_ptr() };
        } else if (stage == Stage::SDPA) {
            args.inputs = {  instance.input_memory_ptr(0),
                             instance.key_memory_ptr(),
                             instance.value_memory_ptr(),
                             instance.subsequence_begins_memory_ptr() };

            if (!desc->scale_val.has_value()) {
                args.inputs.push_back(instance.input_memory_ptr(9));
            }

            if (desc->has_alibi) {
                args.inputs.push_back(instance.alibi_memory_ptr());
            }

            args.outputs = { instance.output_memory_ptr(0) };
        } else if (stage == Stage::PA_SDPA) {
            if (kernel_idx == 0 || kernel_idx == 1) {
                args.shape_info = instance.shape_info_memory_ptr();

                args.inputs = { instance.input_memory_ptr(0),
                                instance.key_cache_memory_ptr(),
                                instance.value_cache_memory_ptr(),
                                instance.past_lens_memory_ptr(),
                                instance.block_indices_memory_ptr(),
                                instance.block_indices_begins_memory_ptr() };

                if (is_mixed_mode) {
                    // Multi tokens kernel version has additional subsequence_begins_memory memory
                    // dependency
                    args.inputs.push_back(instance.subsequence_begins_memory_ptr());
                }

                if (!desc->scale_val.has_value()) {
                    args.inputs.push_back(instance.input_memory_ptr(9));
                }

                if (desc->has_alibi) {
                    args.inputs.push_back(instance.alibi_memory_ptr());
                }
            } else {
                args.inputs = { instance.past_lens_memory_ptr() };

                if (is_mixed_mode) {
                    // Multi tokens kernel version has additional subsequence_begins_memory memory
                    // dependency
                    args.inputs.push_back(instance.subsequence_begins_memory_ptr());
                }
            }

            args.outputs = { instance.output_memory_ptr(0) };
        }

        return args;
    }

    std::set<size_t> get_lockable_internal_buffers() const override {
        return std::set<size_t>{ 0, 1, 2, /* SDPA and KV_CACHE_UPDATE indexes configuration */
                                 6, /* PA_SDPA multiple tokens mode */ };
    };

    void execute_stage(const std::vector<event::ptr>& events,
                       paged_attention_inst& instance,
                       std::vector<event::ptr>& all_events,
                       size_t stage,
                       bool is_mixed_mode) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }

        // Stages SDPA and KV_CACHE_UPDATE reuse the same internal buffers at prefill stage
        size_t internal_buffers_offset = 0;
        size_t internal_buffers_count = 0;
        if (stage == Stage::PA_SDPA) {
            internal_buffers_offset = _kernels_data[Stage::KV_CACHE_UPDATE].internalBufferSizes.size();
            internal_buffers_count = _kernels_data[Stage::PA_SDPA].internalBufferSizes.size();
        } else {
            internal_buffers_count = _kernels_data[Stage::KV_CACHE_UPDATE].internalBufferSizes.size();
        }

        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;

            auto args = get_arguments(instance, stage, kd_idx, is_mixed_mode);
            args.scalars = &params.scalars;

            const auto& intermediate_memories = instance.get_intermediates_memories();
            args.intermediates.insert(args.intermediates.end(),
                                      intermediate_memories.begin() + internal_buffers_offset,
                                      intermediate_memories.begin() + internal_buffers_offset + internal_buffers_count);

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << " (kd_idx=" << kd_idx << "): "
                                   << "gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }

            all_events.push_back(ev);
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, paged_attention_inst& instance) override {
        std::vector<event::ptr> res_events;
        const auto stage = get_paged_attention_stage(*instance.get_impl_params());
        const auto is_mixed_mode = stage == PagedAttentionStage::MIXED;

        execute_stage(events, instance, res_events, Stage::KV_CACHE_UPDATE, is_mixed_mode);

        std::vector<event::ptr> dep_events(res_events.begin(), res_events.end());
        if (stage == PagedAttentionStage::PREFILL) {
            execute_stage(dep_events, instance, res_events, Stage::SDPA, is_mixed_mode);
        } else if (stage == PagedAttentionStage::GENERATE || stage == PagedAttentionStage::MIXED) {
            execute_stage(dep_events, instance, res_events, Stage::PA_SDPA, is_mixed_mode);
        }

        return instance.get_network().get_stream().aggregate_events(res_events, res_events.size() > 1);
    }

    static int64_t get_aligned_seq_len(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, int64_t target_seq_len_block_size = 16) {
        // Since at prefill stage Q, K, V inputs may contain multiple sequences with arbitrary
        // target sequence lengths each (shape is [sequences_num * target_seq_len, num_heads * head_size]),
        // to apply blocking to the first dimension (target_seq_len of each sequence), we need to calculate aligned total
        // target sequence length for proper kernel dispatching
        // For instance, if input contains two sequences with 35 and 28 sequence lengths each,
        // the Q, K, V inputs at prefill stage will have shapes [35 + 28, num_heads * head_size]; considering kernel's
        // target_seq_len_block_size equals 16, we need to launch kernel instances for the following ranges:
        // [0, 15], [16, 31], [32, 34], [35, 50], [51, 62], so aligned target_seq_len_block_size should be 5 * 16 = 80,
        // and 5 kernels instances should be launched (for each range, some of them containing leftovers)
        //
        // In general, to obtain length for each sequence, we have to parse subsequence_begins input,
        // which contains begin and end indexes for each sequence (for above example it will contain three values: {0, 35, 63})
        // However, as long as kernel's target_seq_len_block_size matches with vLLM's block_size,
        // we can reuse block_indices_shape[0] size to determine total aligned sequences length size, avoiding
        // memory access at runtime, because vLLM internally uses similar logic to configure blocks for KV cache

        auto calculate_aligned_seq_len = [&]() {
            const auto& input_mem = impl_param.memory_deps;
            const auto subsequence_begins_input_idx = 6;
            const auto subsequence_begins_mem = input_mem.at(subsequence_begins_input_idx);
            mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *impl_param.strm);

            auto aligned_seq_len = 0;
            if (stage == PagedAttentionStage::MIXED) {
                const auto past_lens_idx = 5;
                const auto past_lens_mem = input_mem.at(past_lens_idx);
                mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

                for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                    auto past_len = past_lens_mem_lock[i];
                    auto seq_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];

                    // Since in MIXED execution mode the present KV-cache can be appended to the past KV-cache at any offset inside block,
                    // to ensure proper alignment and update_kv_cache kernel scheduling, we need to account for the number of unaligned tokens
                    // in the first block
                    // For example, if we need to store values in the following slots:
                    //
                    // block0: |O|O|O|O|O|O|O|O|O|O|O|O|U|U|U|U|
                    // block1: |U|U|U|U|U|U|U|U|U|U|U|U|U|U|U|U|
                    // block2: |U|U|U|U|U|U|E|E|E|E|E|E|E|E|E|E|
                    // Where O - occupied slots, U - currently beeing updated slots, E - empty slots
                    //
                    // We need to schedule 3 update_kv_cache operations:
                    // - For ranges of block0: [12-15]
                    // - For ranges of block1: [0-15]
                    // - For ranges of block2: [0-5]
                    //
                    // Therefore, consider an additional increment of aligned_seq_len to properly process all the blocks

                    auto occupied_slots_num = past_len % target_seq_len_block_size;
                    if (past_len != 0 && seq_length + occupied_slots_num > target_seq_len_block_size) {
                        aligned_seq_len += target_seq_len_block_size;
                        seq_length -= target_seq_len_block_size - occupied_slots_num;
                    }

                    aligned_seq_len += align_to(seq_length, target_seq_len_block_size);
                }
            } else {
                for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                    auto prompt_length = subsequence_begins_mem_lock[i + 1] - subsequence_begins_mem_lock[i];
                    aligned_seq_len += align_to(prompt_length, target_seq_len_block_size);
                }
            }

            return aligned_seq_len;
        };

        int64_t aligned_seq_len = 0;
        if (stage == PagedAttentionStage::PREFILL) {
            const auto desc = impl_param.typed_desc<paged_attention>();
            if (static_cast<int64_t>(paged_attention::block_size) == target_seq_len_block_size) {
                const auto block_indices_input_idx = 7;
                const auto& block_indices_ps = impl_param.get_input_layout(block_indices_input_idx).get_partial_shape();

                aligned_seq_len = block_indices_ps[0].get_length() * target_seq_len_block_size;
            } else {
                aligned_seq_len = calculate_aligned_seq_len();
            }
        } else {
            aligned_seq_len = calculate_aligned_seq_len();
        }

        return aligned_seq_len;
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        const auto desc = impl_param.typed_desc<paged_attention>();
        config.head_size = desc->head_size;
        config.heads_num = desc->heads_num;
        config.kv_heads_num = desc->kv_heads_num;
        config.has_alibi_input = desc->has_alibi;
        config.is_causal = true;
        config.is_paged_attention = true;
        config.paged_attention_block_size = static_cast<int64_t>(paged_attention::block_size);

        if (desc->scale_val.has_value()) {
            config.has_const_scale_val = true;
            config.scale_val = desc->scale_val.value();
        } else {
            config.has_const_scale_val = false;
        }

        if (desc->heads_num != desc->kv_heads_num) {
            config.broadcast_axis = 1;
            config.group_size = desc->heads_num / desc->kv_heads_num;
        }

        return config;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param,
                                                                             const PagedAttentionStage& stage,
                                                                             bool is_dynamic = false) {
        auto params = get_default_params<kv_cache_update_kernel_params_t>(impl_param, is_dynamic);

        const auto& key_layout = impl_param.get_input_layout(1);
        const auto& value_layout = impl_param.get_input_layout(2);
        const auto& key_cache_layout = impl_param.get_input_layout(3);
        const auto& value_cache_layout = impl_param.get_input_layout(4);
        const auto& past_lens_layout = impl_param.get_input_layout(5);
        const auto& block_indices_layout = impl_param.get_input_layout(7);
        const auto& block_indices_begins_layout = impl_param.get_input_layout(8);
        const auto& subsequence_begins_layout = impl_param.get_input_layout(6);

        const auto inputs_number = 6;
        const auto outputs_number = 2;
        params.inputs.resize(inputs_number);
        params.outputs.resize(outputs_number);
        params.inputs[0] = convert_data_tensor(key_layout);
        params.inputs[1] = convert_data_tensor(value_layout);
        params.inputs[2] = convert_data_tensor(past_lens_layout);
        params.inputs[3] = convert_data_tensor(block_indices_layout);
        params.inputs[4] = convert_data_tensor(block_indices_begins_layout);
        params.inputs[5] = convert_data_tensor(subsequence_begins_layout);
        params.outputs[0] = convert_data_tensor(key_cache_layout);
        params.outputs[1] = convert_data_tensor(value_cache_layout);

        params.conf = get_sdpa_configuration(impl_param);

        params.is_prefill = stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::MIXED;

        if ((stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::MIXED) && !is_dynamic)
            params.conf.paged_attention_aligned_seq_len = get_aligned_seq_len(impl_param, stage);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(1)},
            {1, in_offsets_map.at(2)},
            {2, in_offsets_map.at(5)},
            {3, in_offsets_map.at(7)},
            {4, in_offsets_map.at(8)},
            {5, in_offsets_map.at(6)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
            {1, in_offsets_map.at(4)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static sdpa_kernel_params_t get_sdpa_kernel_params(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, bool is_dynamic = false) {
        const auto desc = impl_param.typed_desc<paged_attention>();
        auto params = get_default_params<sdpa_kernel_params_t>(impl_param, is_dynamic);

        const auto& query_layout = impl_param.get_input_layout(0);
        const auto& key_layout = impl_param.get_input_layout(1);
        const auto& value_layout = impl_param.get_input_layout(2);
        const auto& subsequence_begins_layout = impl_param.get_input_layout(6);
        const auto& scale_layout = impl_param.get_input_layout(9);
        const auto& alibi_layout = impl_param.get_input_layout(11);
        const auto has_alibi = alibi_layout.count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();

        auto inputs_number = 4;
        if (has_scale_input)
            inputs_number++;

        if (has_alibi)
            inputs_number++;

        auto input_idx = 0;
        params.inputs.resize(inputs_number);
        params.inputs[input_idx++] = convert_data_tensor(query_layout);
        params.inputs[input_idx++] = convert_data_tensor(key_layout);
        params.inputs[input_idx++] = convert_data_tensor(value_layout);
        params.inputs[input_idx++] = convert_data_tensor(subsequence_begins_layout);

        if (has_scale_input)
            params.inputs[input_idx++] = convert_data_tensor(scale_layout);

        if (has_alibi)
            params.inputs[input_idx++] = convert_data_tensor(alibi_layout);

        params.conf = get_sdpa_configuration(impl_param);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(1)},
            {2, in_offsets_map.at(2)},
            {3, in_offsets_map.at(6)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        input_idx = 4;
        if (has_scale_input)
            in_tensor_to_offset_map.insert({input_idx++, in_offsets_map.at(9)});

        if (has_alibi)
            in_tensor_to_offset_map.insert({input_idx++, in_offsets_map.at(11)});

        if ((stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::MIXED) && !is_dynamic)
            params.conf.paged_attention_aligned_seq_len = get_aligned_seq_len(impl_param, stage);

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static pa_sdpa_kernel_params_t get_pa_sdpa_params(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, bool is_dynamic = false) {
        const auto desc = impl_param.typed_desc<paged_attention>();
        auto params = get_default_params<pa_sdpa_kernel_params_t>(impl_param, is_dynamic);

        const auto& query_layout = impl_param.get_input_layout(0);
        const auto& key_cache_layout = impl_param.get_input_layout(3);
        const auto& value_cache_layout = impl_param.get_input_layout(4);
        const auto& past_lens_layout = impl_param.get_input_layout(5);
        const auto& block_indices_layout = impl_param.get_input_layout(7);
        const auto& block_indices_begins_layout = impl_param.get_input_layout(8);
        const auto& subsequence_begins_layout = impl_param.get_input_layout(6);
        const auto& scale_layout = impl_param.get_input_layout(9);
        const auto& alibi_layout = impl_param.get_input_layout(11);
        const auto has_alibi = alibi_layout.count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();

        auto inputs_number = 7;
        if (has_scale_input)
            inputs_number++;

        if (has_alibi)
            inputs_number++;

        auto input_idx = 0;
        params.inputs.resize(inputs_number);
        params.inputs[input_idx++] = convert_data_tensor(query_layout);
        params.inputs[input_idx++] = convert_data_tensor(key_cache_layout);
        params.inputs[input_idx++] = convert_data_tensor(value_cache_layout);
        params.inputs[input_idx++] = convert_data_tensor(past_lens_layout);
        params.inputs[input_idx++] = convert_data_tensor(block_indices_layout);
        params.inputs[input_idx++] = convert_data_tensor(block_indices_begins_layout);
        params.inputs[input_idx++] = convert_data_tensor(subsequence_begins_layout);
        params.conf = get_sdpa_configuration(impl_param);

        if (has_scale_input)
            params.inputs[input_idx++] = convert_data_tensor(scale_layout);

        if (has_alibi)
            params.inputs[input_idx++] = convert_data_tensor(alibi_layout);

        params.multi_tokens_mode = stage == PagedAttentionStage::MIXED;

        if ((stage == PagedAttentionStage::GENERATE || stage == PagedAttentionStage::MIXED) && !is_dynamic) {
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
            {4, in_offsets_map.at(7)},
            {5, in_offsets_map.at(8)},
            {6, in_offsets_map.at(6)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        input_idx = 7;
        if (has_scale_input)
            in_tensor_to_offset_map.insert({input_idx++, in_offsets_map.at(9)});

        if (has_alibi)
            in_tensor_to_offset_map.insert({input_idx++, in_offsets_map.at(11)});

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        const auto stage = get_paged_attention_stage(impl_param);

        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, stage, impl_param.is_dynamic());
        (_kernels_data[Stage::KV_CACHE_UPDATE].update_dispatch_data_func)(kv_cache_update_kernel_params, _kernels_data[Stage::KV_CACHE_UPDATE]);

        if (stage == PagedAttentionStage::PREFILL) {
            auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, stage, impl_param.is_dynamic());
            (_kernels_data[Stage::SDPA].update_dispatch_data_func)(sdpa_kernel_params, _kernels_data[Stage::SDPA]);
        } else if (stage == PagedAttentionStage::GENERATE || stage == PagedAttentionStage::MIXED) {
            auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, stage, impl_param.is_dynamic());
            (_kernels_data[Stage::PA_SDPA].update_dispatch_data_func)(pa_sdpa_kernel_params, _kernels_data[Stage::PA_SDPA]);
        }
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        const auto stage = PagedAttentionStage::UNKNOWN;

        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, stage, impl_param.is_dynamic());
        auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(kv_cache_update_kernel_selector.get_best_kernel(kv_cache_update_kernel_params));

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, stage, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params));

        auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, stage, impl_param.is_dynamic());
        auto& pa_sdpa_kernel_selector = pa_sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(pa_sdpa_kernel_selector.get_best_kernel(pa_sdpa_kernel_params));

        return cldnn::make_unique<paged_attention_impl>(kernels_data);
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
