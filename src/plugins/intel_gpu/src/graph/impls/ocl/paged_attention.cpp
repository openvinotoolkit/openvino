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
#include "sdpa/pa_kv_cache_rotate_kernel_ref.h"
#include "sdpa/pa_kv_cache_update_kernel_ref.h"
#include "sdpa/pa_sdpa_kernel_opt.h"
#include "sdpa/sdpa_kernel_micro.h"

namespace cldnn {
namespace ocl {
using PagedAttentionStage = kernel_selector::PagedAttentionStage;

struct paged_attention_impl : multi_stage_primitive<paged_attention> {
    using parent = multi_stage_primitive<paged_attention>;
    using parent::parent;

    using sdpa_kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using sdpa_kernel_params_t = kernel_selector::sdpa_params;

    using pa_sdpa_kernel_selector_t = kernel_selector::pa_sdpa_kernel_selector;
    using pa_sdpa_kernel_params_t = kernel_selector::pa_sdpa_params;

    using kv_cache_rotate_kernel_selector_t = kernel_selector::kv_cache_rotate_kernel_selector;
    using kv_cache_rotate_kernel_params_t = kernel_selector::kv_cache_rotate_params;

    using kv_cache_update_kernel_selector_t = kernel_selector::kv_cache_update_kernel_selector;
    using kv_cache_update_kernel_params_t = kernel_selector::kv_cache_update_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::paged_attention_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<paged_attention_impl>(*this);
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
        KV_CACHE_ROTATE,
    };

    PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param) const {
        const auto& query_shape = impl_param.get_input_layout(0).get_partial_shape();
        const auto& past_lens_shape = impl_param.get_input_layout(5).get_partial_shape();

        if (query_shape.is_static() && past_lens_shape.is_static()) {
            if (query_shape[0].get_length() == past_lens_shape[0].get_length()) {
                return PagedAttentionStage::GENERATE;
            }

            const auto past_lens_idx = 5;
            const auto& memory_deps = impl_param.memory_deps;
            const auto past_lens_mem = memory_deps.at(past_lens_idx);
            mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

            const auto past_lens_size = past_lens_mem_lock.size();
            for (size_t i = 0; i < past_lens_size; i++) {
                if (past_lens_mem_lock[i] != 0) {
                    return PagedAttentionStage::MIXED;
                }
            }

            return PagedAttentionStage::PREFILL;
        }

        return PagedAttentionStage::UNKNOWN;
    }

    bool requires_update(primitive_inst& inst, const kernel_impl_params& impl_params) const override {
        const auto stage = get_paged_attention_stage(impl_params);

        // In case of MIXED mode execution Paged Attention may require dispatch data update and internal
        // buffers reallocation even if the input shapes haven't been changed. Therefore, check the current execution
        // mode and update parameters if needed
        return stage == PagedAttentionStage::MIXED;
    }

    size_t get_query_block_size(const PagedAttentionStage& stage) const {
        const auto default_block_size = 16;

        if (stage == PagedAttentionStage::PREFILL) {
#ifdef ENABLE_ONEDNN_FOR_GPU
            if (use_micro_sdpa)
                return kernel_selector::SDPAKernelMicro::GetTileQSize(_kernels_data[Stage::SDPA]);
#endif
            return default_block_size;
        } else {
            return default_block_size;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> make_data(&has_scores_output, sizeof(bool));
        ib >> make_data(&has_rotated_blocks, sizeof(bool));
        ib >> make_data(&use_micro_sdpa, sizeof(bool));
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

            if (has_rotated_blocks) {
                auto& kv_cache_rotate_kernel_selector = kv_cache_rotate_kernel_selector_t::Instance();
                auto kv_cache_rotate_kernel_impl = kv_cache_rotate_kernel_selector.GetImplementation(_kernels_data[Stage::KV_CACHE_ROTATE].kernelName);
                kv_cache_rotate_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[Stage::KV_CACHE_ROTATE]);
            }
        }
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << make_data(&has_scores_output, sizeof(bool));
        ob << make_data(&has_rotated_blocks, sizeof(bool));
        ob << make_data(&use_micro_sdpa, sizeof(bool));
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        /*
        * Internal buffers allocation owners and users:
        * +--------------------------------------+--------------------+--------------------+
        * | Stage                                | Allocates & uses   | Reuses             |
        * +--------------------------------------+--------------------+--------------------+
        * | KV_CACHE_UPDATE                      | [0, 1, 2]          |                    |
        * +--------------------------------------+--------------------+--------------------+
        * | SDPA (1st token)                     |                    | [0, 1, 2]          |
        * +--------------------------------------+--------------------+--------------------+
        * | PA_SDPA (2nd+ token)                 | [5, 6, 7]          |                    |
        * +--------------------------------------+--------------------+--------------------+
        * | PA_SDPA (mixed mode)                 | [5, 6, 7, 8]       |                    |
        * +--------------------------------------+--------------------+--------------------+
        * | SDPA (1st token) + scores output     |                    | [0, 1, 2, 3, 4]    |
        * +--------------------------------------+--------------------+--------------------+
        * | PA_SDPA (2nd+ token) + scores output | [3, 4, 5, 6, 7]    |                    |
        * +--------------------------------------+--------------------+--------------------+
        * | PA_SDPA (mixed mode) + scores output | [3, 4, 5, 6, 7, 8] |                    |
        * +--------------------------------------+--------------------+--------------------+
        * | SDPA (1st token, micro-kernel)       | [last (8/9)]       |                    |
        * +--------------------------------------+--------------------+--------------------+
        *
        * Description:
        * 0, 1, 2 - Buffers used for proper blocks distribution for kv_cache_update and
        *           sdpa_opt (1st token calculation) block configuration over target_seq_len dimension.
        *           Filled in paged_attention_inst::on_execute() call.
        * 3, 4    - Optional buffers used for PA scores output calculation, storing intermediate
        *           softmax values by partitions (filled in PA/SDPA kernels) and sequence length offsets
        *           for each subsequence (filled in paged_attention_inst::on_execute() call).
        * 5, 6, 7 - Used for 2nd+ PA calculation (for softmax exp_sums, max_logits, and intermediate output).
        *           Filled in PA/SDPA kernels.
        * 8       - Optional buffer used for mixed PA execution mode, mapping gws idx to subsequence id.
        *           Filled in paged_attention_inst::on_execute() call.
        * last    - Used for defining query block index for the currently processing subsequence and mapping
        *           gws index to subsequence idx. Values stored in pairs like:
        *           [block_idx0, subsequence_idx0, block_idx1, subsequence_idx0, ..., block_idx0, subsequence_idx1].
        *           Filled in paged_attention_inst::on_execute() call for sdpa-micro kernel only.
        */

        auto add_internal_buffers = [](std::vector<BufferDescriptor>& internal_buffers,
                                       const kernel_selector::KernelData& kd) {
            for (const auto& buffer_desc : kd.internalBuffers) {
                internal_buffers.emplace_back(buffer_desc.byte_count, ov::element::u8, buffer_desc.lockable);
            }
        };

        std::vector<BufferDescriptor> internal_buffers;
        add_internal_buffers(internal_buffers, _kernels_data[Stage::KV_CACHE_UPDATE]);
        add_internal_buffers(internal_buffers, _kernels_data[Stage::PA_SDPA]);

        if (use_micro_sdpa)
            add_internal_buffers(internal_buffers, _kernels_data[Stage::SDPA]);

        return internal_buffers;
    }

    kernel_arguments_data get_arguments(const paged_attention_inst& instance, size_t stage, size_t kernel_idx, bool is_mixed_mode) const {
        const auto desc = instance.get_node().as<paged_attention>().get_primitive();

        kernel_arguments_data args;
        if (stage == Stage::KV_CACHE_UPDATE || stage == Stage::SDPA || stage == Stage::KV_CACHE_ROTATE)
            args.shape_info = instance.shape_info_memory_ptr();

        if (stage == Stage::KV_CACHE_ROTATE) {
            args.inputs = {  instance.rotated_block_indices_memory_ptr(),
                             instance.rotation_deltas_memory_ptr(),
                             instance.rotation_trig_lut_memory_ptr() };

            args.outputs = { instance.key_cache_memory_ptr() };
        } else if (stage == Stage::KV_CACHE_UPDATE) {
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
            if (kernel_idx == 0 || kernel_idx == 1 || kernel_idx == 2) {
                // 2nd+ token calculation or mixed stage tokens calculation
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
            } else if (kernel_idx == 3 || kernel_idx == 4) {
                // Finalization kernel or mixed stage finalization kernel
                args.inputs = { instance.past_lens_memory_ptr() };

                if (is_mixed_mode) {
                    // Multi tokens kernel version has additional subsequence_begins_memory memory
                    // dependency
                    args.inputs.push_back(instance.subsequence_begins_memory_ptr());
                }
                if (desc->has_rotated_blocks) {
                    args.inputs.push_back(instance.rotated_block_indices_memory_ptr());
                    args.inputs.push_back(instance.rotation_deltas_memory_ptr());
                    args.inputs.push_back(instance.rotation_trig_lut_memory_ptr());
                }
            } else if (kernel_idx == 5) {
                // Output scores calculation kernel
                args.inputs = { instance.past_lens_memory_ptr(),
                                instance.subsequence_begins_memory_ptr() };
            }

            args.outputs = { instance.output_memory_ptr(0) };

            if (kernel_idx == 5) {
                args.outputs.push_back(instance.output_memory_ptr(1));
            }
        }

        return args;
    }

    void prepare_internal_buffers(paged_attention_inst& instance, const PagedAttentionStage& stage) {
        const auto& desc = instance.get_impl_params()->typed_desc<paged_attention>();
        const bool has_scores_output = desc->has_scores_output();

        if ((stage == PagedAttentionStage::UNKNOWN) ||
            (stage == PagedAttentionStage::GENERATE && !has_scores_output))
            return;

        auto& stream = instance.get_network().get_stream();
        const auto past_lens_mem = instance.past_lens_memory_ptr();
        const auto subsequence_begins_mem = instance.subsequence_begins_memory_ptr();
        auto intermediates_memories = instance.get_intermediates_memories();
        mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, stream);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, stream);
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> subsequence_offsets_lock = nullptr;

        if (has_scores_output) {
            const size_t subsequence_offsets_idx = 4;

            OPENVINO_ASSERT(intermediates_memories.size() > subsequence_offsets_idx,
                            "[GPU] Unexpected number of intermediates buffers for Paged Attention for scores output calculation");

            auto subsequence_offsets_mem = intermediates_memories[subsequence_offsets_idx];
            subsequence_offsets_lock.reset(new mem_lock<int32_t, mem_lock_type::write>(subsequence_offsets_mem, stream));
        }

        if (stage == PagedAttentionStage::GENERATE) {
            // For the generate stage it's not necessary to configure any other intermediate
            // buffers. Simply calculate the offsets and exit
            size_t subsequence_offsets_acc = 0;
            for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
                const auto past_len = past_lens_mem_lock[i];
                const auto seq_start = subsequence_begins_mem_lock[i];
                const auto seq_end = subsequence_begins_mem_lock[i + 1];
                const auto seq_length = seq_end - seq_start;

                if (subsequence_offsets_lock) {
                    subsequence_offsets_lock->operator[](i) = static_cast<int32_t>(subsequence_offsets_acc);
                    subsequence_offsets_acc += seq_length + past_len;
                }
            }

            return;
        }

        OPENVINO_ASSERT(intermediates_memories.size() >= 3, "Unexpected number of intermediates buffers for Paged Attention at prefill stage");

        const auto blocks_indexes_start_idx = 0;
        const auto blocks_indexes_end_idx = 1;
        const auto blocked_gws_subseq_mapping_idx = 2;

        auto blocks_indexes_start_mem = intermediates_memories[blocks_indexes_start_idx];
        auto blocks_indexes_end_mem = intermediates_memories[blocks_indexes_end_idx];
        auto blocked_gws_subseq_mapping_mem = intermediates_memories[blocked_gws_subseq_mapping_idx];

        OPENVINO_ASSERT(subsequence_begins_mem->get_layout().data_type == data_types::i32);

        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_start_lock(blocks_indexes_start_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_end_lock(blocks_indexes_end_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocked_gws_subseq_mapping_mem_lock(blocked_gws_subseq_mapping_mem, stream);
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> sequential_gws_subseq_mapping_lock = nullptr;
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> micro_sdpa_block_starts_and_gws_mapping_lock = nullptr;

        if (stage == PagedAttentionStage::MIXED) {
            const size_t sequential_gws_subseq_mapping_idx = has_scores_output ? 8 : 6;

            OPENVINO_ASSERT(intermediates_memories.size() > sequential_gws_subseq_mapping_idx,
                            "[GPU] Unexpected number of intermediates buffers for Paged Attention for mixed stage");

            auto sequential_gws_subseq_mapping_mem = intermediates_memories[sequential_gws_subseq_mapping_idx];
            sequential_gws_subseq_mapping_lock.reset(new mem_lock<int32_t, mem_lock_type::write>(sequential_gws_subseq_mapping_mem, stream));
        }

        if (stage == PagedAttentionStage::PREFILL && use_micro_sdpa) {
            const auto memory_idx = intermediates_memories.size() - 1;

            auto memory = intermediates_memories[memory_idx];
            micro_sdpa_block_starts_and_gws_mapping_lock.reset(new mem_lock<int32_t, mem_lock_type::write>(memory, stream));
        }

        size_t index = 0;
        size_t micro_sdpa_index = 0;
        size_t subsequence_offsets_acc = 0;
        size_t query_block_size = get_query_block_size(stage);
        const auto pa_block_size = static_cast<int>(paged_attention::block_size);
        for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
            const auto past_len = past_lens_mem_lock[i];
            const auto seq_start = subsequence_begins_mem_lock[i];
            const auto seq_end = subsequence_begins_mem_lock[i + 1];
            const auto seq_length = seq_end - seq_start;

            int32_t j = 0;
            if (past_len != 0) {
                auto block_start_pos = seq_start;
                auto empty_slots = pa_block_size - (past_len % pa_block_size);
                auto block_end_pos = seq_start + std::min(empty_slots, seq_length);

                blocks_indexes_start_lock[index] = block_start_pos;
                blocks_indexes_end_lock[index] = block_end_pos;
                blocked_gws_subseq_mapping_mem_lock[index] = static_cast<int32_t>(i);

                index++;

                auto added_slots = block_end_pos - block_start_pos;
                j += added_slots;
            }

            for (; j < seq_length; j += pa_block_size) {
                auto block_start_pos = subsequence_begins_mem_lock[i] + j;
                auto block_end_pos = std::min(block_start_pos + pa_block_size, seq_end);

                blocks_indexes_start_lock[index] = block_start_pos;
                blocks_indexes_end_lock[index] = block_end_pos;
                blocked_gws_subseq_mapping_mem_lock[index] = static_cast<int32_t>(i);

                index++;
            }

            if (micro_sdpa_block_starts_and_gws_mapping_lock) {
                const auto block_size = static_cast<int>(query_block_size);
                for (int32_t j = 0; j < seq_length; j += block_size) {
                    auto block_start_pos = subsequence_begins_mem_lock[i] + j;

                    micro_sdpa_block_starts_and_gws_mapping_lock->operator[](micro_sdpa_index++) = block_start_pos;
                    micro_sdpa_block_starts_and_gws_mapping_lock->operator[](micro_sdpa_index++) = static_cast<int32_t>(i);
                }
            }

            if (stage == PagedAttentionStage::MIXED) {
                for (int32_t idx = seq_start; idx < seq_end; idx++) {
                    sequential_gws_subseq_mapping_lock->operator[](idx) = static_cast<int32_t>(i);
                }
            }

            if (subsequence_offsets_lock) {
                subsequence_offsets_lock->operator[](i) = static_cast<int32_t>(subsequence_offsets_acc);
                subsequence_offsets_acc += seq_length + past_len;
            }
        }
    }

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
            internal_buffers_offset = _kernels_data[Stage::KV_CACHE_UPDATE].internalBuffers.size();
            internal_buffers_count = _kernels_data[Stage::PA_SDPA].internalBuffers.size();
        } else if (stage == Stage::KV_CACHE_UPDATE) {
            internal_buffers_count = _kernels_data[Stage::KV_CACHE_UPDATE].internalBuffers.size();
        } else if (stage == Stage::SDPA) {
            internal_buffers_count = _kernels_data[Stage::KV_CACHE_UPDATE].internalBuffers.size();

            const auto desc = instance.get_node().as<paged_attention>().get_primitive();
            if (desc->has_scores_output()) {
                // Add intermediate buffers for PagedAttention scores calculation:
                // softmax_results, subsequence_offsets, exp_sums, max_logits, tmp_out
                internal_buffers_count += 5;
            }
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

            if (use_micro_sdpa && stage == Stage::SDPA) {
                args.intermediates.push_back(intermediate_memories.back());
            }

            GPU_DEBUG_TRACE_DETAIL << "Execute stage=" << stage << " kernel=" << kd_idx << " " << _kernels_data[stage].kernelName << " start_offset="
                                   << internal_buffers_offset << " count=" << internal_buffers_count << "\n";

            GPU_DEBUG_TRACE_DETAIL << "Configured kernel arguments:\n";
            for (size_t i = 0; i < _kernels_data[stage].kernels[kd_idx].params.arguments.size(); i++) {
                GPU_DEBUG_TRACE_DETAIL << "\t" << i << ": type=" << static_cast<int>(_kernels_data[stage].kernels[kd_idx].params.arguments[i].t) << " "
                                       << "index=" << _kernels_data[stage].kernels[kd_idx].params.arguments[i].index << "\n";
            }

            GPU_DEBUG_TRACE_DETAIL << "Memory buffers:"
                                   << "shape_info=" << args.shape_info << " "
                                   << "inputs=" << args.inputs.size() << " "
                                   << "outputs=" << args.outputs.size() << " "
                                   << "intermediates=" << args.intermediates.size() << " "
                                   << "weights=" << args.weights << " "
                                   << "scalars=" << (args.scalars ? args.scalars->size() : 0) << "\n";

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
        const auto stage = get_paged_attention_stage(*instance.get_impl_params());
        const auto is_mixed_mode = stage == PagedAttentionStage::MIXED;

        prepare_internal_buffers(instance, stage);

        std::vector<event::ptr> res_events;
        std::vector<event::ptr> dep_events = events;
        if (has_rotated_blocks && !_kernels_data[Stage::KV_CACHE_ROTATE].kernels[0].skip_execution) {
            execute_stage(dep_events, instance, res_events, Stage::KV_CACHE_ROTATE, is_mixed_mode);
            dep_events = res_events;
        }

        execute_stage(dep_events, instance, res_events, Stage::KV_CACHE_UPDATE, is_mixed_mode);

        if (stage == PagedAttentionStage::PREFILL) {
            dep_events = res_events;
            execute_stage(dep_events, instance, res_events, Stage::SDPA, is_mixed_mode);
        }

        if (stage == PagedAttentionStage::GENERATE || stage == PagedAttentionStage::MIXED || has_scores_output) {
            dep_events = res_events;
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

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param, bool is_dynamic = true) {
        kernel_selector::sdpa_configuration config;

        const auto desc = impl_param.typed_desc<paged_attention>();
        config.head_size = desc->head_size;
        config.heads_num = desc->heads_num;
        config.kv_heads_num = desc->kv_heads_num;
        config.has_alibi_input = desc->has_alibi;
        config.is_causal = true;
        config.is_paged_attention = true;
        config.paged_attention_block_size = static_cast<int64_t>(paged_attention::block_size);
        config.paged_attention_sliding_window = desc->sliding_window;

        if (desc->scale_val.has_value()) {
            config.has_const_scale_val = true;
            config.scale_val = desc->scale_val.value();
        } else {
            config.has_const_scale_val = false;
        }

        config.has_rotated_blocks = desc->has_rotated_blocks;

        if (desc->heads_num != desc->kv_heads_num) {
            config.broadcast_axis = 1;
            config.kv_group_size = desc->heads_num / desc->kv_heads_num;
        }

        if (desc->has_scores_output() && !is_dynamic) {
            const auto& input_mem = impl_param.memory_deps;
            const auto max_context_len = input_mem.at(12);
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *impl_param.strm);
            config.paged_attention_max_len = max_context_len_mem_lock[0];
        }

        if (data_type_traits::is_i8_u8(impl_param.get_input_layout(3).data_type)) {
            config.is_kv_compressed = true;
            config.use_asymmetric_quantization = true;
        }

        return config;
    }

    static kv_cache_rotate_kernel_params_t get_kv_cache_rotate_kernel_params(const kernel_impl_params& impl_param,
                                                                             const kernel_selector::MultiDataTensor& input_tensors,
                                                                             bool is_dynamic = false) {
        auto params = get_default_params<kv_cache_rotate_kernel_params_t>(impl_param, is_dynamic);

        const auto& key_cache_tensor = input_tensors[3];
        const auto& rotated_block_indices_tensor = input_tensors[13];
        const auto& rotation_deltas_tensor = input_tensors[14];
        const auto& rotation_trig_lut_tensor = input_tensors[15];

        const auto inputs_number = 3;
        const auto outputs_number = 1;
        params.inputs.resize(inputs_number);
        params.outputs.resize(outputs_number);
        params.inputs[0] = rotated_block_indices_tensor;
        params.inputs[1] = rotation_deltas_tensor;
        params.inputs[2] = rotation_trig_lut_tensor;
        params.outputs[0] = key_cache_tensor;

        params.original_cache_dt = to_data_type(impl_param.get_input_layout(1).data_type);
        params.conf = get_sdpa_configuration(impl_param, is_dynamic);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(13)},
            {1, in_offsets_map.at(14)},
            {2, in_offsets_map.at(15)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static kv_cache_update_kernel_params_t get_kv_cache_update_kernel_params(const kernel_impl_params& impl_param,
                                                                             const PagedAttentionStage& stage,
                                                                             const kernel_selector::MultiDataTensor& input_tensors,
                                                                             bool is_dynamic = false) {
        auto params = get_default_params<kv_cache_update_kernel_params_t>(impl_param, is_dynamic);

        const auto& key_tensor = input_tensors[1];
        const auto& value_tensor = input_tensors[2];
        const auto& key_cache_tensor = input_tensors[3];
        const auto& value_cache_tensor = input_tensors[4];
        const auto& past_lens_tensor = input_tensors[5];
        const auto& block_indices_tensor = input_tensors[7];
        const auto& block_indices_begins_tensor = input_tensors[8];
        const auto& subsequence_begins_tensor = input_tensors[6];

        const auto inputs_number = 6;
        const auto outputs_number = 2;
        params.inputs.resize(inputs_number);
        params.outputs.resize(outputs_number);
        params.inputs[0] = key_tensor;
        params.inputs[1] = value_tensor;
        params.inputs[2] = past_lens_tensor;
        params.inputs[3] = block_indices_tensor;
        params.inputs[4] = block_indices_begins_tensor;
        params.inputs[5] = subsequence_begins_tensor;
        params.outputs[0] = key_cache_tensor;
        params.outputs[1] = value_cache_tensor;

        params.conf = get_sdpa_configuration(impl_param, is_dynamic);

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

    static sdpa_kernel_params_t get_sdpa_kernel_params(const kernel_impl_params& impl_param,
                                                       const PagedAttentionStage& stage,
                                                       const kernel_selector::MultiDataTensor& input_tensors,
                                                       int64_t query_block_size,
                                                       bool is_dynamic) {
        const auto desc = impl_param.typed_desc<paged_attention>();
        auto params = get_default_params<sdpa_kernel_params_t>(impl_param, is_dynamic);

        const auto& query_tensor = input_tensors[0];
        const auto& key_tensor = input_tensors[1];
        const auto& value_tensor = input_tensors[2];
        const auto& subsequence_begins_tensor = input_tensors[6];
        const auto& scale_tensor = input_tensors[9];
        const auto& alibi_tensor = input_tensors[11];

        const auto has_alibi = impl_param.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = desc->has_scores_output();

        auto inputs_number = 4;
        if (has_scale_input)
            inputs_number++;

        if (has_alibi)
            inputs_number++;

        auto input_idx = 0;
        params.inputs.resize(inputs_number);
        params.inputs[input_idx++] = query_tensor;
        params.inputs[input_idx++] = key_tensor;
        params.inputs[input_idx++] = value_tensor;
        params.inputs[input_idx++] = subsequence_begins_tensor;

        if (has_scale_input)
            params.inputs[input_idx++] = scale_tensor;

        if (has_alibi)
            params.inputs[input_idx++] = alibi_tensor;

        if (has_scores_output) {
            params.outputs.resize(2);
            params.outputs[1] = convert_data_tensor(impl_param.get_output_layout(1));
        }

        params.conf = get_sdpa_configuration(impl_param, is_dynamic);

        // Currently, for the processing of the 1st token, plain SDPA kernels are used, which expect
        // uncompressed plain QKV inputs. Therefore, set is_kv_compressed=false
        params.conf.is_kv_compressed = false;
        params.conf.use_asymmetric_quantization = false;

        const std::vector<int64_t> default_order = {0, 1, 2, 3};
        params.input0_order = default_order;
        params.input1_order = default_order;
        params.input2_order = default_order;
        params.output_order = default_order;

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
            params.conf.paged_attention_aligned_seq_len = get_aligned_seq_len(impl_param, stage, query_block_size);

        if (has_scores_output)
            out_tensor_to_offset_map.insert({1, out_offsets_map.at(1)});

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static pa_sdpa_kernel_params_t get_pa_sdpa_params(const kernel_impl_params& impl_param,
                                                      const PagedAttentionStage& stage,
                                                      const kernel_selector::MultiDataTensor& input_tensors,
                                                      bool is_dynamic = false) {
        const auto desc = impl_param.typed_desc<paged_attention>();
        auto params = get_default_params<pa_sdpa_kernel_params_t>(impl_param, is_dynamic);

        const auto& query_tensor = input_tensors[0];
        const auto& key_cache_tensor = input_tensors[3];
        const auto& value_cache_tensor = input_tensors[4];
        const auto& past_lens_tensor = input_tensors[5];
        const auto& block_indices_tensor = input_tensors[7];
        const auto& block_indices_begins_tensor = input_tensors[8];
        const auto& subsequence_begins_tensor = input_tensors[6];
        const auto& scale_tensor = input_tensors[9];
        const auto& alibi_tensor = input_tensors[11];

        const auto has_alibi = impl_param.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = desc->has_scores_output();

        auto inputs_number = 7;
        if (has_scale_input)
            inputs_number++;

        if (has_alibi)
            inputs_number++;

        auto input_idx = 0;
        params.inputs.resize(inputs_number);
        params.inputs[input_idx++] = query_tensor;
        params.inputs[input_idx++] = key_cache_tensor;
        params.inputs[input_idx++] = value_cache_tensor;
        params.inputs[input_idx++] = past_lens_tensor;
        params.inputs[input_idx++] = block_indices_tensor;
        params.inputs[input_idx++] = block_indices_begins_tensor;
        params.inputs[input_idx++] = subsequence_begins_tensor;

        params.conf = get_sdpa_configuration(impl_param, is_dynamic);

        if (has_scale_input)
            params.inputs[input_idx++] = scale_tensor;

        if (has_alibi)
            params.inputs[input_idx++] = alibi_tensor;

        if (has_scores_output) {
            params.outputs.resize(2);
            params.outputs[1] = convert_data_tensor(impl_param.get_output_layout(1));
        }

        params.stage = stage;

        if (!has_scores_output && !is_dynamic) {
            const auto& input_mem = impl_param.memory_deps;
            const auto max_context_len = input_mem.at(12);
            mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *impl_param.strm);
            params.conf.paged_attention_max_len = max_context_len_mem_lock[0];
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

        if (has_scores_output)
            out_tensor_to_offset_map.insert({1, out_offsets_map.at(1)});

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        const auto stage = get_paged_attention_stage(impl_param);

        kernel_selector::MultiDataTensor input_tensors;
        for (const auto& input_layout : impl_param.input_layouts)
            input_tensors.emplace_back(convert_data_tensor(input_layout));

        if (has_rotated_blocks) {
            auto kv_cache_rotate_kernel_params = get_kv_cache_rotate_kernel_params(impl_param, input_tensors, impl_param.is_dynamic());
            (_kernels_data[Stage::KV_CACHE_ROTATE].update_dispatch_data_func)(kv_cache_rotate_kernel_params, _kernels_data[Stage::KV_CACHE_ROTATE]);
        }

        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, stage, input_tensors, impl_param.is_dynamic());
        (_kernels_data[Stage::KV_CACHE_UPDATE].update_dispatch_data_func)(kv_cache_update_kernel_params, _kernels_data[Stage::KV_CACHE_UPDATE]);

        if (stage == PagedAttentionStage::PREFILL) {
            auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, stage, input_tensors, get_query_block_size(stage), impl_param.is_dynamic());
            (_kernels_data[Stage::SDPA].update_dispatch_data_func)(sdpa_kernel_params, _kernels_data[Stage::SDPA]);
        }

        if (stage == PagedAttentionStage::GENERATE || stage == PagedAttentionStage::MIXED || has_scores_output) {
            auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, stage, input_tensors, impl_param.is_dynamic());
            (_kernels_data[Stage::PA_SDPA].update_dispatch_data_func)(pa_sdpa_kernel_params, _kernels_data[Stage::PA_SDPA]);
        }
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        const auto stage = PagedAttentionStage::UNKNOWN;

        kernel_selector::MultiDataTensor input_tensors;
        for (const auto& input_layout : impl_param.input_layouts)
            input_tensors.emplace_back(convert_data_tensor(input_layout));

        const auto& desc = impl_param.typed_desc<paged_attention>();
        auto kv_cache_update_kernel_params = get_kv_cache_update_kernel_params(impl_param, stage, input_tensors, impl_param.is_dynamic());
        auto& kv_cache_update_kernel_selector = kv_cache_update_kernel_selector_t::Instance();
        kernels_data.push_back(kv_cache_update_kernel_selector.get_best_kernel(kv_cache_update_kernel_params));

        auto sdpa_kernel_params = get_sdpa_kernel_params(impl_param, stage, input_tensors, 0, impl_param.is_dynamic());
        auto& sdpa_kernel_selector = sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(sdpa_kernel_selector.get_best_kernel(sdpa_kernel_params));

        auto pa_sdpa_kernel_params = get_pa_sdpa_params(impl_param, stage, input_tensors, impl_param.is_dynamic());
        auto& pa_sdpa_kernel_selector = pa_sdpa_kernel_selector_t::Instance();
        kernels_data.push_back(pa_sdpa_kernel_selector.get_best_kernel(pa_sdpa_kernel_params));

        if (desc->has_rotated_blocks) {
            auto kv_cache_rotate_kernel_params = get_kv_cache_rotate_kernel_params(impl_param, input_tensors, impl_param.is_dynamic());
            auto& kv_cache_rotate_kernel_selector = kv_cache_rotate_kernel_selector_t::Instance();
            kernels_data.push_back(kv_cache_rotate_kernel_selector.get_best_kernel(kv_cache_rotate_kernel_params));
        }

        auto impl = std::make_unique<paged_attention_impl>(kernels_data);
        impl->has_scores_output = desc->has_scores_output();
        impl->has_rotated_blocks = desc->has_rotated_blocks;

        if (!kernels_data[Stage::SDPA].kernels[0].micro_kernels.empty()) {
            impl->use_micro_sdpa = true;
        }

        return impl;
    }

private:
    bool has_scores_output = false;
    bool has_rotated_blocks = false;
    bool use_micro_sdpa = false;
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
