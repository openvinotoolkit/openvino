// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gen.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace ov;
using namespace ov::intel_gpu::ocl;
using namespace cldnn;
namespace {
constexpr size_t WG_SIZE = 16;
constexpr size_t reduce_split_step = 16;
}  // namespace

#define DEBUG_ENABLED 0

// This function returns the kv_step and kv_split_len based on the architecture.
// return {kv_step, kv_split_len}
inline std::pair<size_t, size_t> get_kv_split_size(size_t arch) {
    if (arch == 1) {
        return {8, 32};  // For Xe1
    } else if (arch == 2) {
        return {16, 32};  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for KV split size");
    return {0, 0};  // Fallback case, should not be reached
}

inline size_t get_q_step(size_t arch, bool is_single_token = false) {
    if (arch == 1) {
        return is_single_token ? 1 : 8;  // For Xe1
    } else if (arch == 2) {
        // For Xe2, q_step = CM_GRF_WIDTH / 32
        return is_single_token ? 1 : 16;  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for Q step");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_kv_len(const RuntimeParams& params, const PagedAttentionStage& stage) {
    if (stage == PagedAttentionStage::PREFILL) {
        auto key_shape = params.input_layouts[PagedAttentionInputIdx::KEY].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        return kv_len;
    } else {
        // key_cache shape = [block_num, head_num, block_size(128), head_size]
        auto key_cache_shape = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE].get_shape();
        const size_t kv_len = key_cache_shape[0] * key_cache_shape[2];
        return kv_len;
    }
    OPENVINO_ASSERT(false, "Unsupported PagedAttentionStage for get_kv_len");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_input_kv_len(const RuntimeParams& params) {
    auto key_shape = params.input_layouts[PagedAttentionInputIdx::KEY].get_shape();
    const size_t kv_len = key_shape[key_shape.size() - 2];
    return kv_len;
}

inline size_t get_aligned_kv_len(const size_t kv_len) {
    return (kv_len + PA_KV_CACHE_BLOCK_SIZE - 1) / PA_KV_CACHE_BLOCK_SIZE * PA_KV_CACHE_BLOCK_SIZE;
}

inline bool get_kv_compressed(const RuntimeParams& params) {
    auto key_cache_layout = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    if (data_type_traits::is_i8_u8(key_cache_layout.data_type)) {
        return true;
    } else {
        return false;
    }
}

int64_t get_aligned_seq_len(const kernel_impl_params& impl_param, const PagedAttentionStage& stage, int64_t target_seq_len_block_size = 16) {
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
        const auto subsequence_begins_mem = input_mem.at(PagedAttentionInputIdx::SUBSEQUENCE_BEGINS);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, *impl_param.strm);

        auto aligned_seq_len = 0;
        if (stage == PagedAttentionStage::MIXED) {
            const auto past_lens_mem = input_mem.at(PagedAttentionInputIdx::PAST_LENS);
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
            const auto& block_indices_ps = impl_param.get_input_layout(PagedAttentionInputIdx::BLOCK_INDICES).get_partial_shape();

            aligned_seq_len = block_indices_ps[0].get_length() * target_seq_len_block_size;
        } else {
            aligned_seq_len = calculate_aligned_seq_len();
        }
    } else {
        aligned_seq_len = calculate_aligned_seq_len();
    }

    return aligned_seq_len;
}

size_t get_partition_size() {
    // size_t k_partition_blok_num = (kv_len + 8191) / 8192;
    // if (k_partition_blok_num < 1)
    //     k_partition_blok_num = 1;
    // const size_t k_partition_blok_num = 16;
    // return k_partition_blok_num * PA_KV_CACHE_BLOCK_SIZE; // 128
    if (PA_KV_CACHE_BLOCK_SIZE < 128) {
        return 128;
    } else {
        return PA_KV_CACHE_BLOCK_SIZE;
    }
}

size_t get_partition_num(const size_t kv_len) {
    const size_t partition_size = get_partition_size();
    const size_t partition_num = (kv_len + partition_size - 1) / partition_size;

    return partition_num;
}

// max_context_len = max(past_lens + prompt_lens)
size_t get_max_context_len(const kernel_impl_params& params) {
    const auto& input_mem = params.memory_deps;
    const auto max_context_len = input_mem.at(PagedAttentionInputIdx::MAX_CONTEXT_LEN);
    mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *params.strm);
    const auto paged_attention_max_len = max_context_len_mem_lock[0];
    return paged_attention_max_len;
}

size_t get_past_len(const kernel_impl_params& params, const size_t seq_idx) {
    const auto& input_mem = params.memory_deps;
    const auto past_len = input_mem.at(PagedAttentionInputIdx::PAST_LENS);
    mem_lock<int32_t, mem_lock_type::read> past_len_mem_lock(past_len, *params.strm);
    const auto paged_attention_past_len = past_len_mem_lock[seq_idx];
    return paged_attention_past_len;
}

const float get_xattn_thresh(const kernel_impl_params& impl_param, const size_t seq_idx) {
    (void) seq_idx; // TODO

    static const char* env = std::getenv("OV_GPU_XATTN_THRESH");
    static const float thresh = env ? std::strtof(env, nullptr) : 0.9;
    return thresh;
}

inline void dump_block_indices_begins(const kernel_impl_params& params) {
    const auto& input_mem = params.memory_deps;
    const auto mem = input_mem.at(PagedAttentionInputIdx::BLOCK_INDICES_BEGINS);
    mem_lock<int32_t, mem_lock_type::read> mem_lock(mem, *params.strm);
    std::cout << "============ dump BLOCK_INDICES_BEGINS [";
    for (size_t i = 0; i < mem->count(); i++)
        std::cout << mem_lock[i] << ", ";
    std::cout << "]" << std::endl;
}

inline void dump_block_indices(const kernel_impl_params& params) {
    const auto& input_mem = params.memory_deps;
    const auto mem = input_mem.at(PagedAttentionInputIdx::BLOCK_INDICES);
    mem_lock<int32_t, mem_lock_type::read> mem_lock(mem, *params.strm);
    std::cout << "============ dump BLOCK_INDICES [";
    for (size_t i = 0; i < mem->count(); i++)
        std::cout << mem_lock[i] << ", ";
    std::cout << "]" << std::endl;
}

PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param) {
    const auto& query_shape = impl_param.get_input_layout(PagedAttentionInputIdx::QUERY).get_partial_shape();
    const auto& past_lens_shape = impl_param.get_input_layout(PagedAttentionInputIdx::PAST_LENS).get_partial_shape();

    if (query_shape.is_static() && past_lens_shape.is_static()) {
        if (query_shape[0].get_length() == past_lens_shape[0].get_length()) {
            return PagedAttentionStage::GENERATE;
        }

        const auto& memory_deps = impl_param.memory_deps;
        const auto past_lens_mem = memory_deps.at(PagedAttentionInputIdx::PAST_LENS);
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

//-----------------------------------------------------------------------------------------------------------------
// Base generator
//-----------------------------------------------------------------------------------------------------------------
JitConstants PagedAttentionGeneratorBase::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
    // std::cout << "PagedAttentionGeneratorBase::get_jit_constants: " << get_entry_point(params) << std::endl;

    // auto desc = params.typed_desc<paged_attention>();
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    jit.make("XE_ARCH", xe_arch);

    auto split_size = get_kv_split_size(xe_arch);
    jit.make("KV_STEP", split_size.first);

    jit.make("WG_SIZE", WG_SIZE);
    jit.make("CAUSAL_MASK", 1);
    return jit;
}

//-----------------------------------------------------------------------------------------------------------------
// KV cache update generator
//-----------------------------------------------------------------------------------------------------------------
JitConstants PagedAttentionGeneratorKVCacheUpdate::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);

    const auto desc = params.typed_desc<paged_attention>();
    jit.make("KV_HEADS_NUM", desc->kv_heads_num);
    jit.make("K_HEAD_SIZE", desc->k_head_size);
    jit.make("V_HEAD_SIZE", desc->v_head_size);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", PA_KV_CACHE_BLOCK_SIZE);

    if (get_kv_compressed(params)) {
        jit.make("KV_CACHE_COMPRESSION_PER_TOKEN", 1);
        jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + 4);
        jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size + 4);
    } else {
        jit.make("KV_CACHE_COMPRESSION_PER_TOKEN", 0);
        jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size);
        jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size);
    }

    return jit;
}

Arguments PagedAttentionGeneratorKVCacheUpdate::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY});                   // queries
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE});                 // keys cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // values cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block indices
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block indices begins
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});    // subsequence begins
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // queries
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});           // keys cache

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // key_pitch
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // key_offset
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // value_pitch
    args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // value_offset
    args.push_back({ArgumentDescriptor::Types::SCALAR, 4});  // batch_size_in_sequences
    return args;
}

DispatchDataFunc PagedAttentionGeneratorKVCacheUpdate::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        const auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

        // const size_t kv_len = get_max_context_len(params);
        const size_t kv_len = get_input_kv_len(params);
        const size_t kv_heads_num = desc->kv_heads_num;
        const size_t wg_count = (kv_len + WG_SIZE - 1) / WG_SIZE;

        wgs.global = {1, kv_heads_num, wg_count * WG_SIZE};
        wgs.local = {1, 1, WG_SIZE};

        auto& scalars = kd.params.scalars;
        size_t key_pitch = desc->k_head_size * kv_heads_num;
        size_t value_pitch = desc->v_head_size * kv_heads_num;
        auto key_layout = params.input_layouts[PagedAttentionInputIdx::KEY];
        auto value_layout = params.input_layouts[PagedAttentionInputIdx::VALUE];

        if (0) {  // Debug
            std::cout << "PagedAttentionGeneratorKVCacheUpdate::get_dispatch_data_func: "
                      << "key_layout: " << key_layout.to_string() << ", value_layout: " << value_layout.to_string() << std::endl;
            std::cout << "\tkey_dims = [";
            for (auto& it : key_layout.get_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tkey_pads = [";
            for (auto& it : key_layout.get_padded_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tvalue_dims = [";
            for (auto& it : value_layout.get_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tvalue_pads = [";
            for (auto& it : value_layout.get_padded_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
        }

        auto get_simple_pitch = [](const layout& layout) {
            size_t pitch = 1;
            auto dims_padding = layout.get_padded_dims();
            for(size_t i = dims_padding.size() - 1; i > 0; --i) {
                pitch = dims_padding[i];
                if(pitch > 1) {
                    break;
                }
            }
            return pitch;
        };
        key_pitch = get_simple_pitch(key_layout);
        value_pitch = get_simple_pitch(value_layout);

        auto get_simple_offset = [](const layout& layout) {
            size_t offset = 0;
            const auto& data_padding = layout.data_padding;
            const auto& lower_pads = data_padding._lower_size;
            for (auto& it : lower_pads) {
                if (it > 0) {
                    offset = it;
                    break;
                }
            }
            return offset;
        };
        size_t key_offset = get_simple_offset(key_layout);
        size_t value_offset = get_simple_offset(value_layout);

        if (DEBUG_ENABLED) {  // Debug
            std::cout << "PagedAttentionGeneratorKVCacheUpdate::get_dispatch_data_func: "
                      << "kv_len: " << kv_len << ", key_pitch: " << key_pitch << ", key_offset: " << key_offset << ", value_pitch: " << value_pitch
                      << ", value_offset: " << value_offset << ", gws: [" << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;
        }

        // TODO: support multiple sequences
        size_t batch_size_in_sequences = 1;
        std::vector<size_t> scaler_value = {key_pitch, key_offset, value_pitch, value_offset, batch_size_in_sequences};
        scalars.resize(scaler_value.size());

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

//-----------------------------------------------------------------------------------------------------------------
// multi token generator
//-----------------------------------------------------------------------------------------------------------------
Arguments PagedAttentionGeneratorMultiToken::get_arguments_desc(const kernel_impl_params& params) const {
    const auto desc = params.typed_desc<paged_attention>();

    Arguments args;
    // Doesn't support Query with dynamic_padding
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});                 // query
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // key_cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});           // value_cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // past_lens
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block_indices
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block_indices_begins
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});    // subsequence_begins

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    const size_t block_size = get_xattn_block_size(params);
    if (block_size > 1) {
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // sparse_block_mask
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});  // sparse_block_mask_wg
    }

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});           // q_len
    if (block_size > 1) {
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});       // q_block_pad
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});       // k_block_pad
    }
    return args;
}

JitConstants PagedAttentionGeneratorMultiToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();
    const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;

    const size_t xattn_block_size = get_xattn_block_size(params);

    jit.make("CMFLA_NUM_HEADS", desc->heads_num);
    jit.make("CMFLA_NUM_KV_HEADS", desc->kv_heads_num);
    jit.make("CMFLA_HEAD_SIZE", desc->k_head_size);
    jit.add(make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor));
    jit.make("CMFLA_IS_CAUSAL", 1);
    jit.make("CMPA_BLOCK_SZ", PA_KV_CACHE_BLOCK_SIZE);
    jit.make("SPARSE_BLOCK_SIZE", xattn_block_size);
    jit.make("Q_STEP", get_q_step(xe_arch, true));

    if (get_kv_compressed(params)) {
        jit.make("CMPA_KVCACHE_U8", 1);
    } else {
        jit.make("CMPA_KVCACHE_U8", 0);
    }
    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PagedAttentionGeneratorMultiToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        auto desc = params.typed_desc<paged_attention>();
        auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        // assert(rt_params != nullptr);
        const size_t heads_num = desc->heads_num;

        auto query_layout = params.input_layouts[PagedAttentionInputIdx::QUERY];

        if (0 && DEBUG_ENABLED) {  // Debug
            std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: query_layout: " << query_layout.to_string() << std::endl;
            std::cout << "\tquery_dims = [";
            for (auto& it : query_layout.get_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tquery_pads = [";
            for (auto& it : query_layout.get_padded_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
        }

        auto out_shape = params.output_layouts[0].get_shape();
        const size_t batch = out_shape.size() < 4 ? 1 : out_shape[0];
        const size_t q_len = out_shape[0];

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        const size_t q_step = get_q_step(xe_arch, false);
        const size_t wg_seq_len = WG_SIZE * q_step;
        const size_t wg_count = align_to(q_len, wg_seq_len) / wg_seq_len;

        wgs.global = {batch, heads_num, wg_count * WG_SIZE};
        wgs.local = {1, 1, WG_SIZE};

        if (DEBUG_ENABLED) {  // Debug
            std::cout << "PagedAttentionGeneratorMultiToken::get_dispatch_data_func: \n"
                      << "\tbatch: " << batch << ", heads_num: " << heads_num << ", q_len: " << q_len << ", q_step: " << q_step
                      << ", wg_seq_len: " << wg_seq_len << ", wg_count: " << wg_count << ", gws: [" << wgs.global[0] << ", " << wgs.global[1] << ", "
                      << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;
        }

        std::vector<size_t> scaler_value = {q_len};
        const size_t block_size = get_xattn_block_size(params);
        if (block_size > 1) {
            scaler_value.push_back(rtp->xattn_q_block_pad);
            scaler_value.push_back(rtp->xattn_k_block_pad);
        }
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

//-----------------------------------------------------------------------------------------------------------------
// single token generator
//-----------------------------------------------------------------------------------------------------------------
JitConstants PagedAttentionGeneratorSingleToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
    // jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
    auto desc = params.typed_desc<paged_attention>();
    const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
    const size_t kv_partition_size = get_partition_size();
    auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;

    jit.make("KV_PARTITION_SIZE", kv_partition_size);
    jit.make("KV_BLOCK_SIZE", PA_KV_CACHE_BLOCK_SIZE);
    jit.add(make_jit_constant("SCALE_FACTOR", scale_factor));
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("HEADS_NUM", desc->heads_num);
    jit.make("KV_HEADS_NUM", desc->kv_heads_num);
    jit.make("Q_STEP", get_q_step(xe_arch, true));

    if (get_kv_compressed(params)) {
        jit.make("KV_CACHE_COMPRESSION", 1);
    } else {
        jit.make("KV_CACHE_COMPRESSION", 0);
    }

    return jit;
}

Arguments PagedAttentionGeneratorSingleToken::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    const auto desc = params.typed_desc<paged_attention>();
    // const auto has_scale_input = !desc->scale_val.has_value();
    const auto has_scores_output = params.output_layouts.size() > 1;
    OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleToken with scores output is not supported yet");

    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});                 // queries
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // keys cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});           // values cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // past lens
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block indices
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block indices begins
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});    // subsequence begins

    // outputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // partition output
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // lse output

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len==1
    // args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_partition_num

    return args;
}

DispatchDataFunc PagedAttentionGeneratorSingleToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;
        const auto desc = params.typed_desc<paged_attention>();
        auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
        assert(rt_params != nullptr);

        const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
        const size_t heads_num = desc->heads_num;
        const size_t kv_heads_num = desc->kv_heads_num;
        const size_t partition_num = rtp->num_of_partitions;  // get_partition_num(rtp->max_context_len);

        wgs.global = {batch, kv_heads_num, partition_num};
        wgs.local = {1, 1, 1};

        // generate stage: q_len=1
        auto& scalars = kd.params.scalars;
        std::vector<size_t> scaler_value = {1};
        scalars.resize(scaler_value.size());

        if (DEBUG_ENABLED) {  // Debug
            size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
            size_t max_context_len = get_max_context_len(params);
            size_t past_len = get_past_len(params, 0);
            std::cout << "PagedAttentionGeneratorSingleToken::get_dispatch_data_func: "
                      << "batch: " << batch << ", heads_num: " << heads_num << ", partition_num: " << partition_num << ", kv_len: " << kv_len
                      << ", max_context_len = " << max_context_len << ", past_len = " << past_len << ", gws: [" << wgs.global[0] << ", " << wgs.global[1]
                      << ", " << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;
        }

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

//-----------------------------------------------------------------------------------------------------------------
// single token finalization generator
//-----------------------------------------------------------------------------------------------------------------
JitConstants PagedAttentionGeneratorSingleTokenFinalization::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
    const auto desc = params.typed_desc<paged_attention>();

    jit.make("REDUCE_SPLIT_SIZE", reduce_split_step);
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("HEADS_NUM", desc->heads_num);
    return jit;
}

Arguments PagedAttentionGeneratorSingleTokenFinalization::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    const auto has_scores_output = params.output_layouts.size() > 1;
    OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleTokenFinalization with scores output is not supported yet");

    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});  // partition data
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});           // output
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});  // lse

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // kv_partition_num

    return args;
}

DispatchDataFunc PagedAttentionGeneratorSingleTokenFinalization::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        const auto desc = params.typed_desc<paged_attention>();
        auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

        assert(rt_params != nullptr);

        const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
        const size_t heads_num = desc->heads_num;
        const size_t head_size = desc->k_head_size;
        wgs.global = {batch, heads_num, head_size / reduce_split_step};
        wgs.local = {1, 1, 1};

        auto& scalars = kd.params.scalars;
        const size_t partition_num = rtp->num_of_partitions;  // get_partition_num(rtp->max_context_len);
        std::vector<size_t> scaler_value = {partition_num};
        scalars.resize(scaler_value.size());

        if (DEBUG_ENABLED) {  // Debug
            std::cout << "PagedAttentionGeneratorSingleTokenFinalization::get_dispatch_data_func: "
                      << "batch: " << batch << ", heads_num: " << heads_num << ", partition_num: " << partition_num << ", gws: [" << wgs.global[0] << ", "
                      << wgs.global[1] << ", " << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;
        }

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}


//-----------------------------------------------------------------------------------------------------------------
// Helpers of XAttention
//-----------------------------------------------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------------------------------
// Base generator of XAttention
//-----------------------------------------------------------------------------------------------------------------
JitConstants XAttentionEstimateGeneratorBase::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

    auto desc = params.typed_desc<paged_attention>();

    const float scale_factor = 1.0f / std::sqrt(static_cast<float>(desc->k_head_size)) / STRIDE;
    int scale_factor_i;
    std::memcpy(static_cast<void*>(&scale_factor_i), &scale_factor, sizeof(scale_factor));

    jit.make("STRIDE", STRIDE);
    jit.make("HQ", desc->heads_num);
    jit.make("HK", desc->kv_heads_num);
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("SG_M", SG_M);
    jit.make("SG_N", SG_N);
    jit.make("BLOCK_SG_M", BLOCK_SG_M);
    jit.make("BLOCK_SG_N", BLOCK_SG_N);
    jit.make("BLOCK_SIZE", get_xattn_block_size(params));
    jit.make("KV_BLOCK_SIZE", PA_KV_CACHE_BLOCK_SIZE);
    jit.add(make_jit_constant("INV_S", scale_factor_i));
    jit.make("BLOCK_SHARE_MAX", BLOCK_WG_N);
    //# loop order walks HQ first and the step is WALK_HQ, 1 means not walk HQ, 2 means walks 2 heads first. Valid value: 1, 2, 4...
    jit.make("WALK_HQ", desc->heads_num != desc->kv_heads_num ? 2 : 1);
    jit.make("IS_CAUSAL", 1);
    if (get_kv_compressed(params)) {
        jit.make("USE_INT8", 1);
        jit.make("HEAD_SIZE_KEY", desc->k_head_size + 2 * 2);
    } else {
        jit.make("USE_INT8", 0);
        jit.make("HEAD_SIZE_KEY", desc->k_head_size);
    }
    jit.make("SOFTMAX_TYPE", "float");

    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;

    return jit;
}

//-----------------------------------------------------------------------------------------------------------------
// XAttention Estimate gemm_qk generator
//-----------------------------------------------------------------------------------------------------------------
Arguments XAttentionEstimateGEMMQK::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // keys cache
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});                 // queries
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block indices
    args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block indices begins

    // outputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // kq_max_wg
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // kq_exp_partial_sum

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // M
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // N
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // K
    args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // query_pitch
    args.push_back({ArgumentDescriptor::Types::SCALAR, 4});  // slice_no
    args.push_back({ArgumentDescriptor::Types::SCALAR, 5});  // slice
    args.push_back({ArgumentDescriptor::Types::SCALAR, 6});  // q_start_strided

    return args;
}

DispatchDataFunc XAttentionEstimateGEMMQK::get_dispatch_data_func() const {
    return DispatchDataFunc{[&](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        const auto desc = params.typed_desc<paged_attention>();

        // XAttention estimate is following afer kvcache_update.
        const size_t kv_len = get_max_context_len(params) / STRIDE * STRIDE;
        // const size_t kv_heads_num = desc->kv_heads_num;
        const size_t heads_num = desc->heads_num;
        const size_t head_size = desc->k_head_size;

        auto querry_layout = params.input_layouts[PagedAttentionInputIdx::QUERY];
        auto key_layout = params.input_layouts[PagedAttentionInputIdx::KEY];

        if (DEBUG_ENABLED) {  // Debug
            std::cout << "XAttentionEstimateGEMMQK::get_dispatch_data_func: "
                      << "key_layout: " << key_layout.to_string() << ", querry_layout: " << querry_layout.to_string() << std::endl;
            std::cout << "\tkey_dims = [";
            for (auto& it : key_layout.get_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tkey_pads = [";
            for (auto& it : key_layout.get_padded_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tquery_dims = [";
            for (auto& it : querry_layout.get_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "\tquery_pads = [";
            for (auto& it : querry_layout.get_padded_dims()) {
                std::cout << static_cast<size_t>(it) << ", ";
            }
            std::cout << "]" << std::endl;
        }

        auto out_shape = params.output_layouts[0].get_shape();
        const size_t q_len = out_shape[0];

        const uint32_t M = static_cast<uint32_t>(q_len / STRIDE);   //# will slient drop the tails which is less than `stride`
        const uint32_t N = static_cast<uint32_t>(kv_len / STRIDE);
        const uint32_t K = STRIDE * head_size;
        auto get_simple_pitch = [](const layout& layout) {
            size_t pitch = 1;
            auto dims_padding = layout.get_padded_dims();
            for(size_t i = dims_padding.size() - 1; i > 0; --i) {
                pitch = dims_padding[i];
                if(pitch > 1) {
                    break;
                }
            }
            return pitch;
        };
        const uint32_t query_pitch = get_simple_pitch(querry_layout) * STRIDE;
        const uint32_t slice_no = 0, slice = 0;

        const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);
        const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

        //# loop order walks HQ first and the step is WALK_HQ, 1 means not walk HQ, 2 means walks 2 heads first. Valid value: 1, 2, 4...
        const size_t WALK_HQ = desc->heads_num != desc->kv_heads_num ? 2 : 1;

        auto& wgs = kd.params.workGroups;
        wgs.global = {N_kq_groups * (q_stride_pad / BLOCK_WG_M) * SG_N * WALK_HQ, SG_M, heads_num / WALK_HQ};
        wgs.local = {SG_N, SG_M, 1};

        const uint32_t q_start_strided = N - M;
        OPENVINO_ASSERT(N >= M, "length of key cache must be greater or equal than query");

        auto& scalars = kd.params.scalars;
        std::vector<size_t> scaler_value = {M, N, K, query_pitch, slice_no, slice, q_start_strided};
        scalars.resize(scaler_value.size());

        if (DEBUG_ENABLED) {  // Debug
            size_t kv_len = get_kv_len(params, PagedAttentionStage::PREFILL);
            size_t max_context_len = get_max_context_len(params);
            size_t past_len = get_past_len(params, 0);
            std::cout << "XAttentionEstimateGEMMQK::get_dispatch_data_func: "
                      << "N_kq_groups: " << N_kq_groups << ", q_stride_pad: " << q_stride_pad << ", scaler_value: " << PartialShape(scaler_value) << ", kv_len: " << kv_len
                      << ", max_context_len = " << max_context_len << ", past_len = " << past_len << ", gws: [" << wgs.global[0] << ", " << wgs.global[1]
                      << ", " << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;

            dump_block_indices_begins(params);
            dump_block_indices(params);
        }

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            if (i == 4  || i == 5) {
                scalars[i].t = ScalarDescriptor::Types::INT32;
                scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
            } else {
                scalars[i].t = ScalarDescriptor::Types::UINT32;
                scalars[i].v.u32 = static_cast<uint32_t>(scaler_value[i]);
            }
        }
    }};
}

//-----------------------------------------------------------------------------------------------------------------
// XAttention Estimate find_block generator
//-----------------------------------------------------------------------------------------------------------------
Arguments XAttentionEstimateFindBlock::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    // inputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});  // kq_max_wg
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // kq_exp_partial_sum

    // outputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // block_mask

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // q_stride
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // q_stride_pad
    args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // q_block_pad
    args.push_back({ArgumentDescriptor::Types::SCALAR, 4});  // k_block_pad
    args.push_back({ArgumentDescriptor::Types::SCALAR, 5});  // causal_start_index
    args.push_back({ArgumentDescriptor::Types::SCALAR, 6});  // thresh

    return args;
}

DispatchDataFunc XAttentionEstimateFindBlock::get_dispatch_data_func() const {
    return DispatchDataFunc{[&](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        const auto desc = params.typed_desc<paged_attention>();
        // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

        assert(rt_params != nullptr);

        const uint32_t wg_k = BLOCK_WG_M;
        const uint32_t wg_q = BLOCK_WG_N;
        const size_t block_size = get_xattn_block_size(params);
        OPENVINO_ASSERT(wg_k % block_size == 0, "wg_k should be multiple of block_size then there is no tails from block_size");
        OPENVINO_ASSERT(wg_q % block_size == 0, "wg_q should be multiple of block_size then there is no tails from block_size");

        const size_t sum_per_n_token_in_block = static_cast<size_t>(block_size / STRIDE);

        // const size_t batch = params.input_layouts[PagedAttentionInputIdx::QUERY].get_partial_shape()[0].get_length();
        const size_t heads_num = desc->heads_num;
        // const size_t head_size = desc->k_head_size;

        auto out_shape = params.output_layouts[0].get_shape();
        const size_t kv_len = get_max_context_len(params) / STRIDE * STRIDE;
        const size_t q_len = out_shape[0];
        const uint32_t M = static_cast<uint32_t>(q_len / STRIDE);   //# will slient drop the tails which is less than `stride`
        const uint32_t N = static_cast<uint32_t>(kv_len / STRIDE);
        const uint32_t q_stride = M;
        const uint32_t k_stride = N;
        const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);
        const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

        const uint32_t sum_per_token_in_block = static_cast<uint32_t>(block_size / STRIDE);
        const uint32_t k_block_in_group = static_cast<uint32_t>(BLOCK_WG_N / sum_per_token_in_block);
        const uint32_t k_block_pad = k_block_in_group * N_kq_groups;
        const uint32_t q_block_pad = ceil_div(q_len, block_size);

        const uint32_t q_block = ceil_div(q_stride, sum_per_n_token_in_block);
        const uint32_t k_block = ceil_div(k_stride, sum_per_n_token_in_block);

        const float xattn_thresh = get_xattn_thresh(params, 0); // TODO: seq_idx

        wgs.global = {q_block_pad, heads_num, 1};
        wgs.local = {1, 1, 1};

        auto& scalars = kd.params.scalars;
        std::vector<size_t> scaler_value = {q_len, q_stride, q_stride_pad, q_block_pad, k_block_pad, k_block - q_block};
        scalars.resize(scaler_value.size() + 1);

        if (DEBUG_ENABLED) {  // Debug
            std::cout << "XAttentionEstimateFindBlock::get_dispatch_data_func: "
                      << "xattn_thresh : " << xattn_thresh
                      << " k_block: " << k_block << ", q_block: " << q_block
                      << " q_stride: " << q_stride << ", q_stride_pad: " << q_stride_pad << ", k_block_pad: " << k_block_pad << ", gws: [" << wgs.global[0] << ", "
                      << wgs.global[1] << ", " << wgs.global[2] << "]"
                      << ", lws: [" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "]" << std::endl;
        }

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::UINT32;
            scalars[i].v.u32 = static_cast<uint32_t>(scaler_value[i]);
        }
        scalars[scaler_value.size()].t = ScalarDescriptor::Types::FLOAT32;  // the last is for thresh with f32 dtype
        scalars[scaler_value.size()].v.f32 = static_cast<float>(xattn_thresh);
    }};
}

//-----------------------------------------------------------------------------------------------------------------
// XAttention Estimate post_proc generator
//-----------------------------------------------------------------------------------------------------------------
JitConstants XAttentionEstimatePostProc::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = XAttentionEstimateGeneratorBase::get_jit_constants(params);

    jit.make("MERGED_Q_NUM", 2);  // TODO

    return jit;
}

Arguments XAttentionEstimatePostProc::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    // inputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});  // block_mask

    // outputs
    args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});  // block_mask_merged

    // scalar
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_stride_pad
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // q_block_pad
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // k_block_pad

    return args;
}

DispatchDataFunc XAttentionEstimatePostProc::get_dispatch_data_func() const {
    return DispatchDataFunc{[&](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        assert(!params.is_dynamic());
        auto& wgs = kd.params.workGroups;

        const auto desc = params.typed_desc<paged_attention>();

        assert(rt_params != nullptr);

        const size_t block_size = get_xattn_block_size(params);
        const size_t heads_num = desc->heads_num;

        auto out_shape = params.output_layouts[0].get_shape();
        const size_t kv_len = get_max_context_len(params) / STRIDE * STRIDE;
        const size_t q_len = out_shape[0];
        const uint32_t M = static_cast<uint32_t>(q_len / STRIDE);   //# will slient drop the tails which is less than `stride`
        const uint32_t N = static_cast<uint32_t>(kv_len / STRIDE);
        const size_t q_stride_pad = round_up_to(M, BLOCK_WG_M);
        const size_t N_kq_groups = ceil_div(N, BLOCK_WG_N);

        const uint32_t sum_per_token_in_block = static_cast<uint32_t>(block_size / STRIDE);
        const uint32_t k_block_in_group = static_cast<uint32_t>(BLOCK_WG_N / sum_per_token_in_block);
        const uint32_t k_block_pad = k_block_in_group * N_kq_groups;
        const uint32_t q_block_pad = ceil_div(q_len, block_size);

        const uint32_t MERGED_Q_NUM = 2; // TODO
        const uint32_t q_block_pad_merged = ceil_div(q_block_pad, MERGED_Q_NUM);

        wgs.global = {q_block_pad_merged, heads_num, 1};
        wgs.local = {1, 1, 1};

        auto& scalars = kd.params.scalars;
        std::vector<size_t> scaler_value = {q_stride_pad, q_block_pad, k_block_pad};
        scalars.resize(scaler_value.size());

        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::UINT32;
            scalars[i].v.u32 = static_cast<uint32_t>(scaler_value[i]);
        }
    }};
}

}  // namespace ov::intel_gpu::cm