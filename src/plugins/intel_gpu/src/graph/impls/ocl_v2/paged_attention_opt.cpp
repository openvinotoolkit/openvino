// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "paged_attention_opt.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "sdpa_base.hpp"
#include "sdpa_gen_micro.hpp"
#include "sdpa_gen_opt.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
constexpr size_t paged_attention_block_size = 16;
constexpr size_t seq_len_partition_size = 256;
constexpr size_t subgroup_size = 16;

struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
    PagedAttentionStage stage;
    size_t num_of_partitions;
    size_t partition_size;
    size_t paged_attention_aligned_seq_len;
    size_t sdpa_opt_max_seq_len;
    size_t sdpa_opt_seq_len_partition_size;

    size_t paged_attention_snap_kv_tokens;
    // bool is_kv_compressed = false;
    bool use_micro_sdpa = false;
    size_t query_block_size = 16;
};

inline bool get_kv_compressed(const RuntimeParams& params) {
    auto key_cache_layout = params.input_layouts[PagedAttentionInputIdx::KEY_CACHE];
    if (data_type_traits::is_i8_u8(key_cache_layout.data_type)) {
        return true;
    } else {
        return false;
    }
}

inline size_t get_element_size(ov::element::Type_t type) {
    switch (type) {
    case ov::element::Type_t::i64:
        return 8;
    case ov::element::Type_t::f32:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::u32:
        return 4;
    case ov::element::Type_t::f16:
        return 2;
    case ov::element::Type_t::i8:
    case ov::element::Type_t::u8:
        return 1;
    default:
        OPENVINO_ASSERT(false, "Unsupported element type for get_element_size");
        return 0;  // Fallback case, should not be reached
    }
}

static size_t get_heads_per_wi(const size_t kv_group_size) {
    if (kv_group_size > 1) {
        std::vector<size_t> preferable_head_nums = {4, 3, 2};
        for (const auto& heads_num : preferable_head_nums) {
            const auto leftovers = kv_group_size % heads_num;
            if (leftovers == 0 || heads_num - leftovers <= 1) {
                return heads_num;
            }
        }
    }

    return 1;
}

size_t get_generate_stage_block_size(size_t head_size) {
    auto preferred_block_size = {4, 2, 1};
    for (const auto& block_size : preferred_block_size) {
        if (head_size % (block_size * subgroup_size) == 0) {
            return block_size;
        }
    }

    return 1;
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

std::pair<size_t, size_t> get_partitioning_params(const kernel_impl_params& params, size_t head_size, PagedAttentionStage stage) {
    const auto& input_mem = params.memory_deps;
    const auto max_context_len = input_mem.at(PagedAttentionInputIdx::MAX_CONTEXT_LEN);
    mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *params.strm);
    const auto paged_attention_max_len = max_context_len_mem_lock[0];

    size_t partition_size = 0;
    if (stage == PagedAttentionStage::PREFILL) {
        partition_size = head_size * get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::MULTI_TOKENS);
    } else {
        partition_size = seq_len_partition_size;
    }

    return {ceil_div(paged_attention_max_len, partition_size), partition_size};
}

PagedAttentionStage get_paged_attention_stage(const kernel_impl_params& impl_param) {
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

class PagedAttentionGeneratorBase : public KernelGenerator {
public:
    explicit PagedAttentionGeneratorBase(std::string_view stage_suffix) : KernelGenerator("paged_attention_opt", stage_suffix) {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);
        auto desc = params.typed_desc<paged_attention>();
        const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
        jit.make("K_HEAD_SIZE", desc->k_head_size);
        jit.make("V_HEAD_SIZE", desc->v_head_size);
        jit.make("HEADS_NUM", desc->heads_num);
        jit.make("KV_HEADS_NUM", desc->kv_heads_num);
        jit.make("KV_HEADS_GROUP_SIZE", kv_group_size);
        jit.make("SEQ_LEN_PARTITION_SIZE", seq_len_partition_size);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        jit.make("SLIDING_WINDOW_SIZE", desc->sliding_window);

        jit.make("HEADS_PER_WI", 1);

        bool is_kv_compressed = get_kv_compressed(params);
        jit.make("IS_KV_COMPRESSED", is_kv_compressed);
        jit.make("XE2_QK_MULTIPLICATION", params.get_device_info().arch == gpu_arch::xe2);

        // if (desc->heads_num != desc->kv_heads_num) {
        //     jit.make("BROADCAST_GROUP_SIZE", desc->heads_num / desc->kv_heads_num);
        // }

        if (is_kv_compressed) {
            auto& kv_dt = params.input_layouts[PagedAttentionInputIdx::KEY].data_type;
            auto scales_zp_size = get_element_size(kv_dt) * 2;  // scale + zp
            jit.make("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size);
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + scales_zp_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size + scales_zp_size);
        } else {
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size);
        }

        if (desc->scale_val.has_value()) {
            jit.make("SCALE_VAL", desc->scale_val.value());
        } else {
            const size_t scale_input_idx = PagedAttentionInputIdx::SCALE;
            jit.make("HAS_SCALE_INPUT", 1);
            jit.add(make_type_jit_constants("SCALE_INPUT", params.input_layouts[scale_input_idx].data_type));
        }

        if (desc->has_alibi) {
            // const size_t alibi_input_idx = desc->scale_val.has_value() ? 7 : 8; // why?
            const size_t alibi_input_idx = PagedAttentionInputIdx::ALIBI;
            jit.make("HAS_ALIBI", 1);
            jit.add(make_type_jit_constants("ALIBI_INPUT", params.input_layouts[alibi_input_idx].data_type));
        }

        if (params.output_layouts.size() > 1) {
            jit.make("PAGED_ATTENTION_SCORES_OUTPUT", 1);
            if (desc->has_score_aggregation) {
                jit.make("HAS_SCORE_AGGREGATION", 1);
            }
        }

        const size_t score_aggregation_idx = PagedAttentionInputIdx::SCORE_AGGREGATION;
        jit.add(make_type_jit_constants("SCORE_AGGREGATION_INPUT", params.input_layouts[score_aggregation_idx].data_type));

        if (desc->has_rotated_blocks) {
            jit.make("HAS_ROTATED_BLOCKS", 1);
        }

        jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", softmax_accumulator_type));

        return jit;
    }

    static void add_intermediate_inputs(Arguments& args, bool has_scores_output, bool is_multi_token_kernel = false) {
        uint32_t internal_buffers_num = 3;  // kv cache update buffers
        if (has_scores_output) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // softmax_results
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // subsequent_offsets
        }

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // exp_sums
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // max_logits
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // intermediate output

        if (is_multi_token_kernel) {
            // MULTIPLE_TOKENS kernels needs additional information related to mapping
            // launched kernel instances to subsequence indexes
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, internal_buffers_num++});  // gws_subseq_mapping
        }
    }
};

class PagedAttentionGeneratorSingleToken : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleToken() : PagedAttentionGeneratorBase("_single_token") {}

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        jit.make("SDPA_STAGE_0", 1);

        const auto desc = params.typed_desc<paged_attention>();
        const size_t kv_group_size = desc->heads_num / desc->kv_heads_num;
        if (kv_group_size > 1) {  // KernelsTypes::SINGLE_TOKEN_GQA
            auto heads_per_wi = get_heads_per_wi(kv_group_size);
            // std::cout << "heads_per_wi = " << heads_per_wi << std::endl;
            jit.make("HEADS_PER_WI", heads_per_wi);
            jit.make("ITERATIONS_PER_KV_HEADS_GROUP", ceil_div(kv_group_size, heads_per_wi));
            jit.make("HEADS_LEFTOVERS_NUM", kv_group_size % heads_per_wi);
        } else {
            jit.make("HEADS_PER_WI", 1);
        }

        const auto has_alibi = params.get_input_layout(PagedAttentionInputIdx::ALIBI).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(params.get_device_info(), desc->k_head_size, SDPAStage::SINGLE_TOKEN));
        // constexpr static std::array input_ids = {0, 3, 4, 5, 7, 8};
        constexpr static std::array input_ids = {PagedAttentionInputIdx::QUERY,
                                                 PagedAttentionInputIdx::KEY_CACHE,
                                                 PagedAttentionInputIdx::VALUE_CACHE,
                                                 PagedAttentionInputIdx::PAST_LENS,
                                                 PagedAttentionInputIdx::BLOCK_INDICES,
                                                 PagedAttentionInputIdx::BLOCK_INDICES_BEGINS};
        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids[i];
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_scale_input) {
            const size_t tensor_id = PagedAttentionInputIdx::SCALE;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(6), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_alibi) {
            const size_t tensor_id = PagedAttentionInputIdx::ALIBI;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(7), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

        // for (auto it : jit) {
        //     std::cout << "jit[" << it.name << "] = " << it.value << std::endl;
        // }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = params.output_layouts.size() > 1;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});                 // queries
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // keys
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});           // values
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block_indices
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block_indices_begins

        if (has_scale_input) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});  // scale
        }

        if (has_alibi) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ALIBI});  // alibi
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        add_intermediate_inputs(args, has_scores_output);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto desc = params.typed_desc<paged_attention>();

            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t total_tokens = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->v_head_size;

            auto sg_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::SINGLE_TOKEN);
            wgs.global = {total_tokens, heads_num, head_size * rtp->num_of_partitions * sg_scale};
            wgs.local = {1, 1, head_size * sg_scale};

            // std::cout << "PagedAttentionGeneratorSingleToken: total_tokens = " << total_tokens
            //           << ", heads_num = " << heads_num
            //           << ", head_size = " << head_size
            //           << ", num_of_partitions = " << rtp->num_of_partitions
            //           << ", global_work_size = {" << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << "}"
            //           << ", local_work_size = {" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "}"
            //           << std::endl;
        }};
    }
};

class PagedAttentionGeneratorSingleTokenFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorSingleTokenFinalization() : PagedAttentionGeneratorBase("_single_token_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        jit.make("SDPA_STAGE_1", 1);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        jit.add(make_layout_jit_constants("INPUT3", params.input_layouts[5], in_offsets_map.at(5)));
        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

        const auto desc = params.typed_desc<paged_attention>();
        jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(params.get_device_info(), desc->k_head_size, SDPAStage::FINALIZATION));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});  // past_lens
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        const auto has_scores_output = params.output_layouts.size() > 1;
        add_intermediate_inputs(args, has_scores_output);

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // total_partitions_num

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.resize(1);

            const auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t total_tokens = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->v_head_size;

            wgs.global = {total_tokens, heads_num, head_size};
            wgs.local = {1, 1, subgroup_size};

            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(rtp->num_of_partitions);

            // std::cout << "PagedAttentionGeneratorSingleTokenFinalization: total_tokens = " << total_tokens
            //           << ", heads_num = " << heads_num
            //           << ", head_size = " << head_size
            //           << ", num_of_partitions = " << rtp->num_of_partitions
            //           << ", global_work_size = {" << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << "}"
            //           << ", local_work_size = {" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "}"
            //           << std::endl;
        }};
    }
};

class PagedAttentionGeneratorMultiTokens : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorMultiTokens() : PagedAttentionGeneratorBase("_multi_tokens") {}

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        jit.make("SDPA_STAGE_0", 1);
        jit.make("MULTI_TOKENS_PROCESSING", 1);

        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        // constexpr static std::array input_ids = {0, 3, 4, 5, 7, 8, 6};
        constexpr static std::array input_ids = {PagedAttentionInputIdx::QUERY,
                                                 PagedAttentionInputIdx::KEY_CACHE,
                                                 PagedAttentionInputIdx::VALUE_CACHE,
                                                 PagedAttentionInputIdx::PAST_LENS,
                                                 PagedAttentionInputIdx::BLOCK_INDICES,
                                                 PagedAttentionInputIdx::BLOCK_INDICES_BEGINS,
                                                 PagedAttentionInputIdx::SUBSEQUENCE_BEGINS};
        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids[i];
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_scale_input) {
            const size_t tensor_id = PagedAttentionInputIdx::SCALE;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(6), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_alibi) {
            const size_t tensor_id = PagedAttentionInputIdx::ALIBI;
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(7), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
        jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(params.get_device_info(), desc->k_head_size, SDPAStage::MULTI_TOKENS));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(PagedAttentionInputIdx::ALIBI).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = params.output_layouts.size() > 1;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});                 // queries
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});             // keys
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});           // values
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block_indices
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block_indices_begins
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});    // subsequence_begins

        if (has_scale_input) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});  // scale
        }

        if (has_alibi) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ALIBI});  // alibi
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        add_intermediate_inputs(args, has_scores_output, true);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto desc = params.typed_desc<paged_attention>();
            auto* rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t total_tokens = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->v_head_size;

            wgs.global = {total_tokens, heads_num, head_size * rtp->num_of_partitions};
            wgs.local = {1, 1, head_size};
        }};
    }
};

class PagedAttentionGeneratorMultiTokensFinalization : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorMultiTokensFinalization() : PagedAttentionGeneratorBase("_multi_tokens_finalization") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        jit.make("SDPA_STAGE_1", 1);
        jit.make("MULTI_TOKENS_PROCESSING", 1);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        jit.add(make_layout_jit_constants("INPUT3", params.input_layouts[5], in_offsets_map.at(5)));
        jit.add(make_layout_jit_constants("INPUT6", params.input_layouts[6], in_offsets_map.at(6)));
        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

        const auto desc = params.typed_desc<paged_attention>();
        jit.make("SG_SCALE_FACTOR", get_sg_number_scale_factor(params.get_device_info(), desc->k_head_size, SDPAStage::FINALIZATION));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});           // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        const auto has_scores_output = params.output_layouts.size() > 1;
        add_intermediate_inputs(args, has_scores_output, true);

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // total_partitions_num

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.resize(1);

            const auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const size_t total_tokens = params.input_layouts[0].get_partial_shape()[0].get_length();
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->v_head_size;

            wgs.global = {total_tokens, heads_num, head_size};
            wgs.local = {1, 1, subgroup_size};

            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(rtp->num_of_partitions);
        }};
    }
};

class PagedAttentionGeneratorScoresCalculation : public PagedAttentionGeneratorBase {
public:
    PagedAttentionGeneratorScoresCalculation() : PagedAttentionGeneratorBase("_scores_calculation") {}
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
        jit.make("SDPA_STAGE_2", 1);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        jit.add(make_layout_jit_constants("INPUT3", params.input_layouts[5], in_offsets_map.at(5)));
        jit.add(make_layout_jit_constants("INPUT6", params.input_layouts[6], in_offsets_map.at(6)));
        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
        jit.add(make_layout_jit_constants("OUTPUT1", params.output_layouts[1], out_offsets_map.at(1)));

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});           // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});                                          // out scores

        const auto has_scores_output = params.output_layouts.size() > 1;
        add_intermediate_inputs(args, has_scores_output);

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // total_partitions_num

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const auto& past_lens = params.input_layouts[PagedAttentionInputIdx::PAST_LENS];
            const auto subsequences_number = static_cast<size_t>(past_lens.get_partial_shape()[0].get_length());

            wgs.global = {rtp->partition_size * rtp->num_of_partitions, 1, subsequences_number};
            wgs.local = {rtp->partition_size, 1, 1};

            auto& scalars = kd.params.scalars;
            scalars.resize(1);
            const auto is_mixed_mode = rtp->stage == PagedAttentionStage::MIXED;
            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(is_mixed_mode);
        }};
    }
};

class KVCacheUpdateGenerator : public KernelGenerator {
public:
    KVCacheUpdateGenerator() : KernelGenerator("pa_kv_cache_update_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;

        // constexpr static std::array input_ids = {1, 2, 5, 7, 8, 6};
        constexpr static std::array input_ids = {PagedAttentionInputIdx::KEY,
                                                 PagedAttentionInputIdx::VALUE,
                                                 PagedAttentionInputIdx::PAST_LENS,
                                                 PagedAttentionInputIdx::BLOCK_INDICES,
                                                 PagedAttentionInputIdx::BLOCK_INDICES_BEGINS,
                                                 PagedAttentionInputIdx::SUBSEQUENCE_BEGINS};

        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids.at(i);
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        constexpr size_t key_cache_id = PagedAttentionInputIdx::KEY_CACHE;
        constexpr size_t value_cache_id = PagedAttentionInputIdx::VALUE_CACHE;

        jit.add(make_layout_jit_constants("OUTPUT", params.input_layouts[key_cache_id], in_offsets_map.at(key_cache_id)));
        jit.add(make_layout_jit_constants("OUTPUT" + to_code_string(1), params.input_layouts[value_cache_id], in_offsets_map.at(value_cache_id)));

        const auto desc = params.typed_desc<paged_attention>();
        jit.make("K_HEAD_SIZE", desc->k_head_size);
        jit.make("V_HEAD_SIZE", desc->v_head_size);
        jit.make("HEADS_NUM", desc->heads_num);
        jit.make("KV_HEADS_NUM", desc->kv_heads_num);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        jit.make("GENERATE_STAGE_K_BLOCK_SIZE", get_generate_stage_block_size(desc->k_head_size));
        jit.make("GENERATE_STAGE_V_BLOCK_SIZE", get_generate_stage_block_size(desc->v_head_size));

        const bool is_kv_compressed = get_kv_compressed(params);
        jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);
        if (is_kv_compressed) {
            auto data_type = params.input_layouts[PagedAttentionInputIdx::KEY].data_type;  // key tensor data size
            auto scales_zp_size = get_element_size(data_type) * 2;                         // scale + zp
            jit.make("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size);
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size + scales_zp_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size + scales_zp_size);
        } else {
            jit.make("ADJUSTED_K_HEAD_SIZE", desc->k_head_size);
            jit.make("ADJUSTED_V_HEAD_SIZE", desc->v_head_size);
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        // Inputs
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY});                   // key
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE});                 // value
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});             // past_lens
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});         // block_indices
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});  // block_indices_begins
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});    // subsequence_begins

        // Outputs
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});    // key_cache
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE_CACHE});  // value_cache

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            scalars.resize(1);

            const auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

            const auto is_prefill = rtp->stage == PagedAttentionStage::PREFILL || rtp->stage == PagedAttentionStage::MIXED;
            auto heads_number = desc->kv_heads_num;

            if (is_prefill) {
                const auto blocks_number = rtp->paged_attention_aligned_seq_len / paged_attention_block_size;

                wgs.global = {blocks_number, heads_number, subgroup_size};
                wgs.local = {1, 1, subgroup_size};
            } else {
                const auto& key_input = params.input_layouts[0];
                const auto sequences_number = key_input.get_partial_shape()[0].get_length();

                wgs.global = {static_cast<size_t>(sequences_number), heads_number, subgroup_size};
                wgs.local = {1, 1, subgroup_size};
            }

            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(is_prefill);

            // std::cout << "KVCacheUpdateGenerator: is_prefill = " << is_prefill
            //           << ", heads_number = " << heads_number
            //           << ", global_work_size = {" << wgs.global[0] << ", " << wgs.global[1] << ", " << wgs.global[2] << "}"
            //           << ", local_work_size = {" << wgs.local[0] << ", " << wgs.local[1] << ", " << wgs.local[2] << "}"
            //           << std::endl;
        }};
    }
};

class KVCacheRotateGenerator : public KernelGenerator {
public:
    KVCacheRotateGenerator() : KernelGenerator("pa_kv_cache_rotate_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);

        const auto desc = params.typed_desc<paged_attention>();

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;

        // constexpr static std::array input_ids = {13+1, 14+1, 15+1};
        constexpr static std::array input_ids = {PagedAttentionInputIdx::ROTATED_BLOCK_INDICES,
                                                 PagedAttentionInputIdx::ROTATION_DELTAS,
                                                 PagedAttentionInputIdx::ROTATION_TRIG_LUT};

        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids.at(i);
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        constexpr size_t key_cache_id = PagedAttentionInputIdx::KEY_CACHE;
        jit.add(make_layout_jit_constants("OUTPUT", params.input_layouts[key_cache_id], in_offsets_map.at(key_cache_id)));

        jit.make("HEAD_SIZE", desc->k_head_size);
        jit.make("HEADS_NUM", desc->heads_num);
        jit.make("KV_HEADS_NUM", desc->kv_heads_num);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);

        const bool is_kv_compressed = get_kv_compressed(params);
        jit.make("IS_KV_COMPRESSED", is_kv_compressed ? 1 : 0);

        const auto original_cache_dt = params.get_input_layout(PagedAttentionInputIdx::KEY).data_type;
        jit.add(make_type_jit_constants("UNCOMPRESSED", original_cache_dt));

        if (is_kv_compressed) {
            auto scales_zp_size = get_element_size(original_cache_dt) * 2;  // scale + zp;
            jit.make("SCALE_ZP_SIZE_PER_TOKEN", scales_zp_size);
            jit.make("ADJUSTED_HEAD_SIZE", desc->k_head_size + scales_zp_size);
        } else {
            jit.make("ADJUSTED_HEAD_SIZE", desc->k_head_size);
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ROTATED_BLOCK_INDICES});  // rotated_block_indices
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ROTATION_DELTAS});        // rotation_deltas
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ROTATION_TRIG_LUT});      // rotation_trig_lut

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});  // key_cache

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            const auto desc = params.typed_desc<paged_attention>();
            const auto& rotated_block_indices_input = params.input_layouts[13];
            auto heads_number = static_cast<size_t>(desc->kv_heads_num);
            auto blocks_to_rotate = static_cast<size_t>(rotated_block_indices_input.get_partial_shape()[0].get_length());
            const bool is_kv_compressed = get_kv_compressed(params);

            wgs.global = {subgroup_size, heads_number, blocks_to_rotate};
            wgs.local = {subgroup_size, is_kv_compressed ? 1 : heads_number, 1};
        }};
    }
};

class PagedAttentionSDPAOptGeneratorMultiToken : public SDPAOptGeneratorBase {
public:
    PagedAttentionSDPAOptGeneratorMultiToken() : SDPAOptGeneratorBase("sdpa_opt", "_multi_tokens", false) {}
    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();
        const auto has_scores_output = desc->has_scores_output();

        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::QUERY});               // query
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY});                 // key
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::VALUE});               // value
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SUBSEQUENCE_BEGINS});  // subsequence_begins
        if (has_scale_input) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::SCALE});  // scale
        }
        if (has_alibi) {
            args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ALIBI});  // alibi
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});

        if (has_scores_output) {
            // Intermediate buffers for PagedAttention scores calculation:
            // softmax_results, subsequence_offsets, exp_sums, max_logits, tmp_out
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 6});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 7});

            // Scalar used for proper offset calculation of intermediate data buffers
            args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        }

        return args;
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = SDPAOptGeneratorBase::get_jit_constants_base(params, SDPAStage::MULTI_TOKENS, false);
        const auto desc = params.typed_desc<paged_attention>();
        const auto has_alibi = params.get_input_layout(11).count() > 0;
        const auto has_scale_input = !desc->scale_val.has_value();

        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& out_offsets_map = params.out_port_to_shape_info_offset;

        // constexpr static std::array input_ids = {0, 1, 2, 6};
        constexpr static std::array input_ids = {PagedAttentionInputIdx::QUERY,
                                                 PagedAttentionInputIdx::KEY,
                                                 PagedAttentionInputIdx::VALUE,
                                                 PagedAttentionInputIdx::SUBSEQUENCE_BEGINS};
        for (size_t i = 0; i < input_ids.size(); i++) {
            const size_t tensor_id = input_ids.at(i);
            jit.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        if (has_scale_input) {
            const size_t tensor_id = PagedAttentionInputIdx::SCALE;
            jit.add(make_layout_jit_constants("INPUT4", params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }
        if (has_alibi) {
            const size_t tensor_id = PagedAttentionInputIdx::ALIBI;
            jit.add(make_layout_jit_constants("INPUT5", params.input_layouts[tensor_id], in_offsets_map.at(tensor_id)));
        }

        jit.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));

        jit.make("SDPA_STAGE_0", 1);
        jit.make("TARGET_SEQ_LEN_BLOCK_SIZE", get_target_seq_len_block_size());
        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            auto& scalars = kd.params.scalars;
            auto desc = params.typed_desc<paged_attention>();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
            const size_t heads_num = desc->heads_num;
            const size_t head_size = desc->v_head_size;

            const size_t sg_num_scale = get_sg_number_scale_factor(params.get_device_info(), head_size, SDPAStage::MULTI_TOKENS);

            wgs.global = {heads_num, ceil_div(rtp->paged_attention_aligned_seq_len, get_target_seq_len_block_size()), head_size * sg_num_scale};
            wgs.local = {1, 1, head_size * sg_num_scale};

            scalars.resize(1);
            scalars[0].t = ScalarDescriptor::Types::UINT32;
            scalars[0].v.u32 = static_cast<uint32_t>(align_to(rtp->sdpa_opt_max_seq_len, rtp->sdpa_opt_seq_len_partition_size));
        }};
    }
};

}  // namespace

class PagedAttentionOptImpl : public SDPAImplBase {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedAttentionOptImpl)

    Stage::Ptr kv_cache_update = make_stage<KVCacheUpdateGenerator>();
    Stage::Ptr pa_single_token = make_stage<PagedAttentionGeneratorSingleToken>();
    Stage::Ptr pa_single_token_finalization = make_stage<PagedAttentionGeneratorSingleTokenFinalization>();
    Stage::Ptr pa_multi_token = make_stage<PagedAttentionGeneratorMultiTokens>();
    Stage::Ptr pa_multi_token_finalization = make_stage<PagedAttentionGeneratorMultiTokensFinalization>();
    Stage::Ptr pa_sdpa_opt = make_stage<PagedAttentionSDPAOptGeneratorMultiToken>();
    Stage::Ptr kv_cache_rotate = make_stage<KVCacheRotateGenerator>();
    Stage::Ptr pa_scores_calc = make_stage<PagedAttentionGeneratorScoresCalculation>();
    Stage::Ptr pa_sdpa_micro = make_stage<SDPAMicroGenerator>(true);

    bool use_micro_sdpa = false;

    PagedAttentionOptImpl() : SDPAImplBase(PagedAttentionOpt::get_type_info_static()) {}
    explicit PagedAttentionOptImpl(const kernel_impl_params& params) : PagedAttentionOptImpl() {
        const auto desc = params.typed_desc<paged_attention>();
        const bool has_scores_output = params.output_layouts.size() > 1;
        const bool has_rotated_blocks = desc->has_rotated_blocks;
        // is_kv_compressed = get_kv_compressed(params);

        add_stage(kv_cache_update, params);
        add_stage(pa_single_token, params);
        add_stage(pa_single_token_finalization, params);
        add_stage(pa_multi_token, params);
        add_stage(pa_multi_token_finalization, params);

        add_stage(pa_sdpa_opt, params);

#ifdef ENABLE_ONEDNN_FOR_GPU
        use_micro_sdpa = supports_micro_sdpa(params);
        if (use_micro_sdpa) {
            add_stage(pa_sdpa_micro, params);
        }
#endif

        if (has_rotated_blocks) {
            add_stage(kv_cache_rotate, params);
        }

        if (has_scores_output) {
            add_stage(pa_scores_calc, params);
        }
    }

    bool supports_micro_sdpa(const kernel_impl_params& params) const {
#ifdef ENABLE_ONEDNN_FOR_GPU
        auto& engine = params.get_program().get_engine();
        const auto supports_microkernels = cldnn::query_microkernels_supported(engine, params.get_program().get_config());
        if (params.get_device_info().arch < gpu_arch::xe_hpg || !supports_microkernels) {
            return false;
        }

        const auto desc = params.typed_desc<paged_attention>();
        ov::Dimension head_num = desc->heads_num;
        ov::Dimension kv_heads_num = desc->kv_heads_num;

        // supposed k_heads_num==v_heads_num?
        if (head_num.is_dynamic() || kv_heads_num.is_dynamic()) {
            return false;
        }

        if (desc->k_head_size != desc->v_head_size) {
            return false;
        }

        if (desc->k_head_size > 256 || desc->v_head_size > 256) {
            return false;
        }

        const auto scale_idx = 4lu;
        const auto has_const_scale_val = desc->scale_val.has_value() ? true : false;
        if (!has_const_scale_val && !params.input_layouts[scale_idx].is_dynamic() && params.input_layouts[scale_idx].count() == 1) {
            return false;
        }

        if (params.output_layouts.size() > 1 || desc->has_score_aggregation) {
            return false;
        }

        if (desc->sliding_window || desc->has_alibi) {
            return false;
        }
        return true;
#else
        return false;
#endif
    }

    static size_t get_micro_tile_qsize(KernelData& kernel_data) {
        OPENVINO_ASSERT(kernel_data.micro_kernels.size() > 0, "[GPU] Invalid kernels passed to get_tile_qsize() function");

        const auto& gemms = kernel_data.micro_kernels;
        const auto wg_tile_q = gemms[0]->p.getSetting("wg_tile_n");
        return wg_tile_q;
    }

    size_t get_query_block_size(const PagedAttentionStage& stage, const bool use_micro_sdpa) {
        const auto default_block_size = 16;
#ifdef ENABLE_ONEDNN_FOR_GPU
        if (use_micro_sdpa && stage == PagedAttentionStage::PREFILL)
            return get_micro_tile_qsize(pa_sdpa_micro->kd);
#endif
        return default_block_size;
    }
    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        const auto& params = *instance.get_impl_params();
        const auto& desc = params.typed_desc<paged_attention>();
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<PagedAttentionRuntimeParams>();
        }

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        rt_params->stage = get_paged_attention_stage(params);
        std::tie(rt_params->num_of_partitions, rt_params->partition_size) = get_partitioning_params(params, desc->v_head_size, rt_params->stage);
        rt_params->paged_attention_aligned_seq_len = static_cast<size_t>(get_aligned_seq_len(params, rt_params->stage));
        rt_params->sdpa_opt_seq_len_partition_size = get_seq_len_partition_size(params.get_device_info(), desc->v_head_size, SDPAStage::MULTI_TOKENS);

        const auto& input_mem = params.memory_deps;
        const auto max_context_len = input_mem.at(PagedAttentionInputIdx::MAX_CONTEXT_LEN);  // max_context_len input memory
        mem_lock<int32_t, mem_lock_type::read> max_context_len_mem_lock(max_context_len, *params.strm);
        rt_params->sdpa_opt_max_seq_len = static_cast<int64_t>(max_context_len_mem_lock[0]);

        if (desc->has_score_aggregation) {
            const auto score_aggregation = input_mem.at(PagedAttentionInputIdx::SCORE_AGGREGATION);
            mem_lock<int32_t, mem_lock_type::read> score_aggregation_mem_lock(score_aggregation, *params.strm);

            auto total_tokens_num = 0;
            for (size_t i = 0; i < score_aggregation_mem_lock.size(); i++) {
                total_tokens_num += score_aggregation_mem_lock[i];
            }
            rt_params->paged_attention_snap_kv_tokens = total_tokens_num;
        } else {
            rt_params->paged_attention_snap_kv_tokens = 0;
        }

        if (rt_params->stage == PagedAttentionStage::PREFILL) {
            // Determine if sdpa_micro can be used based on sliding_window and aliged_seq_len
            rt_params->use_micro_sdpa = supports_micro_sdpa(params);
            // desc->sliding_window == 0 || (desc->sliding_window > 0 && rt_params->paged_attention_aligned_seq_len < desc->sliding_window);
            rt_params->query_block_size = get_query_block_size(rt_params->stage, rt_params->use_micro_sdpa);
        } else {
            rt_params->use_micro_sdpa = false;
        }
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        const auto desc = params.typed_desc<paged_attention>();
        const auto& head_size = desc->v_head_size;
        const bool has_scores_output = params.output_layouts.size() > 1;
        const bool has_rotated_blocks = desc->has_rotated_blocks;

        update_rt_params(instance);

        auto rt_params = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
        assert(rt_params != nullptr);
        prepare_internal_buffers(static_cast<paged_attention_inst&>(instance), rt_params->stage, rt_params->use_micro_sdpa, rt_params->query_block_size);

        std::vector<event::ptr> res_event = events;
        if (has_rotated_blocks) {
            res_event = {execute_stage(res_event, instance, kv_cache_rotate)};
        }

        res_event = {execute_stage(res_event, instance, kv_cache_update)};

        if (rt_params->stage == PagedAttentionStage::PREFILL) {
            if (rt_params->use_micro_sdpa) {
                std::cout << "prefill: using sdpa_micro kernel" << std::endl;
                res_event = {execute_stage(res_event, instance, pa_sdpa_micro)};
            } else {
                std::cout << "prefill: using sdpa_opt kernel" << std::endl;
                res_event = {execute_stage(res_event, instance, pa_sdpa_opt)};
            }
        } else if (rt_params->stage == PagedAttentionStage::GENERATE || rt_params->stage == PagedAttentionStage::MIXED) {
            std::cout << "generate: using sdpa_opt kernel" << std::endl;
            const auto multi_tokens_mode = rt_params->stage == PagedAttentionStage::MIXED;
            auto num_of_partitions = get_partitioning_params(params, head_size, rt_params->stage).first;
            res_event = {execute_stage(res_event, instance, multi_tokens_mode ? pa_multi_token : pa_single_token)};
            if (num_of_partitions > 1) {
                res_event = {execute_stage(res_event, instance, multi_tokens_mode ? pa_multi_token_finalization : pa_single_token_finalization)};
            }
        }

        if (has_scores_output) {
            res_event = {execute_stage(res_event, instance, pa_scores_calc)};
        }

        return res_event[0];
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        /*
         * Internal buffers allocation owners and users (numbers represent unique buffers names,
         * not the real indexes in _intermediates_memory structure):
         * +--------------------------------------------------+-----------------------+--------------------+
         * | Stage                                            | Allocates & uses      | Reuses             |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | KV_CACHE_UPDATE                                  | [0, 1, 2]             |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | SDPA (1st token)                                 |                       | [0, 1, 2]          |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (2nd+ token)                             | [5, 6, 7]             |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (mixed mode)                             | [5, 6, 7, 8]          |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | SDPA (1st token) + scores output                 |                       | [0, 1, 2, 3, 4]    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (2nd+ token) + scores output             | [3, 4, 5, 6, 7]       |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (mixed mode) + scores output             | [3, 4, 5, 6, 7, 8]    |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | SDPA (1st token) + scores output aggregation     |                       | [0, 1, 2, 3, 4, 5] |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (2nd+ token) + scores output aggregation | [3, 4, 5, 6, 7, 8]    |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | PA_SDPA (mixed mode) + scores output aggregation | [3, 4, 5, 6, 7, 8, 9] |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
         * | SDPA (1st token, micro-kernel)                   | [last (8/9/10)]       |                    |
         * +--------------------------------------------------+-----------------------+--------------------+
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

        std::vector<BufferDescriptor> internal_buffers;
        const auto desc = params.typed_desc<paged_attention>();

        const auto indexes_dt = ov::element::i32;
        const int64_t target_seq_len_block_size = 16;
        auto stage = get_paged_attention_stage(params);
        int64_t paged_attention_aligned_seq_len = -1;
        if ((stage == PagedAttentionStage::PREFILL || stage == PagedAttentionStage::MIXED) && !params.is_dynamic()) {
            paged_attention_aligned_seq_len = get_aligned_seq_len(params, stage);
        }
        const auto partition_size = static_cast<int64_t>(get_partitioning_params(params, desc->v_head_size, stage).second);
        const auto num_of_partitions = std::max<int64_t>(static_cast<int64_t>(ceil_div(paged_attention_aligned_seq_len, partition_size)), 1);

        const auto target_seq_len = std::max<int64_t>(paged_attention_aligned_seq_len, 1);
        const auto indexes_buf_size = static_cast<int64_t>(ceil_div(target_seq_len, target_seq_len_block_size));

        const bool lockable = true;
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);  // 0
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);  // 1
        internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);  // 2

        const auto& input = params.input_layouts[0];
        const int64_t total_tokens = input.get_partial_shape()[0].get_length();
        // if (total_tokens == 1) {
        //     std::cout << "batch = 1, paged_attention_aligned_seq_len = " << paged_attention_aligned_seq_len << std::endl;
        // }

        auto buf_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * num_of_partitions);
        auto tmp_out_elements_count = static_cast<int64_t>(total_tokens * desc->heads_num * desc->v_head_size * num_of_partitions);

        const bool has_scores_output = params.output_layouts.size() > 1;
        if (has_scores_output) {
            const auto& past_lens = params.input_layouts[5];
            auto subsequences_number = past_lens.get_partial_shape()[0].get_length();
            auto rtp = static_cast<PagedAttentionRuntimeParams*>(m_rt_params.get());
            auto tokens_number = desc->has_score_aggregation ? rtp->paged_attention_snap_kv_tokens : subsequences_number;
            auto softmax_buf_elements_count = static_cast<int64_t>(tokens_number * desc->heads_num * num_of_partitions * partition_size);

            // Softmax intermediate output
            internal_buffers.emplace_back(softmax_buf_elements_count, softmax_accumulator_type);  // 3
            // Precalculated accumulated sequence length offsets for each subsequence
            internal_buffers.emplace_back(subsequences_number, indexes_dt, lockable);  // 4

            if (desc->has_score_aggregation) {
                // Cumulative window size sum buffer
                internal_buffers.emplace_back(subsequences_number + 1, indexes_dt, lockable);  // 5
            }

            if (stage == PagedAttentionStage::PREFILL) {
                // Recalculate buf_size as in case of PREFILL stage it's not needed to allocate buffer per each input token
                buf_elements_count = tokens_number * static_cast<int64_t>(desc->heads_num) * num_of_partitions;

                // Intermediate tmp output buffer is not used for PREFILL stage
                tmp_out_elements_count = get_element_size(softmax_accumulator_type);
            }
        }

        internal_buffers.emplace_back(buf_elements_count, softmax_accumulator_type);      // 5: softmax exp_sums
        internal_buffers.emplace_back(buf_elements_count, softmax_accumulator_type);      // 6: softmax max_logits
        internal_buffers.emplace_back(tmp_out_elements_count, softmax_accumulator_type);  // 7: intermediate output

        const auto multi_tokens_mode = stage == PagedAttentionStage::MIXED;
        if (multi_tokens_mode) {
            internal_buffers.emplace_back(total_tokens, softmax_accumulator_type, lockable);  // 9
        }

        auto can_use_micro_sdpa = supports_micro_sdpa(params);
        if (can_use_micro_sdpa) {
            const auto wg_tile_q = get_micro_tile_qsize(pa_sdpa_micro->kd);
            const auto target_seq_len = std::max(paged_attention_aligned_seq_len, static_cast<int64_t>(1));
            const auto indexes_buf_size = ceil_div(target_seq_len, wg_tile_q) * 2;
            internal_buffers.emplace_back(indexes_buf_size, indexes_dt, lockable);
        }

        // std::cout << "[GPU] Paged Attention internal buffers: " << internal_buffers.size() << std::endl;
        return internal_buffers;
    }

    static void prepare_internal_buffers(paged_attention_inst& instance, const PagedAttentionStage& stage, bool use_micro_sdpa, size_t query_block_size) {
        const auto& desc = instance.get_impl_params()->typed_desc<paged_attention>();
        const bool has_scores_output = desc->has_scores_output();
        const bool has_score_aggregation = desc->has_score_aggregation;

        if ((stage == PagedAttentionStage::UNKNOWN) || (stage == PagedAttentionStage::GENERATE && !has_scores_output))
            return;

        auto& stream = instance.get_network().get_stream();
        const auto past_lens_mem = instance.past_lens_memory_ptr();
        const auto subsequence_begins_mem = instance.subsequence_begins_memory_ptr();
        const auto& intermediates_memories = instance.get_intermediates_memories();
        mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, stream);
        mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, stream);
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> subsequence_offsets_lock = nullptr;
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> cumulative_window_size_sum_lock = nullptr;

        if (has_scores_output) {
            const size_t subsequence_offsets_idx = 4;

            OPENVINO_ASSERT(intermediates_memories.size() > subsequence_offsets_idx,
                            "[GPU] Unexpected number of intermediates buffers for Paged Attention for scores output calculation");

            const auto& subsequence_offsets_mem = intermediates_memories[subsequence_offsets_idx];
            subsequence_offsets_lock.reset(new mem_lock<int32_t, mem_lock_type::write>(subsequence_offsets_mem, stream));

            if (has_score_aggregation) {
                const size_t cumulative_window_size_sum_idx = 5;
                OPENVINO_ASSERT(intermediates_memories.size() > cumulative_window_size_sum_idx,
                                "[GPU] Unexpected number of intermediates buffers for Paged Attention for scores aggregation");

                auto cumulative_window_size_sum_mem = intermediates_memories[cumulative_window_size_sum_idx];
                cumulative_window_size_sum_lock.reset(new mem_lock<int32_t, mem_lock_type::write>(cumulative_window_size_sum_mem, stream));

                mem_lock<int32_t, mem_lock_type::read> score_aggregation_mem_lock(instance.score_aggregation_memory_ptr(), stream);

                // Transform window sizes to cumulative buffer for offsets precalculation
                // For example:
                // Original score aggregation buffer content: {4, 2, 1, 1}
                // Cumulative window size sum buffer content: {0, 4, 6, 7, 8}
                size_t cumulative_sum = 0;
                cumulative_window_size_sum_lock->operator[](0) = static_cast<int32_t>(cumulative_sum);
                for (size_t i = 0; i < score_aggregation_mem_lock.size(); i++) {
                    cumulative_sum += score_aggregation_mem_lock[i];
                    cumulative_window_size_sum_lock->operator[](i + 1) = static_cast<int32_t>(cumulative_sum);
                }
            }
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

        const auto& blocks_indexes_start_mem = intermediates_memories[blocks_indexes_start_idx];
        const auto& blocks_indexes_end_mem = intermediates_memories[blocks_indexes_end_idx];
        const auto& blocked_gws_subseq_mapping_mem = intermediates_memories[blocked_gws_subseq_mapping_idx];

        OPENVINO_ASSERT(subsequence_begins_mem->get_layout().data_type == data_types::i32);

        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_start_lock(blocks_indexes_start_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocks_indexes_end_lock(blocks_indexes_end_mem, stream);
        mem_lock<int32_t, mem_lock_type::write> blocked_gws_subseq_mapping_mem_lock(blocked_gws_subseq_mapping_mem, stream);
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> sequential_gws_subseq_mapping_lock = nullptr;
        std::unique_ptr<mem_lock<int32_t, mem_lock_type::write>> micro_sdpa_block_starts_and_gws_mapping_lock = nullptr;

        if (stage == PagedAttentionStage::MIXED) {
            size_t sequential_gws_subseq_mapping_idx = 6;
            if (has_score_aggregation) {
                sequential_gws_subseq_mapping_idx = 9;
            } else if (has_scores_output) {
                sequential_gws_subseq_mapping_idx = 8;
            }

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
        // size_t query_block_size = get_query_block_size(stage, use_micro_sdpa);
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

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedAttentionOptImpl>(this);
    }
};

std::unique_ptr<primitive_impl> PagedAttentionOpt::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<paged_attention>());
    return std::make_unique<PagedAttentionOptImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedAttentionOptImpl)
