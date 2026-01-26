// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "paged_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>
#include <sstream>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_attention)

paged_attention_node::typed_program_node(const std::shared_ptr<paged_attention> prim, program& prog)
    : parent(prim, prog) {
    can_share_internal_buffer(false);
}

layout paged_attention_inst::calc_output_layout(const paged_attention_node& /*node*/, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    const auto& q_layout = impl_param.get_input_layout(cldnn::paged_attention::PagedAttentionInputIdx::QUERY);
    const auto& desc = impl_param.typed_desc<paged_attention>();
    auto data_layout = q_layout;

    if (desc->k_head_size != desc->v_head_size) {
        auto data_shape = { q_layout.get_partial_shape()[0],
                            ov::Dimension(desc->heads_num * desc->v_head_size) };

        data_layout = data_layout.clone_with_other_shape(data_shape);
    }

    data_layout.data_padding = padding();

    const auto key_cache_idx = cldnn::paged_attention::PagedAttentionInputIdx::KEY_CACHE;
    const auto& key_cache_ps = impl_param.get_input_layout(key_cache_idx).get_partial_shape();
    const auto& key_cache_quant_mode = impl_param.get_program().get_config().get_key_cache_quant_mode();
    bool key_cache_compressed = impl_param.get_input_layout(key_cache_idx).data_type == ov::element::i8 ||
                                impl_param.get_input_layout(key_cache_idx).data_type == ov::element::u8;
    auto expected_block_size = desc->has_xattention ? paged_attention::block_size_xattn : paged_attention::block_size;
    if (key_cache_compressed && key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
        expected_block_size += 4;
    }
    OPENVINO_ASSERT((key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) == desc->is_key_by_channel,
                     "[GPU] Paged Attention key cache quantization mode mismatch: prim.is_key_by_channel : ",
                     desc->is_key_by_channel, " but exec_config : ", impl_param.get_program().get_config().get_key_cache_quant_mode());

    const auto block_size_idx = desc->has_xattention ? 2 : 3;
    bool valid_block_size = key_cache_ps.is_dynamic() ||
                            (key_cache_ps[block_size_idx].get_length() == static_cast<ov::Dimension::value_type>(expected_block_size));
    OPENVINO_ASSERT(valid_block_size, "[GPU] Incorrect block size for Paged Attention operation for key cache quant mode "
                    , key_cache_quant_mode, ". Expected ", expected_block_size, ", but got ", key_cache_ps[block_size_idx].get_length());

    // TODO: as a preview feature, only single sequence is supported so far. Will remove this check once
    // full function ready in near future.
    if (desc->has_xattention) {
        const auto& subseq_begins_ps = impl_param.get_input_layout(PagedAttentionInputIdx::SUBSEQUENCE_BEGINS).get_partial_shape();
        bool valid_subseq_count = subseq_begins_ps.is_dynamic() ||
                                (subseq_begins_ps[0].get_length() == static_cast<ov::Dimension::value_type>(2));
        if (!valid_subseq_count)
            OPENVINO_THROW("[GPU] Unexpected sub sequences count for XAttention. Got ", subseq_begins_ps[0].get_length() - 1);
    }

    std::vector<layout> output_layouts{ data_layout };

    if (desc->has_scores_output()) {
        const auto past_lens_idx = cldnn::paged_attention::PagedAttentionInputIdx::PAST_LENS;
        const auto output_dt = data_layout.data_type;
        if (impl_param.get_input_layout(past_lens_idx).is_static()) {
            const auto& memory_deps = impl_param.memory_deps;
            const auto past_lens_mem = memory_deps.at(past_lens_idx);
            mem_lock<int32_t, mem_lock_type::read> past_lens_mem_lock(past_lens_mem, *impl_param.strm);

            long int total_size = 0;
            for (size_t i = 0; i < past_lens_mem_lock.size(); i++) {
                total_size += past_lens_mem_lock[i];
            }

            total_size += static_cast<long int>(data_layout.get_shape()[0]);

            output_layouts.push_back(layout{ov::PartialShape{total_size}, output_dt, format::bfyx});
        } else {
            output_layouts.push_back(layout{ov::PartialShape::dynamic(1), output_dt, format::bfyx});
        }
        if (desc->has_adaptive_rkv) {
            // expecting 3 outputs, 2nd as above, 3rd - Adaptive R-KV block diversity
            const auto evictable_sizes_idx = cldnn::paged_attention::PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES;
            const auto output_dt = data_layout.data_type;
            if (impl_param.get_input_layout(past_lens_idx).is_static()) {
                size_t num_elements_in_output = 0;
                const auto& memory_deps = impl_param.memory_deps;
                const auto evictable_sizes_mem = memory_deps.at(evictable_sizes_idx);
                mem_lock<int32_t, mem_lock_type::read> evictable_sizes_mem_lock(evictable_sizes_mem, *impl_param.strm);

                for (size_t i = 0; i < evictable_sizes_mem_lock.size(); i++) {
                    size_t evictable_size = evictable_sizes_mem_lock[i];
                    num_elements_in_output += evictable_size * evictable_size / desc->block_size;
                }

                output_layouts.push_back(layout{ov::PartialShape{static_cast<long int>(num_elements_in_output)}, output_dt, format::bfyx});

            } else {
                output_layouts.push_back(layout{ov::PartialShape::dynamic(1), output_dt, format::bfyx});
            }
        }
    }

    return output_layouts;
}

template std::vector<layout>
    paged_attention_inst::calc_output_layouts<ov::PartialShape>(paged_attention_node const& node, const kernel_impl_params& impl_param);

std::string paged_attention_inst::to_string(const paged_attention_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite paged_attention_info;
    paged_attention_info.add("paged_attention_block_size", desc->block_size);
    paged_attention_info.add("k_head_size", desc->k_head_size);
    paged_attention_info.add("v_head_size", desc->v_head_size);
    paged_attention_info.add("heads_num", desc->heads_num);
    paged_attention_info.add("kv_heads_num", desc->kv_heads_num);
    paged_attention_info.add("has_alibi", desc->has_alibi);
    paged_attention_info.add("has_rotated_blocks", desc->has_rotated_blocks);
    paged_attention_info.add("sliding_window", desc->sliding_window);
    paged_attention_info.add("score_output", desc->has_scores_output());
    paged_attention_info.add("has_score_aggregation", desc->has_score_aggregation);
    paged_attention_info.add("has_xattention", desc->has_xattention);
    paged_attention_info.add("has_sink_input", desc->has_sink_input);
    paged_attention_info.add("has_adaptive_rkv", desc->has_adaptive_rkv);
    paged_attention_info.add("scale", desc->scale_val.value_or(1.0f));
    paged_attention_info.add("is_key_by_channel", desc->is_key_by_channel);
    paged_attention_info.add("key_cache_dt", node.get_input_layout(cldnn::paged_attention::PagedAttentionInputIdx::KEY_CACHE).data_type);
    paged_attention_info.add("value_cache_dt", node.get_input_layout(cldnn::paged_attention::PagedAttentionInputIdx::VALUE_CACHE).data_type);
    node_info->add("paged_attention primitive info", paged_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

paged_attention_inst::typed_primitive_inst(network& network, const paged_attention_node& node)
    : parent(network, node) {
    const auto desc = node.get_primitive();

    const auto k_head_size = desc->k_head_size;
    const auto v_head_size = desc->v_head_size;
    const auto heads_num = desc->heads_num;
    const auto kv_heads_num = desc->kv_heads_num;
    const auto pa_block_size = desc->block_size;

    if (desc->has_alibi) {
        const auto alibi_input_idx = PagedAttentionInputIdx::ALIBI;
        const auto alibi_layout = node.get_input_layout(alibi_input_idx);
        OPENVINO_ASSERT(heads_num == alibi_layout.count(), "[GPU] ALiBi layout count must match heads_num");
    }

    OPENVINO_ASSERT(heads_num % kv_heads_num == 0, "[GPU] heads_num must be divisible by kv_heads_num");
    if (!desc->has_xattention) {
        OPENVINO_ASSERT(k_head_size % pa_block_size == 0, "[GPU] k_head_size must be divisible by block_size");
        OPENVINO_ASSERT(v_head_size % pa_block_size == 0, "[GPU] v_head_size must be divisible by block_size");
    }
}
}  // namespace cldnn
