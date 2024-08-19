// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "paged_attention_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

#include <string>
#include <sstream>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(paged_attention)

constexpr size_t paged_attention::block_size;

bool is_prefill_stage(const kernel_impl_params& impl_param) {
    const auto& query_shape = impl_param.get_input_layout(0).get_partial_shape();
    const auto& past_lens_shape = impl_param.get_input_layout(5).get_partial_shape();

    if (query_shape.is_static() && past_lens_shape.is_static())
        return query_shape[0].get_length() != past_lens_shape[0].get_length();

    return false;
}

layout paged_attention_inst::calc_output_layout(const paged_attention_node& /*node*/, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    const auto& key_cache_ps = impl_param.get_input_layout(3).get_partial_shape();
    bool valid_block_size = key_cache_ps[3].is_dynamic() || key_cache_ps[3].get_length() == paged_attention::block_size;
    OPENVINO_ASSERT(valid_block_size, "[GPU] Incorrect block size for Paged Attention operation. "
                                      "Expected ", paged_attention::block_size, ", but got ", key_cache_ps[3].get_length());

    return {out_layout};
}

template std::vector<layout>
    paged_attention_inst::calc_output_layouts<ov::PartialShape>(paged_attention_node const& node, const kernel_impl_params& impl_param);

std::string paged_attention_inst::to_string(const paged_attention_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite paged_attention_info;
    paged_attention_info.add("paged_attention_block_size", desc->block_size);
    paged_attention_info.add("head_size", desc->head_size);
    paged_attention_info.add("heads_num", desc->heads_num);
    paged_attention_info.add("kv_heads_num", desc->kv_heads_num);
    paged_attention_info.add("scale", desc->scale_val.value_or(1.0f));
    paged_attention_info.add("has_alibi", desc->has_alibi);
    node_info->add("paged_attention primitive info", paged_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void paged_attention_inst::on_execute() {
    if (!is_prefill_stage(*_impl_params))
        return;

    OPENVINO_ASSERT(_intermediates_memory.size() == 3, "Unexpected number of intermediates buffers for Paged Attention at prefill stage");

    const auto blocks_indexes_start_idx = 0;
    const auto blocks_indexes_end_idx = 1;
    const auto gws_seq_indexes_correspondence_idx = 2;

    auto subsequence_begins_mem = subsequence_begins_memory_ptr();
    auto blocks_indexes_start_mem = _intermediates_memory[blocks_indexes_start_idx];
    auto blocks_indexes_end_mem = _intermediates_memory[blocks_indexes_end_idx];
    auto gws_seq_indexes_correspondence_mem = _intermediates_memory[gws_seq_indexes_correspondence_idx];

    OPENVINO_ASSERT(subsequence_begins_mem->get_layout().data_type == data_types::i32);

    auto& stream = get_network().get_stream();
    mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, stream);
    mem_lock<int32_t, mem_lock_type::write> blocks_indexes_start_lock(blocks_indexes_start_mem, stream);
    mem_lock<int32_t, mem_lock_type::write> blocks_indexes_end_lock(blocks_indexes_end_mem, stream);
    mem_lock<int32_t, mem_lock_type::write> gws_seq_indexes_correspondence_lock(gws_seq_indexes_correspondence_mem, stream);

    size_t index = 0;
    const auto target_seq_len_block_size = 16; // TODO: Get block size from the impl
    for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
        const auto seq_start = subsequence_begins_mem_lock[i];
        const auto seq_end = subsequence_begins_mem_lock[i + 1];
        const auto seq_length = seq_end - seq_start;

        for (int32_t j = 0; j < seq_length; j += target_seq_len_block_size) {
            auto block_start_pos = subsequence_begins_mem_lock[i] + j;
            auto block_end_pos = std::min(block_start_pos + target_seq_len_block_size, seq_end);

            blocks_indexes_start_lock[index] = block_start_pos;
            blocks_indexes_end_lock[index] = block_end_pos;
            gws_seq_indexes_correspondence_lock[index] = static_cast<int32_t>(i);

            index++;
        }
    }
}

void paged_attention_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    parent::update_shape_info_tensor(params);
}

paged_attention_inst::typed_primitive_inst(network& network, const paged_attention_node& node)
    : parent(network, node) {
    const auto desc = node.get_primitive();

    const auto head_size = desc->head_size;
    const auto heads_num = desc->heads_num;
    const auto kv_heads_num = desc->kv_heads_num;
    const auto pa_block_size = desc->block_size;

    if (desc->has_alibi) {
        const auto alibi_input_idx = 11;
        const auto alibi_layout = node.get_input_layout(alibi_input_idx);
        OPENVINO_ASSERT(heads_num == alibi_layout.count());
    }

    OPENVINO_ASSERT(heads_num % kv_heads_num == 0);
    OPENVINO_ASSERT(head_size % pa_block_size == 0);
}
}  // namespace cldnn
