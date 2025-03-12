// Copyright (C) 2018-2025 Intel Corporation
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

layout paged_attention_inst::calc_output_layout(const paged_attention_node& /*node*/, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto data_layout = impl_param.get_input_layout(0);
    data_layout.data_padding = padding();

    const auto& key_cache_ps = impl_param.get_input_layout(3).get_partial_shape();
    bool valid_block_size = key_cache_ps[3].is_dynamic() || key_cache_ps[3].get_length() == paged_attention::block_size;
    OPENVINO_ASSERT(valid_block_size, "[GPU] Incorrect block size for Paged Attention operation. "
                                      "Expected ", paged_attention::block_size, ", but got ", key_cache_ps[3].get_length());

    std::vector<layout> output_layouts{ data_layout };

    const auto& desc = impl_param.typed_desc<paged_attention>();
    if (desc->has_scores_output()) {
        const auto past_lens_idx = 5;
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
    paged_attention_info.add("head_size", desc->head_size);
    paged_attention_info.add("heads_num", desc->heads_num);
    paged_attention_info.add("kv_heads_num", desc->kv_heads_num);
    paged_attention_info.add("scale", desc->scale_val.value_or(1.0f));
    paged_attention_info.add("has_alibi", desc->has_alibi);
    paged_attention_info.add("has_rotated_blocks", desc->has_rotated_blocks);
    paged_attention_info.add("key_cache_dt", node.get_input_layout(3).data_type);
    paged_attention_info.add("value_cache_dt", node.get_input_layout(4).data_type);
    node_info->add("paged_attention primitive info", paged_attention_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
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
