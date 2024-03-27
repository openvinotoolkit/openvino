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


layout paged_attention_inst::calc_output_layout(const paged_attention_node& node, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template<typename ShapeType>
std::vector<layout> paged_attention_inst::calc_output_layouts(paged_attention_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto out_layout = impl_param.get_input_layout(0);

    return {out_layout};
}

template std::vector<layout> paged_attention_inst::calc_output_layouts<ov::PartialShape>(paged_attention_node const& node, const kernel_impl_params& impl_param);

std::string paged_attention_inst::to_string(const paged_attention_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite custom_gpu_prim_info;
    node_info->add("paged attention primitive info", custom_gpu_prim_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void paged_attention_inst::on_execute() {
    GPU_DEBUG_TRACE_DETAIL << "paged_attention_inst::on_execute\n";

    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    GPU_DEBUG_TRACE_DETAIL << "Internal buffers layouts: " << ibuf_layouts.size() << "\n";
    GPU_DEBUG_TRACE_DETAIL << "Internal buffers: " << _intermediates_memory.size() << "\n";

    if (_intermediates_memory.size() < 6) {
        GPU_DEBUG_TRACE_DETAIL << "Number of intermediate buffers is less than expected\n";
        return;
    }


    for (size_t i = 0; i < ibuf_layouts.size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << "Layout " << ibuf_layouts[i].to_short_string() << " mem = " << _intermediates_memory[i]->buffer_ptr() << ", " << _intermediates_memory[i]->get_layout().to_short_string() << "\n";
    }

    auto print_arr = [&](cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read>& vec, size_t max_len) {
        std::stringstream ss;
        for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
            ss << vec[i] << ", ";
        }
        GPU_DEBUG_TRACE_DETAIL << "subsequence_begins from graph (len=" << vec.size() << ") content: " << ss.str() << "\n";
    };

    auto print_arr2 = [&](std::vector<int32_t> vec, size_t max_len, std::string name) {
        std::stringstream ss;
        for (size_t i = 0; i < std::min(max_len, vec.size()); i++) {
            ss << vec[i] << ", ";
        }
        GPU_DEBUG_TRACE_DETAIL << name << " content: " << ss.str() << "\n";
    };


    auto& stream = get_network().get_stream();
    auto& mem_deps = _impl_params->memory_deps;
    const auto subsequence_begins_mem = mem_deps.at(6);
    mem_lock<int32_t, mem_lock_type::read> subsequence_begins_mem_lock(subsequence_begins_mem, stream);

    std::vector<int32_t> blocks_indexes_start;
    std::vector<int32_t> blocks_indexes_end;
    std::vector<int32_t> gws_seq_indexes_correspondence;

    for (size_t i = 0; i < subsequence_begins_mem_lock.size() - 1; i++) {
        const auto seq_start = subsequence_begins_mem_lock[i];
        const auto seq_end = subsequence_begins_mem_lock[i + 1];
        const auto seq_length = seq_end - seq_start;

        // TODO: get block size from the impl
        const auto block_size = 16;
        for (int32_t j = 0; j < seq_length; j += block_size) {
            auto block_start_pos = subsequence_begins_mem_lock[i] + j;
            blocks_indexes_start.push_back(block_start_pos);

            auto block_end_pos = std::min(block_start_pos + block_size, seq_end);
            blocks_indexes_end.push_back(block_end_pos);

            gws_seq_indexes_correspondence.push_back(i);
        }
    }

    print_arr(subsequence_begins_mem_lock, subsequence_begins_mem_lock.size());
    print_arr2(blocks_indexes_start, blocks_indexes_start.size(), "blocks_indexes_start");
    print_arr2(blocks_indexes_end, blocks_indexes_end.size(), "blocks_indexes_end");
    print_arr2(gws_seq_indexes_correspondence, gws_seq_indexes_correspondence.size(), "gws_seq_indexes_correspondence");

    auto buf_size = _intermediates_memory[3]->get_layout().bytes_count();
    mem_lock<int32_t, mem_lock_type::write> blocks_indexes_start_lock(_intermediates_memory[3], stream);
    mem_lock<int32_t, mem_lock_type::write> blocks_indexes_end_lock(_intermediates_memory[4], stream);
    mem_lock<int32_t, mem_lock_type::write> gws_seq_indexes_correspondence_lock(_intermediates_memory[5], stream);
    std::memcpy(blocks_indexes_start_lock.data(), blocks_indexes_start.data(), buf_size);
    std::memcpy(blocks_indexes_end_lock.data(), blocks_indexes_end.data(), buf_size);
    std::memcpy(gws_seq_indexes_correspondence_lock.data(), gws_seq_indexes_correspondence.data(), buf_size);

}

void paged_attention_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    parent::update_shape_info_tensor(params);
}

paged_attention_inst::typed_primitive_inst(network& network, const paged_attention_node& node)
    : parent(network, node) {}
}  // namespace cldnn
