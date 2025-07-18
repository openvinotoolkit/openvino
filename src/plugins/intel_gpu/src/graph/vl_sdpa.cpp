// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vl_sdpa_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(vl_sdpa);

std::string vl_sdpa_inst::to_string(const vl_sdpa_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite vlsdpa_info;
    vlsdpa_info.add("q", node.input(0).id());
    vlsdpa_info.add("k", node.input(1).id());
    vlsdpa_info.add("v", node.input(2).id());
    vlsdpa_info.add("cu_seq_lens", node.input(3).id());

    node_info->add("vlsdpa_info", vlsdpa_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

vl_sdpa_inst::typed_primitive_inst(network& network, const vl_sdpa_node& node) : parent(network, node) {}

std::vector<int32_t> vl_sdpa_inst::get_mask_seqlens_from_memory(memory::ptr cu_seqlens_mem, stream& stream) {
    // TODO: wait for attention_mask_seqlen input only
    stream.finish();

    std::vector<int32_t> buf(cu_seqlens_mem->count());
    cu_seqlens_mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

    GPU_DEBUG_TRACE_DETAIL << " get_mask_seqlens_from_memory " << cu_seqlens_mem->buffer_ptr() << std::endl;

    return buf;
}

}  // namespace cldnn
