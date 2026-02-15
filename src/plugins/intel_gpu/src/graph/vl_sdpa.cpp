// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vl_sdpa_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include "intel_gpu/runtime/utils.hpp"

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

void vl_sdpa_inst::get_mask_seqlens_from_memory(std::vector<int32_t>& cu_seqlens) const {
    cldnn::stream& stream = get_network().get_stream();
    stream.finish();

    const auto cu_seqlens_mem = cu_seqlens_memory_ptr();

    auto size = cu_seqlens_mem->count();
    OPENVINO_ASSERT(cu_seqlens_mem->get_layout().data_type == ov::element::i32 && size > 0);

    cu_seqlens.resize(size);
    cu_seqlens_mem->copy_to(stream, cu_seqlens.data(), 0, 0, size * sizeof(int32_t), true);

    GPU_DEBUG_TRACE_DETAIL << " get_mask_seqlens_from_memory " << cu_seqlens << std::endl;
}

}  // namespace cldnn
