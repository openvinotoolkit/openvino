// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "json_object.h"
#include "msda_inst.h"
#include "primitive_type_base.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(msda);

namespace {
// Overload << operator for vectors
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}
};  // namespace

std::string msda_inst::to_string(const msda_node& node) {
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

msda_inst::typed_primitive_inst(network& network, const msda_node& node) : parent(network, node) {}

std::vector<int32_t> msda_inst::get_mask_seqlens_from_memory() const {
    auto& stream = get_network().get_stream();

    memory::ptr cu_seqlens_mem = dep_memory_ptr(3);
    const auto& cu_seqlens_dep = dependencies()[3];
    const auto cu_seqlens_impl = cu_seqlens_dep.first->get_impl_params();
    const auto& cu_seqlens_layout = cu_seqlens_impl->get_output_layout(cu_seqlens_dep.second);

    // TODO: wait for attention_mask_seqlen input only
    stream.finish();

    const auto& shape = cu_seqlens_layout.get_shape();
    OPENVINO_ASSERT(shape.size() == 1, "cu_seqlens should be a vector, but it is ", shape);
    std::vector<int32_t> buf(shape[0]);
    cu_seqlens_mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

    GPU_DEBUG_TRACE_DETAIL << " get_mask_seqlens_from_memory " << cu_seqlens_mem->buffer_ptr() << " : " << buf << std::endl;

    return buf;
}

std::vector<int32_t> msda_inst::get_mask_seqlens_from_memory2(memory::ptr cu_seqlens_mem, stream& stream) {
    // TODO: wait for attention_mask_seqlen input only
    stream.finish();

    std::vector<int32_t> buf(cu_seqlens_mem->count());
    cu_seqlens_mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

    GPU_DEBUG_TRACE_DETAIL << " get_mask_seqlens_from_memory2 " << cu_seqlens_mem->buffer_ptr() << " : " << buf << std::endl;

    return buf;
}

}  // namespace cldnn