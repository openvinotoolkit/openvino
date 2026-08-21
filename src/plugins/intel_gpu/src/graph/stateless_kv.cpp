// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/stateless_kv.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "primitive_type_base.h"
#include "stateless_kv_inst.h"
#include <sstream>
#include <json_object.h>
#include "utils.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(stateless_kv)

stateless_kv_inst::typed_primitive_inst(network& network, const stateless_kv_node& node) : parent{network, node, false} {
    update_output_memory();
}

std::optional<int64_t> stateless_kv_inst::compute_update_offset(const kernel_impl_params& impl_param, const stateless_kv& desc) {
    const auto mem_dep_it = impl_param.memory_deps.find(2);
    if (mem_dep_it == impl_param.memory_deps.end())
        return {};

    const auto& seq_len_mem = mem_dep_it->second;
    const auto seq_len_layout = seq_len_mem->get_layout();
    if (seq_len_layout.count() == 0)
        return {};

    OPENVINO_ASSERT(seq_len_layout.count() == 1);
    cldnn::mem_lock<uint8_t, mem_lock_type::read> seq_len_mem_lock(seq_len_mem, impl_param.get_stream());
    auto seq_len_tensor = make_tensor(seq_len_layout, seq_len_mem_lock.data());
    const auto seq_len = ov::get_tensor_data_as<int64_t>(seq_len_tensor)[0];

    const auto& past_layout = impl_param.get_input_layout(0);
    const auto past_shape = past_layout.get_partial_shape();
    const auto past_sequence_axis = ov::util::normalize(desc.concat_axis, past_shape.size());
    OPENVINO_ASSERT(past_sequence_axis >= 0);
    const auto& past_dim = past_shape[static_cast<size_t>(past_sequence_axis)];
    OPENVINO_ASSERT(past_dim.is_static());

    const auto& current_layout = impl_param.get_input_layout(1);
    const auto current_shape = current_layout.get_partial_shape();
    const auto current_sequence_axis = ov::util::normalize(desc.concat_axis, current_shape.size());
    OPENVINO_ASSERT(current_sequence_axis >= 0);
    const auto& current_dim = current_shape[static_cast<size_t>(current_sequence_axis)];
    OPENVINO_ASSERT(current_dim.is_static());

    int64_t past_seq_len = 0;
    int64_t present_seq_len = 0;
    if (desc.is_present_len) {
        present_seq_len = seq_len;
        past_seq_len = present_seq_len - current_dim.get_length();
    } else {
        past_seq_len = seq_len;
        present_seq_len = past_seq_len + current_dim.get_length();
    }
    GPU_DEBUG_TRACE_DETAIL << desc.id << " : " << (desc.is_present_len ? "present" : "past") << "_len[" << seq_len << "] cur_len[" << current_dim.get_length()
                           << "] past_tensor[" << past_dim.get_length() << "] " << (present_seq_len <= past_dim.get_length() ? "update" : "concat")
                           << std::endl;
    OPENVINO_ASSERT(past_seq_len >= 0, "[GPU] new_token_data shouldn't exceed present_seq_length");

    return past_seq_len;
}

void stateless_kv_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }
    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;

    const auto node_input_layouts = get_node().get_shape_info_input_layouts();
    for (size_t i = 0; i < get_node().get_dependencies().size(); ++i) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        fill_shape_info_data(params.input_layouts[i], node_input_layouts[i], shape_info_ptr, offset);
    }

    for (size_t i = 0; i < get_node().get_output_layouts().size(); ++i) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        fill_shape_info_data(params.output_layouts[i], get_node().get_output_layout(i), shape_info_ptr, offset);
    }
}

layout stateless_kv_inst::calc_output_layout(const stateless_kv_node& node, kernel_impl_params const& impl_param) {
    return calc_output_layouts<ov::PartialShape>(node, impl_param).front();
}

template<typename ShapeType>
std::vector<layout> stateless_kv_inst::calc_output_layouts(const stateless_kv_node& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<stateless_kv>();

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(), impl_param.get_input_layout(1).get<ShapeType>()};
    const auto concat_axis = ov::util::normalize(desc->concat_axis, input_shapes[0].size());
    OPENVINO_ASSERT(concat_axis >= 0 && static_cast<size_t>(concat_axis) < input_shapes[0].size(), "[GPU] concat_axis exceed range");
    GPU_DEBUG_TRACE_DETAIL << desc->id << " : input[" << input_shapes[0] << "][" << input_shapes[1] << "]" << std::endl;

    ov::intel_gpu::op::StatelessKV op;
    op.set_output_size(2);
    op.set_concat_axis(concat_axis);
    op.set_is_present_len(desc->is_present_len);
    op.set_update_offset(stateless_kv_inst::compute_update_offset(impl_param, *desc));

    auto output_shapes = shape_infer(&op, input_shapes);
    int64_t padding = 0;
    if (output_shapes[0][concat_axis].is_static() && output_shapes[1][concat_axis].is_static()) {
        padding = output_shapes[0][concat_axis].get_length() - output_shapes[1][concat_axis].get_length();
        OPENVINO_ASSERT(padding >= 0);
    }
    GPU_DEBUG_TRACE_DETAIL << desc->id << " : output[" << output_shapes[0] << "][" << output_shapes[1] << "] padding: " << padding << std::endl;

    std::vector<layout> out_layouts;
    out_layouts.emplace_back(output_shapes[0], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(0).format);
    out_layouts.emplace_back(output_shapes[1], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(1).format);
    padding::DynamicDimsMask seq_padding_info;
    seq_padding_info[concat_axis] = 1;
    out_layouts[1].data_padding._dynamic_dims_mask = seq_padding_info;
    out_layouts[1].data_padding._upper_size[concat_axis] = padding;

    return out_layouts;
}

template std::vector<layout> stateless_kv_inst::calc_output_layouts<ov::PartialShape>(stateless_kv_node const& node, const kernel_impl_params& impl_param);

std::string stateless_kv_inst::to_string(const stateless_kv_node& node) {
    auto node_info = node.desc_to_json();
    json_composite stateless_kv_info;
    stateless_kv_info.add("input id", node.input().id());
    stateless_kv_info.add("concat axis", node.get_primitive()->concat_axis);
    stateless_kv_info.add("is present len", node.get_primitive()->is_present_len);
    node_info->add("stateless_kv info", stateless_kv_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void stateless_kv_inst::update_output_memory() {
    if (_node != nullptr)
        build_deps();

    if (input_memory_ptr() == nullptr || _outputs.empty())
        return;

    OPENVINO_ASSERT(_outputs.size() == 2);
    if (!_outputs[0])
        return;

    auto& engine = _network.get_engine();
    OPENVINO_ASSERT(_outputs[1], "[GPU] output1 should be available when output0 is present");
    OPENVINO_ASSERT(engine.is_the_same_buffer(output_memory(0), output_memory(1)), "[GPU] output1 should be same tensor with output0");
    m_is_inplace = engine.is_the_same_buffer(output_memory(), input_memory());
    GPU_DEBUG_TRACE_DETAIL << id() << ": update_output_memory in[" << input_memory().get_layout().to_short_string() << "] out["
                           << output_memory(0).get_layout().to_short_string() << "][" << output_memory(1).get_layout().to_short_string() << "] inplace["
                           << (m_is_inplace ? 'Y' : 'N') << "]" << std::endl;
    _mem_allocated = false;
}

void stateless_kv_inst::on_execute() {
    update_output_memory();
    set_arguments();
}

} // namespace cldnn
