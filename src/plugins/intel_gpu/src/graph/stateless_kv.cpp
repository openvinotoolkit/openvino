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

#include <algorithm>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(stateless_kv)

stateless_kv_inst::typed_primitive_inst(network& network, const stateless_kv_node& node) : parent{network, node, false} {
    update_output_memory();
}

static layout get_input_view(const layout& input_layout, int64_t concat_axis, int64_t source_offset, int64_t copy_len) {
    OPENVINO_ASSERT(concat_axis >= 0 && concat_axis < static_cast<int64_t>(input_layout.get_rank()));
    auto view_layout = input_layout;
    auto input_shape = view_layout.get_partial_shape();
    const auto input_len = input_shape[concat_axis].get_length();
    OPENVINO_ASSERT(source_offset >= 0 && copy_len >= 0 && source_offset + copy_len <= input_len);
    input_shape[concat_axis] = copy_len;
    view_layout.set_partial_shape(input_shape);
    view_layout.data_padding._lower_size[concat_axis] += source_offset;
    view_layout.data_padding._upper_size[concat_axis] += input_len - source_offset - copy_len;
    return view_layout;
}

layout stateless_kv_runtime_state::get_past_to_sdpa_input_layout(const layout& input_layout, int64_t concat_axis) const {
    return get_input_view(input_layout, concat_axis, get_sdpa_in_past_offset(), get_new_in_sdpa_offset());
}

layout stateless_kv_runtime_state::get_sdpa_to_present_input_layout(const layout& input_layout, int64_t concat_axis) const {
    return get_input_view(input_layout, concat_axis, -get_sdpa_in_present_offset(), get_present_output_len());
}

int64_t stateless_kv_inst::get_swa_sequence_length(int64_t abs_seq_len, int64_t cache_capacity, int64_t window_size) {
    OPENVINO_ASSERT(abs_seq_len >= 0, "[GPU] stateless_kv absolute sequence length must be non-negative");
    OPENVINO_ASSERT(window_size > 0 && window_size <= cache_capacity,
                    "[GPU] stateless_kv window_size must be within cache capacity");
    if (abs_seq_len <= cache_capacity)
        return abs_seq_len;

    const auto rolling_gap = cache_capacity - window_size + 1;
    const auto overflow = abs_seq_len - cache_capacity;
    const auto rolling_count = 1 + (overflow - 1) / rolling_gap;
    return abs_seq_len - rolling_gap * rolling_count;
}

std::optional<stateless_kv_runtime_state> stateless_kv_inst::compute_runtime_state(const kernel_impl_params& impl_param,
                                                                                   const stateless_kv& desc) {
    const auto mem_dep_it = impl_param.memory_deps.find(2);
    if (mem_dep_it == impl_param.memory_deps.end())
        return std::nullopt;

    const auto& seq_len_mem = mem_dep_it->second;
    const auto seq_len_layout = seq_len_mem->get_layout();
    if (seq_len_layout.count() == 0)
        return std::nullopt;

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
    if (desc.is_seq_len_present_len) {
        present_seq_len = seq_len;
        past_seq_len = present_seq_len - current_dim.get_length();
    } else {
        past_seq_len = seq_len;
        present_seq_len = past_seq_len + current_dim.get_length();
    }
    GPU_DEBUG_TRACE_DETAIL << desc.id << " : " << (desc.is_seq_len_present_len ? "present" : "past") << "_len[" << seq_len << "] cur_len["
                           << current_dim.get_length() << "] past_tensor[" << past_dim.get_length() << "] "
                           << (present_seq_len <= past_dim.get_length() ? "update" : "concat") << std::endl;
    OPENVINO_ASSERT(past_seq_len >= 0, "[GPU] new_token_data shouldn't exceed present_seq_length");

    stateless_kv_runtime_state state;
    state.past_seq_len = past_seq_len;
    state.present_seq_len = present_seq_len;
    state.new_token_len = current_dim.get_length();

    if (desc.window_size > 0) {
        OPENVINO_ASSERT(desc.input.size() == 3, "[GPU] stateless_kv doesn't support window_size together with pos_idx");
        const auto cache_capacity = past_dim.get_length();
        state.past_front_idx = past_seq_len - get_swa_sequence_length(past_seq_len, cache_capacity, desc.window_size);
        state.present_front_idx = present_seq_len - get_swa_sequence_length(present_seq_len, cache_capacity, desc.window_size);
        state.sdpa_front_idx = std::max<int64_t>(0, past_seq_len - desc.window_size + 1);
    }

    OPENVINO_ASSERT(state.get_past_output_len() >= 0 && state.get_present_output_len() >= 0);
    OPENVINO_ASSERT(state.get_new_in_sdpa_offset() >= 0);
    OPENVINO_ASSERT(state.get_sdpa_len(desc.window_size) ==
                    (desc.window_size > 0
                         ? std::min(state.present_seq_len, state.new_token_len + desc.window_size - 1)
                         : state.present_seq_len));

    return state;
}

void stateless_kv_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    if (!_shape_info_memory) {
        allocate_shape_info_memory();
    }
    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto* shape_info_ptr = lock.data();
    size_t offset = 0;

    const auto desc = get_typed_desc<stateless_kv>();
    const auto node_input_layouts = get_node().get_shape_info_input_layouts();
    for (size_t i = 0; i < get_node().get_dependencies().size(); ++i) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        if (i == 0 && desc->input.size() == 3) {
            OPENVINO_ASSERT(m_runtime_state.has_value(), "[GPU] stateless_kv runtime state wasn't initialized during shape update");
            const auto& state = *m_runtime_state;
            const auto copy_len = state.is_rolling() ? 0 : state.get_new_in_present_offset();
            const auto past_view = get_input_view(params.get_input_layout(0), desc->concat_axis, 0, copy_len);
            fill_shape_info_data(past_view, node_input_layouts[i], shape_info_ptr, offset);
        } else {
            fill_shape_info_data(params.input_layouts[i], node_input_layouts[i], shape_info_ptr, offset);
        }
    }

    if (desc->window_size > 0) {
        OPENVINO_ASSERT(m_runtime_state.has_value(), "[GPU] stateless_kv SWA runtime state wasn't initialized during shape update");
        OPENVINO_ASSERT(desc->input.size() == 3);
        OPENVINO_ASSERT(node_input_layouts.size() == 5);
        const auto& state = *m_runtime_state;
        if (state.is_rolling()) {
            auto past_view = state.get_past_to_sdpa_input_layout(params.get_input_layout(0), desc->concat_axis);
            GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for SWA sdpa's past input" << std::endl;
            fill_shape_info_data(past_view, node_input_layouts[3], shape_info_ptr, offset);
            auto sdpa_view = state.get_sdpa_to_present_input_layout(params.get_output_layout(1), desc->concat_axis);
            GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for SWA present's sdpa input" << std::endl;
            fill_shape_info_data(sdpa_view, node_input_layouts[4], shape_info_ptr, offset);
        }
    }

    offset = get_node().get_total_shape_info_input_size();
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
    op.set_is_seq_len_present_len(desc->is_seq_len_present_len);
    op.set_window_size(desc->window_size);
    const auto runtime_state = stateless_kv_inst::compute_runtime_state(impl_param, *desc);
    if (runtime_state) {
        op.set_update_offset(runtime_state->get_new_in_present_offset());
        op.set_is_rolling(runtime_state->is_rolling());
        op.set_past_seq_len(runtime_state->past_seq_len);
    }

    auto output_shapes = shape_infer(&op, input_shapes);
    int64_t lower_padding = 0;
    int64_t upper_padding = 0;
    if (runtime_state && !runtime_state->is_rolling() && output_shapes[0][concat_axis].is_static()) {
        lower_padding = runtime_state->get_sdpa_in_present_offset();
        upper_padding = output_shapes[0][concat_axis].get_length() - output_shapes[1][concat_axis].get_length() - lower_padding;
        OPENVINO_ASSERT(lower_padding >= 0 && upper_padding >= 0);
    }
    GPU_DEBUG_TRACE_DETAIL << desc->id << " : output[" << output_shapes[0] << "][" << output_shapes[1] << "] padding: [" << lower_padding << ", "
                           << upper_padding << "]" << std::endl;

    std::vector<layout> out_layouts;
    out_layouts.emplace_back(output_shapes[0], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(0).format);
    out_layouts.emplace_back(output_shapes[1], impl_param.get_input_layout(0).data_type, impl_param.get_output_layout(1).format);
    padding::DynamicDimsMask seq_padding_info;
    seq_padding_info[concat_axis] = 1;
    out_layouts[1].data_padding._dynamic_dims_mask = seq_padding_info;
    out_layouts[1].data_padding._lower_size[concat_axis] = lower_padding;
    out_layouts[1].data_padding._upper_size[concat_axis] = upper_padding;

    return out_layouts;
}

template std::vector<layout> stateless_kv_inst::calc_output_layouts<ov::PartialShape>(stateless_kv_node const& node, const kernel_impl_params& impl_param);

std::string stateless_kv_inst::to_string(const stateless_kv_node& node) {
    auto node_info = node.desc_to_json();
    json_composite stateless_kv_info;
    stateless_kv_info.add("input id", node.input().id());
    stateless_kv_info.add("concat axis", node.get_primitive()->concat_axis);
    stateless_kv_info.add("is present len", node.get_primitive()->is_seq_len_present_len);
    stateless_kv_info.add("window size", node.get_primitive()->window_size);
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
    m_is_inplace = engine.is_the_same_buffer(output_memory(), input_memory());
    GPU_DEBUG_TRACE_DETAIL << id() << ": update_output_memory in[" << input_memory().get_layout().to_short_string() << "] out["
                           << output_memory(0).get_layout().to_short_string() << "][" << output_memory(1).get_layout().to_short_string() << "] inplace["
                           << (m_is_inplace ? 'Y' : 'N') << "]" << std::endl;
}

void stateless_kv_inst::on_execute() {
    update_output_memory();
    get_impl()->update(*this, *get_impl_params());
    set_arguments();
}

} // namespace cldnn
