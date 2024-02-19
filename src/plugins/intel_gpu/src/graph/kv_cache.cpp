// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/optionals.hpp"
#include "kv_cache_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(kv_cache)

kv_cache_inst::typed_primitive_inst(network& network, const kv_cache_node& node) :
    parent{network, node, false},
    memory_state::variable{node.get_primitive()->variable_info.variable_id} {
    kv_cache_id = network.get_kv_cache_ids().size();
}

layout kv_cache_inst::calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> kv_cache_inst::calc_output_layouts(kv_cache_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<kv_cache>();

    ov::intel_gpu::op::KVCache op;
    op.set_output_size(desc->num_outputs);
    op.set_concat_axis(desc->concat_axis);
    op.set_gather_axis(desc->gather_axis);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(), impl_param.get_input_layout(1).get<ShapeType>()};
    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    const std::map<size_t, size_t> ports_map = {{0, 0}, {1, 2}};

    std::vector<layout> out_layouts;
    for (size_t i = 0; i < desc->num_outputs; i++) {
        auto out_type = desc->output_data_types[i].value_or(impl_param.get_input_layout(ports_map.at(i)).data_type);
        out_layouts.push_back(layout(output_shapes[i], out_type, impl_param.get_output_layout(i).format));
    }

    return out_layouts;
}

template std::vector<layout> kv_cache_inst::calc_output_layouts<ov::PartialShape>(kv_cache_node const& node, const kernel_impl_params& impl_param);

std::string kv_cache_inst::to_string(const kv_cache_node& node) {
    auto node_info = node.desc_to_json();
    json_composite kv_cache_info;
    kv_cache_info.add("input id", node.input().id());
    kv_cache_info.add("variable id", node.get_primitive()->variable_info.variable_id);
    kv_cache_info.add("variable shape", node.get_primitive()->variable_info.data_shape);
    kv_cache_info.add("variable type", node.get_primitive()->variable_info.data_type);
    kv_cache_info.add("concat axis", node.get_primitive()->concat_axis);
    kv_cache_info.add("gather axis", node.get_primitive()->gather_axis);
    kv_cache_info.add("indirect", node.get_primitive()->indirect);
    node_info->add("kv_cache info", kv_cache_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

int32_t kv_cache_inst::get_prealloc_iter_num() {
    return 128 + kv_cache_id % 64;
}

void kv_cache_inst::update_shape_info_tensor(const kernel_impl_params& params) {
    mem_lock<int32_t> lock(_shape_info_memory, _network.get_stream());
    auto shape_info_ptr = lock.data();
    size_t offset = 0;

    std::vector<std::pair<layout, layout>> input_layouts; // [kv_state, kv_new_token, [beam_idx, bt_past]]
    for (size_t i = 0; i < _node->get_dependencies().size(); i++) {
        const auto& node_in_lay = _node->get_input_layout(i);
        const auto& runtime_in_lay = params.input_layouts[i];

        input_layouts.emplace_back(runtime_in_lay, node_in_lay);
    }

    if (params.typed_desc<kv_cache>()->indirect) {
        auto& var = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCache&>(get_network().get_variable(variable_id()));
        const auto& bt_state = var.get_beam_table_state();
        auto bt_layout = bt_state->get_layout();
        if (bt_layout.is_dynamic()) {
            auto bt_shape = bt_layout.get_partial_shape();
            for (auto& d : bt_shape) {
                if (d.is_dynamic())
                    d = 0;
            }
            bt_layout.set_partial_shape(bt_shape);
        }
        input_layouts.emplace_back(bt_layout, bt_state->get_initial_layout());
    }

    for (size_t i = 0; i < input_layouts.size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for input[" << i << "]" << std::endl;
        fill_shape_info_data(input_layouts[i].first, input_layouts[i].second, shape_info_ptr, offset);
    }

    for (size_t i = 0; i < _node->get_output_layouts().size(); i++) {
        GPU_DEBUG_TRACE_DETAIL << id() << " : update shape_info for output[" << i << "]" << std::endl;
        const auto& node_out_lay = _node->get_output_layout(i);
        const auto& runtime_out_lay = params.output_layouts[i];
        fill_shape_info_data(runtime_out_lay, node_out_lay, shape_info_ptr, offset);
    }
}

} // namespace cldnn
