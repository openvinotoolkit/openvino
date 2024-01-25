// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/runtime/optionals.hpp"
#include "kv_cache_inst.h"
#include "read_value_inst.h"
#include "gather_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(kv_cache)

kv_cache_inst::typed_primitive_inst(network& network, const kv_cache_node& node) :
    parent{network, node, false},
    memory_state::variable{node.get_primitive()->variable_info.variable_id} {
}

layout kv_cache_inst::calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

template<typename ShapeType>
std::vector<layout> kv_cache_inst::calc_output_layouts(kv_cache_node const& /*node*/, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<kv_cache>();
    auto output_data_type = desc->output_data_types[0].value_or(impl_param.get_input_layout().data_type);

    ov::intel_gpu::op::KVCache op;
    op.set_concat_axis(desc->concat_axis);
    op.set_gather_axis(desc->gather_axis);

    std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>(), impl_param.get_input_layout(1).get<ShapeType>()};

    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    return {layout({output_shapes[0], output_data_type, impl_param.get_output_layout().format})};
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
    node_info->add("kv_cache info", kv_cache_info);
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void kv_cache_inst::post_realloc_optimization(const layout& allocated_layout) {
    auto desc = _node->as<kv_cache>().get_primitive();
    auto& variable = get_network().get_variable(desc->variable_info.variable_id);
    auto present_layout = _impl_params->output_layouts[0];
    const auto& sequence_axis = desc->concat_axis;
    auto sequence_axis_legacy =
        kv_cache_inst::get_sequence_axis_legacy(sequence_axis, present_layout.get_partial_shape().size());
    GPU_DEBUG_TRACE_DETAIL << id() << " is kv_cache => set the variable with newly allocated output memory"
                           << std::endl;
    bool axis_is_outer_most = true;
    for (int64_t dim = 0; dim < sequence_axis; ++dim) {
        if (present_layout.get_shape()[dim] > 1) {
            axis_is_outer_most = false;
            break;
        }
    }
    if (present_layout.data_padding.get_dynamic_pad_dims().sizes()[sequence_axis_legacy] == 1) {
        // Apply padding of variable to make it be optimized in the next iteration
        auto max_pad = kv_cache_inst::get_max_pad(
            present_layout,
            allocated_layout.get_buffer_size().count(),
            sequence_axis_legacy,
            "present_layout");
        if (max_pad > 0) {
            kv_cache_inst::update_pad(present_layout, max_pad, sequence_axis_legacy);
            if (!axis_is_outer_most) {
                GPU_DEBUG_TRACE_DETAIL << id() << ": Update impl with new output padding" << std::endl;
                set_shape_change();
                _impl_params->output_layouts[0] = present_layout;
                update_impl();
            }
            GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                                   << "'s memory with allocated kv cache output: " << present_layout.to_short_string()
                                   << " is_set  = " << variable.is_set() << std::endl;
            variable.set_memory(_outputs[0], present_layout);
            _impl_params->_can_be_optimized = true;
            // No need to copy, still it can be optimized
            GPU_DEBUG_TRACE_DETAIL << id() << ": Set can_be_optimized = true " << std::endl;
            {
                // Garbage collection of kv cache meories :
                // Once the corresponding kv cache's execution is done, the input mems are no
                // longer needed and can be released.
                GPU_DEBUG_TRACE_DETAIL << ": Check releasable kv cache memories" << std::endl;
                std::vector<primitive_id> mem_deps_eol;
                for (auto kms : _network.get_kv_cache_mem_deps()) {
                    const auto kv_cache_id = kms.first;
                    auto queue_type = get_network().get_stream().get_queue_type();
                    if (queue_type == QueueTypes::in_order ||
                        (_network.has_event(kv_cache_id) && _network.get_primitive_event(kv_cache_id)->is_set())) {
                        for (auto mem_deps : kms.second) {
                            mem_deps_eol.push_back(mem_deps);
                        }
                    }
                }
                for (auto mem_dep : mem_deps_eol) {
                    auto mem_dep_inst = _network.get_primitive(mem_dep);
                    GPU_DEBUG_TRACE_DETAIL << "Release output memory of " << mem_dep_inst->id() << ": "
                              << ((mem_dep_inst->_outputs.size() > 0 && mem_dep_inst->_outputs[0]) ? mem_dep_inst->_outputs[0]->buffer_ptr() : " 0x0")
                              << std::endl;

                    mem_dep_inst->_outputs[0] = nullptr;
                }
            }
            {
                // Add mem_deps for current kv_cache op for future release
                GPU_DEBUG_TRACE_DETAIL << "Record kv cache mem deps for future garbage collection " << id() << ": " << std::endl;
                if (_deps[0].first->get_node().is_type<gather>() && _deps[0].first->can_be_optimized()) {
                    _network.add_kv_cache_mem_deps(id(), _deps[0].first->id());
                    GPU_DEBUG_TRACE_DETAIL << id() << " can clear " << _deps[0].first->id() << "'s mem" << std::endl;
                    if (_deps[0].first->_deps[0].first->get_node().is_type<read_value>() &&
                        _deps[0].first->_deps[0].first->can_be_optimized()) {
                        _network.add_kv_cache_mem_deps(id(), _deps[0].first->_deps[0].first->id());
                        GPU_DEBUG_TRACE_DETAIL << id() << " can clear " << _deps[0].first->_deps[0].first->id() << "'s mem" << std::endl;
                    }
                }
            }
        } else {
            GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                                   << "'s layout with allocated kv cache output: " << present_layout.to_short_string()
                                   << " (is_set  = " << variable.is_set() << ") " << std::endl;
            variable.set_layout(present_layout);
        }
    } else {
        GPU_DEBUG_TRACE_DETAIL << id() << ": Update variable " << variable.get_name()
                               << "'s layout with allocated kv cache output: " << present_layout.to_short_string()
                               << " (is_set  = " << variable.is_set() << ") " << std::endl;
        variable.set_layout(present_layout);
    }
}

} // namespace cldnn
