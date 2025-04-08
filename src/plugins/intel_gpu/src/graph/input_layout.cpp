// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "input_layout_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace {
bool has_optimized_users(cldnn::input_layout_node const& node) {
    for (auto& user : node.get_users()) {
        if (user->can_be_optimized()) {
            return true;
        }
    }

    return false;
}
}  // namespace

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(input_layout)

input_layout_node::typed_program_node(const std::shared_ptr<input_layout> dprim, program& prog)
    : parent(dprim, prog) {
    can_share_buffer(false);
}

input_layout_inst::typed_primitive_inst(network& network, input_layout_node const& node)
    : parent(network, node, !node.is_dynamic() && (!network.is_internal() || has_optimized_users(node))) {
    _has_valid_input = false;  // by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

event::ptr input_layout_inst::set_data(memory::ptr mem, bool need_to_check_memory_to_set) {
    auto ol = get_node_output_layout();

    bool empty_mem = mem->size() == 0 && (ol.is_dynamic() || ol.count() == 0);
    if (!empty_mem && need_to_check_memory_to_set) {
        check_memory_to_set(*mem, ol);
    }

    event::ptr ev = nullptr;
    auto& engine = get_network().get_engine();
    auto& stream = get_network().get_stream();

    // Allow to set dummy simple_attached_memory empty tensor as network input
    if (mem->is_allocated_by(engine) || mem->get_layout().count() == 0) {
        OPENVINO_ASSERT(!_outputs.empty(), "[GPU] Can't set data for empty input memory");
        _outputs[0] = mem;
    } else {
        if (_outputs.empty() || !_outputs[0]) {
            _outputs.resize(1);
            _outputs[0] = engine.allocate_memory(mem->get_layout(), engine.get_preferred_memory_allocation_type(), false);
        }

        if (ol.is_dynamic() && _outputs[0]->size() < mem->size()) {
            _outputs[0] = engine.allocate_memory(mem->get_layout(), engine.get_preferred_memory_allocation_type(), false);
        }
        mem_lock<uint8_t> src(mem, stream);
        ev = _outputs[0]->copy_from(stream, src.data(), false);
    }
    _has_valid_input = true;
    return ev;
}

void input_layout_inst::update_shape() {
    OPENVINO_ASSERT(!_outputs.empty() && _outputs[0] != nullptr, "[GPU] input memory is not set");
    auto mem_layout = _outputs[0]->get_layout();
    // Set SHAPE_CHANGED flag if the actual data layout has changed, or if the node is included
    // into shape_of subgraph to trigger proper shape_of subgraph shape recalculation
    if (_impl_params->get_output_layout() != mem_layout || _node->is_in_shape_of_subgraph()) {
        set_flag(ExecutionFlags::SHAPE_CHANGED);
    }
    _impl_params->output_layouts[0] = mem_layout;
}

std::string input_layout_inst::to_string(input_layout_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
