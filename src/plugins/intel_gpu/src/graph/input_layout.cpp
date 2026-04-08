// Copyright (C) 2018-2026 Intel Corporation
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

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(input_layout)

input_layout_node::typed_program_node(const std::shared_ptr<input_layout> dprim, program& prog) : parent(dprim, prog) {
    can_share_buffer(false);
}

input_layout_inst::typed_primitive_inst(network& network, const input_layout_node& node)
    // allocate memory for scalars as they assume the memory is available and not lazily allocated...
    : parent(network, node, /*allocate_mem*/ !node.is_dynamic() && node.get_output_layout().count() <= 1) {
    _has_valid_input = false;                          // by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

event::ptr input_layout_inst::set_data(memory::ptr mem, bool need_to_check_memory_to_set) {
    auto ol = get_node_output_layout();

    bool empty_mem = mem->size() == 0 && (ol.is_dynamic() || ol.count() == 0);
    if (!empty_mem && need_to_check_memory_to_set) {
        check_memory_compatibility(*mem, ol);
    }

    event::ptr ev = nullptr;
    auto& engine = get_network().get_engine();
    auto& stream = get_network().get_stream();

    // Allow to set dummy simple_attached_memory empty tensor as network input
    if (mem->is_allocated_by(engine) || mem->get_layout().count() == 0) {
        if (_outputs.empty()) {
            _outputs.resize(1);
        }
        _outputs[0] = mem;
    } else {
        if (_outputs.empty()) {
            _outputs.resize(1);
        }
        // Only allocate if needed and not already large enough
        if (!_outputs[0] || _outputs[0]->size() < mem->size()) {
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
    if (_impl_params->get_output_layout() != mem_layout || get_node().is_in_shape_of_subgraph()) {
        set_flag(ExecutionFlags::SHAPE_CHANGED);
    }
    _impl_params->output_layouts[0] = mem_layout;
}

std::string input_layout_inst::to_string(const input_layout_node& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
