// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(lora);

std::string lora_inst::to_string(const lora_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite lora_info;
    lora_info.add("main_input", node.input().id());
    lora_info.add("lora_input", node.input(1).id());
    lora_info.add("state_a", node.input(2).id());
    lora_info.add("state_alpha", node.input(3).id());
    lora_info.add("state_b", node.input(4).id());
    lora_info.add("lora_count", (node.get_inputs_count() - 2ul) / 3ul);

    node_info->add("lora_info", lora_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

void lora_inst::on_execute() {
    update_output_memory();
}

void lora_inst::update_output_memory() {
    bool is_empty_lora = true;
    for (size_t i = 2; i < _impl_params->desc->input_size(); ++i) {
        is_empty_lora &= _impl_params->get_input_layout(i).count() == 0;
    }

    if (!is_empty_lora)
       return;

    if (static_cast<bool>(_outputs[0]) && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    GPU_DEBUG_TRACE_DETAIL << id() << " : update_output_memory with mem of input " << get_node().get_dependency(0).id()
                           << " : " << input_memory_ptr()->buffer_ptr() << std::endl;

    // Can_be_optimized nodes are allocating from memory_pool too. In this case,
    // we need release the legacy output memory from memory pool explicitly.
    if (static_cast<bool>(_outputs[0]) &&
        _node->get_program().get_config().get_enable_memory_pool()) {
        _network.get_memory_pool().release_memory(_outputs[0].get(), _node->get_unique_id(), _node->id(), _network.get_id());
    }

    _outputs[0] = input_memory_ptr();
    _mem_allocated = false;
}

lora_inst::typed_primitive_inst(network& network, const lora_node& node) : parent(network, node) {}

}  // namespace cldnn
