// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_expert)

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout moe_expert_inst::calc_output_layout(moe_expert_node const& /* node */, kernel_impl_params const& impl_param) {
    if (impl_param.memory_deps.empty()) {
        return impl_param.input_layouts[0];
    } else {
        return impl_param.memory_deps.at(0)->get_layout();
    }
}

bool moe_expert_inst::get_pred_from_memory(memory::ptr mem, stream& stream, size_t expert_no) {
    const auto& shape = mem->get_layout().get_shape();
    auto offset = expert_no * shape[1] * shape[2];
    mem_lock<int32_t, mem_lock_type::read> lock_data{mem, stream};
    auto p = lock_data.data() + offset;
    for (size_t i = 0; i < shape[1] * shape[2]; i++)
        if (p[i])
            return true;
    return false;
}

template<typename ShapeType>
std::vector<layout> moe_expert_inst::calc_output_layouts(moe_expert_node const& /* node */, kernel_impl_params const& impl_param) {
    if (impl_param.memory_deps.empty()) {
        return {impl_param.input_layouts[0]};
    } else {
        return {impl_param.memory_deps.at(0)->get_layout()};
    }
}

template std::vector<layout> moe_expert_inst::calc_output_layouts<ov::PartialShape>(moe_expert_node const& node, const kernel_impl_params& impl_param);

std::string moe_expert_inst::to_string(moe_expert_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite moe_expert_info;

    node_info->add("moe_expert info", moe_expert_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
moe_expert primitive is reusing memory with the input.
*/
moe_expert_inst::typed_primitive_inst(network& network, moe_expert_node const& node)
    : parent(network, node),
      _net(network::allocate_network(network.get_stream_ptr(), node.get_branch().inner_program)) {
    this->set_inner_networks({_net});
}

void moe_expert_inst::update_output_layout() {
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }
        auto dep_id = _node->get_dependency(i).id();

        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
    }
    _impl_params->memory_deps = memory_deps;

    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    if (new_layouts.empty()) {
        auto new_layout = _node->type()->calc_output_layout(*_node, *_impl_params);
        new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(0), new_layout.data_padding);
        _impl_params->output_layouts[0] = new_layout;
    } else {
        for (size_t i = 0; i != new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(i), new_layout.data_padding);
            _impl_params->output_layouts[i] = new_layout;
        }
    }
}

void moe_expert_inst::postprocess_output_memory(network::ptr executed_net, cldnn::moe_expert::branch& branch) {
    _outputs.resize(outputs_memory_count());
    auto out_mem_idx = 0;
    auto inner_out_id = executed_net->get_output_ids()[0];

    auto mem_ptr = executed_net->get_output_memory(inner_out_id);
    if (mem_ptr) {
        auto layout = _impl_params->get_output_layout(out_mem_idx);
        GPU_DEBUG_LOG << "Reshape output from " << mem_ptr->get_layout().to_short_string()
                    << " to " << layout.to_short_string() << std::endl;
        // Preallocation logic may allocate more memory than actually produced on current iteration, so we need to adjust output buffers layout
        mem_ptr = get_network().get_engine().reinterpret_buffer(*mem_ptr, layout);
    }

    _outputs[out_mem_idx] = mem_ptr;
    if (mem_ptr)
        GPU_DEBUG_LOG << "Inner net - Outputs[" << out_mem_idx << "]" << mem_ptr->get_layout().to_short_string() << std::endl;
}
}  // namespace cldnn
