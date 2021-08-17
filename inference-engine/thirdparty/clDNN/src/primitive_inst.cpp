// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "generic_layer_inst.h"
#include "input_layout_inst.h"
#include "arg_max_min_inst.h"
#include "fused_conv_eltwise_inst.h"

#include "cldnn/graph/network.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/memory.hpp"

#include "cldnn/runtime/error_handler.hpp"
#include "cldnn/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>
#include <stack>
#include <vector>
#include <memory>
#include <algorithm>

namespace cldnn {

bool is_user_cpu(const program_node* user) {
    if (user->can_be_optimized()) {
        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_user_cpu(u)) {
                return true;
            }
        }
        return false;
    }
    return user->get_selected_impl()->is_cpu();
}

bool is_any_user_cpu(const std::list<const program_node*>& users) {
    for (const auto& user : users) {
        if (is_user_cpu(user))
            return true;
    }
    return false;
}

uint32_t primitive_inst::get_network_id() const { return _network.get_id(); }

void primitive_inst::check_memory_to_set(const memory& mem, const layout& layout) const {
    CLDNN_ERROR_LAYOUT_MISMATCH("network layout",
        "set memory layout",
        mem.get_layout(),
        "expected layout",
        layout,
        "");

    // check shared image/buffer compatibility, if applicable
    auto params = mem.get_internal_params();
    if (params.mem_type != shared_mem_type::shared_mem_empty) {
        if (!mem.is_allocated_by(get_network().get_engine())) {
            CLDNN_ERROR_MESSAGE(_node.id(), "Memory object is not suitable");
        }

        switch (params.mem_type) {
        case shared_mem_type::shared_mem_vasurface:
        case shared_mem_type::shared_mem_image:
            if (!layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output image instead of a buffer");
            break;
        case shared_mem_type::shared_mem_buffer:
        case shared_mem_type::shared_mem_dxbuffer:
            if (layout.format.is_image_2d())
                CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output buffer instead of an image");
            break;
        case shared_mem_type::shared_mem_usm:
            break;
        default:
            CLDNN_ERROR_MESSAGE(_node.id(), "Attempt to set user-supplied input or output memory of unknown/invalid type");
            break;
        }
    }
}

void primitive_inst::set_output_memory(memory::ptr mem_new, bool check) {
    auto& eng = _network.get_engine();
    // skip all the buzz if no action actually required
    if (eng.is_the_same_buffer(*mem_new, *_output)) {
        return;
    }

    auto ol = _node.get_output_layout();

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (_node.is_constant()) {
        mem_new->copy_from(_network.get_stream(), *_output);
    } else {
        _output = mem_new;
    }
}

event::ptr primitive_inst::execute(const std::vector<event::ptr>& events) {
    const auto primitive_id = id();
    CLDNN_ERROR_BOOL(primitive_id,
                     "Invalid/unset input",
                     !_has_valid_input,
                     "Cannot execute primitive " + primitive_id + " with invalid/unset input");
    on_execute();

    if (_exec_deps.empty())
        return _impl->execute(events, *this);

    std::vector<event::ptr> dependencies;
    dependencies.reserve(_exec_deps.size());
    for (auto& input : _exec_deps) {
        auto id = input->id();
        try {
            // if the requested event does not exits it means that it has not been executed, so the processing_order is
            // wrong or synchronization failed.
            auto ev = get_network().get_primitive_event(id);
            dependencies.emplace_back(ev);
        } catch (const std::out_of_range& oor) {
            std::string temp = std::string("internal CLDNN error: execution order corrupted.") + std::string("\n") +
                               std::string(oor.what() + std::string("\n"));
            CLDNN_ERROR_MESSAGE(id, temp);
        }
    }
    return _impl->execute(dependencies, *this);
}

void primitive_inst::set_arguments() {
    const auto primitive_id = id();
    CLDNN_ERROR_BOOL(primitive_id,
                     "Invalid/unset input",
                     !_has_valid_input,
                     "Cannot set arguments for primitive " + primitive_id + " with invalid/unset input");

    _impl->set_arguments(*this);
}

void primitive_inst::build_deps() {
    if (_deps.empty() && !_node.get_dependencies().empty()) {
        _deps = _network.get_primitives(_node.get_dependencies());
        _exec_deps = build_exec_deps(_deps);
    }
}

primitive_inst::primitive_inst(network& network, program_node const& node, bool allocate_memory)
    : _network(network), _node(node), _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr), _output(), _output_changed(false) {
    if (allocate_memory) {
        // In case when output is mutable_data primitive, and other users dependencies are only used for
        // suychronization, The output memory of such primitive will be fused with mutable_data
        auto users = node.get_users();
        auto user_count = users.size();
        uint32_t mutable_data_count = 0;
        for (auto& user : users) {
            // Get mutable_data nodes count from nodes users
            if (user->is_type<mutable_data>()) {
                mutable_data_count++;
            } else if (user->is_type<fused_conv_eltwise>()) {
                if (!user->as<fused_conv_eltwise>().get_users().empty() &&
                    (*user->as<fused_conv_eltwise>().get_users().begin())->is_type<mutable_data>()) {
                    if (user->as<fused_conv_eltwise>().get_dependency(1).id() == node.id()) {
                        user_count--;
                    }
                }
            }
        }

        // TODO: Remove WA for arg_max_min node.
        // For now it's required to handle the case when only second output of TopK primitive is used in plugin,
        // but kernels always write both outputs to the same memory object which leads to wrong result.
        if (user_count == 1 && mutable_data_count == 1 && !node.is_type<arg_max_min>()) {
            for (auto& user : node.get_users())
                if (user->is_type<mutable_data>())
                    _output = user->as<mutable_data>().get_attached_memory_ptr();
        } else {
            _output = allocate_output();
        }
    }
}

void primitive_inst::allocate_internal_buffers(void) {
    if (_impl == nullptr) return;
    const auto& ibuf_info = _impl->get_internal_buffer_info();
    if (ibuf_info.sizes.empty()) return;

    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "[" << _node.id() << ": internal buf]" << std::endl;
    }

    auto& engine = get_network().get_engine();
    auto alloc_type = allocation_type::usm_host;

    for (size_t i = 0; i < dependencies().size(); ++i) {
        if (dep_memory(i).get_allocation_type() == allocation_type::usm_device) {
            std::cout << "allocating to usm_device" << std::endl;
            alloc_type = allocation_type::usm_device;
            break;
        }
    }

    for (auto size : ibuf_info.sizes) {
        const auto bpp = data_type_traits::size_of(ibuf_info.dtype);
        layout expected_layout = {ibuf_info.dtype,
                                  format::bfyx,  // simple linear format (flatten to x channel)
                                  {1, 1, 1, (tensor::value_type)(size / bpp)}};
        _intermediates_memory.push_back(engine.allocate_memory(expected_layout, alloc_type));
    }
}

memory::ptr primitive_inst::allocate_output() {
    auto layout = _node.get_output_layout();
    auto& engine = get_network().get_engine();

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    auto use_lockable_memory = _node.is_output() || _node.get_selected_impl()->is_cpu()
                               || std::any_of(_node.get_users().begin(), _node.get_users().end(),
                                              [](const program_node* n) {
                                     return n->get_selected_impl()->is_cpu() || is_any_user_cpu(n->get_users());
                                  }) || !engine.supports_allocation(allocation_type::usm_device);

    allocation_type alloc_type = use_lockable_memory ?
                                 engine.get_lockable_preffered_memory_allocation_type(layout.format.is_image_2d())
                                                     : allocation_type::usm_device;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    if (!_network.is_internal() && (_node.can_be_optimized() || _node.is_type<generic_layer>())) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return _network.get_memory_from_pool(layout,
                                             _node.id(),
                                             _node.get_memory_dependencies(),
                                             alloc_type,
                                             false);
    } else if (_network.is_internal() && _node.is_output() && _node.is_type<generic_layer>() &&
               engine.supports_allocation(allocation_type::usm_device)) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return engine.allocate_memory(layout, allocation_type::usm_device, false);
    } else if (_network.is_internal() && !_node.is_output() && _node.is_type<input_layout>()) {
        // Skip memory reset for input_layout primitives, since data will be copied from cldnn::data primitive
        // or just reuse primitive's memory
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": constant]" << std::endl;
        }
        return engine.allocate_memory(layout, alloc_type, false);
    } else if (_network.is_internal() || (!_node.can_share_buffer()) || _node.can_be_optimized() || _node.is_output()) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return engine.allocate_memory(layout, alloc_type);
    } else {
        return _network.get_memory_from_pool(layout,
                                             _node.id(),
                                             _node.get_memory_dependencies(),
                                             alloc_type,
                                             true);
    }
}

std::vector<std::shared_ptr<primitive_inst>> primitive_inst::build_exec_deps(
    std::vector<std::shared_ptr<primitive_inst>> const& deps) {
    std::vector<std::shared_ptr<primitive_inst>> exec_deps;
    exec_deps.reserve(deps.size());
    for (auto& dep : deps)
        if (dep->get_impl() != nullptr)
            exec_deps.push_back(dep);

    return exec_deps;
}

std::string primitive_inst::generic_to_string(program_node const& node, const char* type_name) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_inputs;

    for (size_t i = 0; i < node.get_dependencies().size(); ++i) {
        auto& in = node.get_dependency(i);
        ss_inputs << in.id();
        ss_inputs << ", count: " << in.get_output_layout().count();
        i != (node.get_dependencies().size() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    json_composite generic_info;
    generic_info.add("type_name", type_name);
    generic_info.add("deps count", node.get_dependencies().size());
    generic_info.add("deps", ss_inputs.str());

    node_info->add("generic info", generic_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
