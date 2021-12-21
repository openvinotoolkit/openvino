// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#if 0 // TODO(taylor)
#include "mutable_data_inst.h"
#include "generic_layer_inst.h"
#endif
#include "input_layout_inst.h"
#include "arg_max_min_inst.h"
#if 0 // TODO(taylor)
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#endif

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include <string>
#include <stack>
#include <vector>
#include <memory>
#include <algorithm>

namespace {

bool is_optimized_output_user(const program_node* user) {
    if (user->can_be_optimized()) {
        if (user->is_output())
            return true;

        auto users = user->get_users();
        for (const auto& u : users) {
            if (is_optimized_output_user(u)) {
                return true;
            }
        }
        return false;
    }
    return false;
}

bool is_output_buffer(const program_node& node) {
    if (node.is_output())
        return true;

    // Try to recursively find any optimized out user which is also network output
    for (const auto& user : node.get_users()) {
        if (is_optimized_output_user(user)) {
            return true;
        }
    }

    return false;
}

}  // namespace

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
    // skip all the buzz if no action actually require
    bool all_same_buffer = true;
    for (auto a : _outputs) {
        all_same_buffer &= eng.is_the_same_buffer(*mem_new, *a);
    }
    if (all_same_buffer) return;

    // TODO(taylor) to support multiple output
    auto ol = _node.get_output_layout(0);

    if (check)
        check_memory_to_set(*mem_new, ol);

    if (_node.is_constant()) {
        mem_new->copy_from(_network.get_stream(), *_outputs[0]);
    } else {
        _outputs[0] = mem_new;
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
        auto id = input.first->id();
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
#if 0 // TODO(taylor)
        std::vector<program_node*> dep_nodes;
        std::vector<int32_t> dep_idxes;
        for (const auto& n : _node.get_dependencies()) {
            dep_nodes.push_back(n.first);
            dep_idxes.push_back(n.second);
        }
        //_deps = _network.get_primitives(dep_nodes);
        auto prims = _network.get_primitives(dep_nodes);
        for (int32_t i = 0; i < prims.size(); ++i) {
            _deps.push_back(std::make_pair(prims[i], dep_idxes[i]));
        }
        _exec_deps = build_exec_deps(_deps);
#endif
        _deps = _network.get_primitives(_node.get_dependencies());
        _exec_deps = build_exec_deps(_deps);
    }
}

primitive_inst::primitive_inst(network& network, program_node const& node, bool allocate_memory)
    : _network(network), _node(node), _impl(node.get_selected_impl() ? node.get_selected_impl()->clone() : nullptr),
      _outputs({}), _output_changed(false), _mem_allocated(allocate_memory) {
    if (allocate_memory) {
#if 0 // TODO(taylor)
        // In case when output is mutable_data primitive, and other users dependencies are only used for
        // suychronization, The output memory of such primitive will be fused with mutable_data
        auto users = node.get_users();
        auto user_count = users.size();
        uint32_t mutable_data_count = 0;
        for (auto& user : users) {
            // Get mutable_data nodes count from nodes users
            if (user->is_type<mutable_data>()) {
                mutable_data_count++;
            }
        }
        // TODO: Remove WA for arg_max_min node.
        // For now it's required to handle the case when only second output of TopK primitive is used in plugin,
        // but kernels always write both outputs to the same memory object which leads to wrong result.
        if (user_count == 1 && mutable_data_count == 1 && !node.is_type<arg_max_min>()
                                                       && !node.is_type<experimental_detectron_roi_feature_extractor>()) {
            for (auto& user : node.get_users())
                if (user->is_type<mutable_data>())
                    _output = user->as<mutable_data>().get_attached_memory_ptr();
        } else {
            _outputs = allocate_outputs();
        }
#else

        _outputs = allocate_outputs();
#endif
    }
}

void primitive_inst::allocate_internal_buffers(void) {
    if (_impl == nullptr) return;
    const auto& ibuf_layouts = _impl->get_internal_buffer_layouts();
    if (ibuf_layouts.empty()) return;

    auto device_mem_acc = [&](size_t a, std::pair<std::shared_ptr<primitive_inst>, int32_t> b) {
        if (!b.first->mem_allocated()) return a;
        auto res = a;
        for (size_t i = 0; i < b.first->outputs_memory_count(); ++i) {
            if (b.first->output_memory(i).get_allocation_type() == allocation_type::usm_device ||
                    b.first->output_memory(i).get_allocation_type() == allocation_type::cl_mem)
                res += b.first->output_memory(i).size();
        }
        return res;
    };

    auto& engine = get_network().get_engine();
    bool input_device_mem = false;

    // NOTE: Currently the ocl driver aborts at runtime when there are layers using device memory close to max size within multiple streams.
    // Decided the limitation as 85 % empirically, but still it needs further investigation.
    std::vector<program_node*> dep_nodes;
    for (const auto& n : _node.get_dependencies()) {
        dep_nodes.push_back(n.first);
    }

    const auto& inst_deps = _network.get_primitives(_node.get_dependencies());

    auto total_device_mem_size = std::accumulate(inst_deps.begin(), inst_deps.end(), 0, device_mem_acc);
    for (const auto& o : _outputs) {
        if (o->get_allocation_type() ==  allocation_type::usm_device) {
            total_device_mem_size += o->size();
        }
    }

    int64_t available_device_mem_size = engine.get_device_info().max_global_mem_size - total_device_mem_size;
    // check if there is any device mem input
    if (engine.supports_allocation(allocation_type::usm_device)) {
        for (const auto& dep : inst_deps) {
            if (dep.first->output_memory().get_allocation_type() == allocation_type::usm_device) {
                input_device_mem = true;
                break;
            }
        }
    }

    for (auto layout : ibuf_layouts) {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": internal buf]" << std::endl;
        }
        if (input_device_mem && (available_device_mem_size - (int64_t)layout.bytes_count() >= 0))
            _intermediates_memory.push_back(engine.allocate_memory(layout, allocation_type::usm_device));
        else
            _intermediates_memory.push_back(engine.allocate_memory(layout, allocation_type::usm_host));
    }
}
std::vector<memory::ptr> primitive_inst::allocate_outputs() {
//    return allocate_outputs(get_network().get_engine(), _network.get_memory_pool(), _node, _network.is_internal());
    std::vector<memory::ptr> outputs;
    for (auto i = 0; i < get_node().get_outputs_count() ; ++i) {
        // TODO(taylor) : temporal solution for argmax. Future impl should take care of different layouts
        outputs.push_back(allocate_output(get_network().get_engine(), _network.get_memory_pool(), _node, _network.is_internal()));
    }
    return outputs;
}

memory::ptr primitive_inst::allocate_output(engine& _engine, memory_pool& pool, const program_node& _node,
        bool is_internal) {
    auto get_memory_from_pool = [&](engine& _engine, const layout& layout, const primitive_id id, std::set<primitive_id> dependencies,
            allocation_type type, bool reusable) {
        if (_engine.configuration().use_memory_pool)
                return pool.get_memory(layout, id, 0, dependencies, type, reusable);
        return pool.get_memory(layout, type);
    };

    // TODO(taylor) : temporal solution for argmax. Future impl should take care of different layouts
    auto layout = _node.get_output_layout(0);
    // TODO: Add a preprocessing step to do  alloc_type check before actual allocation
    const auto& node_deps = _node.get_dependencies();
    auto device_mem_acc = [&](size_t a, std::pair<program_node*, int32_t> b) {
        size_t res = a;
        for (auto o : b.first->get_output_layouts()) {
            res += o.bytes_count();
        }
        return res;
    };

    bool usm_device_allocatable = true;
    const auto& total_device_input_mem_size = std::accumulate(node_deps.begin(), node_deps.end(), (uint64_t)0, device_mem_acc);
    if (total_device_input_mem_size > _engine.get_device_info().max_global_mem_size)
        usm_device_allocatable = false;

    // For outputs, cpu prim we want to have lockable alloc type
    // Also if the successor of a node is an cpu, then memory needs to be lockable.
    auto use_lockable_memory = is_output_buffer(_node) || _node.get_selected_impl()->is_cpu() || is_any_user_cpu(_node.get_users()) ||
                               !_engine.supports_allocation(allocation_type::usm_device);

    GPU_DEBUG_GET_INSTANCE(debug_config);
    const auto& lockable_mem_type = _engine.get_lockable_preffered_memory_allocation_type(layout.format.is_image_2d());
    const auto& alloc_type = use_lockable_memory ? lockable_mem_type
        : usm_device_allocatable ? allocation_type::usm_device : lockable_mem_type;

    std::set<primitive_id> dep_pids = {_node.id()};
    for (const auto& d : _node.get_memory_dependencies()) {
        dep_pids.insert(d.pid);
    }
#if 0 // TODO(taylor)
    if (is_internal && (_node.can_be_optimized() || _node.is_type<generic_layer>())) {
#else
    if (is_internal && (_node.can_be_optimized())) {
#endif
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                dep_pids,
                alloc_type,
                false);
#if 0 // TODO(taylor)
    } else if (is_internal && _node.is_output() && _node.is_type<generic_layer>() &&
            _engine.supports_allocation(allocation_type::usm_device) && usm_device_allocatable) {
#else
    } else if (is_internal && _node.is_output() &&
            _engine.supports_allocation(allocation_type::usm_device) && usm_device_allocatable) {
#endif
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return _engine.allocate_memory(layout, allocation_type::usm_device, false);
    } else if (is_internal && !_node.is_output() && _node.is_type<input_layout>()) {
        // Skip memory reset for input_layout primitives, since data will be copied from cldnn::data primitive
        // or just reuse primitive's memory
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": constant]" << std::endl;
        }
        return _engine.allocate_memory(layout, alloc_type, false);
    } else if (is_internal || (!_node.can_share_buffer()) || _node.can_be_optimized() || _node.is_output()) {
        GPU_DEBUG_IF(debug_config->verbose >= 2) {
            GPU_DEBUG_COUT << "[" << _node.id() << ": output]" << std::endl;
        }
        return _engine.allocate_memory(layout, alloc_type);
    } else {
#if 0 // TODO (taylor) turn on this once required memory reuse support is enabled for multiple output
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                dep_pids,
                alloc_type,
                true);
#else
        return get_memory_from_pool(_engine,
                layout,
                _node.id(),
                dep_pids,
                alloc_type,
                false);
#endif
    }
}
memory::ptr primitive_inst::allocate_output() {
    return allocate_output(get_network().get_engine(), _network.get_memory_pool(), _node, _network.is_internal());
}

std::vector<std::pair<std::shared_ptr<primitive_inst>, int32_t>> primitive_inst::build_exec_deps(
    std::vector<std::pair<std::shared_ptr<primitive_inst>, int32_t>> const& deps) {
    std::vector<std::pair<std::shared_ptr<primitive_inst>, int32_t>> exec_deps;
    exec_deps.reserve(deps.size());
    for (auto& dep : deps)
        if (dep.first->get_impl() != nullptr)
            exec_deps.push_back(dep);

    return exec_deps;
}
#if 0 // TODO(taylor)
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
#endif
}  // namespace cldnn
