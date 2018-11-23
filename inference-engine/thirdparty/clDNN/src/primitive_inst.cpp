/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "primitive_inst.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "generic_layer_inst.h"
#include "input_layout_inst.h"
#include "max_unpooling_inst.h"
#include "apply_adam_inst.h"

#include "network_impl.h"
#include "engine_impl.h"
#include "memory_impl.h"

#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{

uint32_t primitive_inst::get_network_id() const 
{
    return _network.get_id();
}

event_impl::ptr primitive_inst::execute(const std::vector<event_impl::ptr>& events)
{
    CLDNN_ERROR_BOOL(id(), "Invalid/unset input", !_has_valid_input, "Cannot execute primitive " + id() + " with invalid/unset input");
    on_execute();

    if (_exec_deps.size() == 0)
       return _impl->execute(events, *this);      

    std::vector<event_impl::ptr> dependencies;
    dependencies.reserve(_exec_deps.size());
    for (auto& input : _exec_deps)
    {
        auto id = input->id();
        try {
            // if the requested event deos not exits it means that it has not been executed, so the processing_order is wrong or synchronization failed.
            auto ev = get_network().get_primitive_event(id); 
            dependencies.emplace_back(ev);
        }
        catch (const std::out_of_range& oor) {
            std::string temp = std::string("internal CLDNN error: execution order corrupted.") + std::string("\n") +  std::string(oor.what() + std::string("\n"));
            CLDNN_ERROR_MESSAGE(id, temp);
        }
    }
    return _impl->execute(dependencies, *this);  
}

void primitive_inst::build_deps()
{
    if (_deps.empty() && !_node.get_dependencies().empty())
    {
        _deps = _network.get_primitives(_node.get_dependencies());
        _exec_deps = build_exec_deps(_deps);
    }
}

primitive_inst::primitive_inst(network_impl& network, program_node const& node, bool allocate_memory)
    : _network(network)
    , _node(node)
    , _impl(node.get_selected_impl())
    , _output()
    , _output_changed(false)
{
    if (allocate_memory)
    {
        //In case when output is mutable_data primitive, and other users dependencies are only used for suychronization,
        //The output memory of such primitive will be fused with mutable_data
        auto users = node.get_users();
        auto user_count = users.size();
        uint32_t mutable_data_count = 0;
        for (auto& user : users)
        {
            //Get mutable_data nodes count from nodes users
            if (user->is_type<mutable_data>())
                mutable_data_count++;
            //For certain primitives, it is known which dependency is used for synchronization only
            else if (user->is_type<apply_adam>() && (user->as<apply_adam>().has_additional_dep()) && (user->as<apply_adam>().additional_dep().id() == node.id()))
                user_count--;
        }

        if (user_count == 1 && mutable_data_count == 1)
        {
            for (auto& user : node.get_users())
                if (user->is_type<mutable_data>())
                    _output = user->as<mutable_data>().get_attached_memory_ptr();
        }
        else
            _output = allocate_output();
    }
}

memory_impl::ptr primitive_inst::allocate_output()
{
    auto layout = _node.get_output_layout();

    if (!_network.is_internal() &&
        (_node.can_be_optimized() ||
        _node.is_type<generic_layer>()))
    {
        return get_network().get_engine().allocate_memory(layout, _node.id(), get_network_id(), _node.get_memory_dependencies(), false);
    }
    else if (_network.is_internal() ||
        _node.is_type<data>() ||
        _node.is_type<mutable_data>() ||
        _node.is_type<input_layout>() ||
        //for max_unpooling initial zero values are significant
        _node.is_type<max_unpooling>() ||
        //apply adam's output initial val should be either 0 or use same buffer as mutable_data after it (no allocation needed)
        _node.is_type<apply_adam>() ||
        _node.can_be_optimized() ||
        _node.is_output())
    {
        return get_network().get_engine().allocate_memory(layout);
    }
    return get_network().get_engine().allocate_memory(layout, _node.id(), get_network_id(), _node.get_memory_dependencies(), true);
}

std::vector<std::shared_ptr<primitive_inst>> primitive_inst::build_exec_deps(std::vector<std::shared_ptr<primitive_inst>> const& deps)
{
    std::vector<std::shared_ptr<primitive_inst>> exec_deps;
    exec_deps.reserve(deps.size());
    for (auto& dep : deps)
        if (dep->get_impl() != nullptr)
            exec_deps.push_back(dep);

    return exec_deps;
}

std::string primitive_inst::generic_to_string(program_node const& node, const char* type_name)
{
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_inputs;

    for (size_t i = 0; i < node.get_dependencies().size(); ++i)
    {
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

}
