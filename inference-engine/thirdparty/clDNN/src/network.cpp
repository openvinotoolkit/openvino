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
#include "network_impl.h"
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "api/CPP/data.hpp"
#include "api/CPP/mutable_data.hpp"
#include "api/CPP/input_layout.hpp"

#include "error_handler.h"
#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "kernel_selector_helper.h"
#include <algorithm>

namespace cldnn
{
/*
Network_impl will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by const. propagator).
*/
network_impl::network_impl(const program_impl& program, bool is_internal)
    : _program(&program)
    , _internal(is_internal)
{
    static std::atomic<uint32_t> id_gen{ 0 };
    if (!_internal)
    {
        net_id = ++id_gen;
    }

    allocate_primitives();
    build_insts_deps();
    build_exec_order();

    _program->dump_memory_pool();
}

network_impl::network_impl(engine_impl& engine, const topology_impl& topo, const build_options& options, bool is_internal)
    : network_impl(*engine.build_program(topo, options, is_internal), is_internal)
{
}

void network_impl::reset_execution(bool wait)
{
    if (wait && _events.size() > 0)
    {
        std::vector<event_impl::ptr> events;
        for (auto& pair : _events)
        {
            auto& ev = pair.second;
            if (ev->is_set())
                continue;

            events.push_back(ev);
        }

        get_engine().wait_for_events(events);
    }
    _events.clear();
}

void network_impl::set_input_data(const primitive_id& id, memory_impl& data)
{
    std::shared_ptr<primitive_inst> primitive_inst;
    try {
        primitive_inst = _primitives.at(id);
    }
    catch (...)
    {
        throw std::runtime_error("topology doesn't contain prmitive:" + id);
    }
    if (primitive_inst->type() != input_layout::type_id())
    {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }

    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);

    //Wait for previous execution completion
    reset_execution(true);
    input->set_data(data);
}

void network_impl::set_learning_rate(const float lr)
{
    _learning_rate = lr;
}

float network_impl::get_learning_rate()
{
    return _learning_rate;
}

std::string network_impl::get_primitive_info(const primitive_id& id) const
{    
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

void network_impl::allocate_primitives()
{
    auto nodes = _program->get_nodes();
    std::vector<std::shared_ptr<program_node>> nodes_to_allocate{};
    nodes_to_allocate.insert(nodes_to_allocate.begin(), nodes.begin(), nodes.end());
    std::sort(nodes_to_allocate.begin(), nodes_to_allocate.end(), [](auto const& lhs, auto const& rhs)
    {
        return (lhs->get_output_layout().bytes_count() > rhs->get_output_layout().bytes_count());
    });

    for (auto const& node : nodes_to_allocate)
    {
        allocate_primitive_instance(*node);
    }
}

void network_impl::build_exec_order_vist(program_node* node)
{
    if (!(!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())))
    {
        return;
    }
    auto it = std::find_if(_exec_order.begin(), _exec_order.end(),
        [&](std::shared_ptr<primitive_inst> inst) 
    {
        return inst->id() == node->id();
    });
    if (_exec_order.end() != it) //found
    {
        return;
    }
    for (auto& dep : node->get_dependencies())
    {
        build_exec_order_vist(dep);
    }
    _exec_order.push_back(get_primitive(node->id()));
}

void network_impl::build_exec_order()
{
    _program->get_nodes().reverse();
    for (auto& node : _program->get_nodes())
    {
        build_exec_order_vist(node.get());
    }
    _program->get_nodes().reverse();
}

void network_impl::build_insts_deps()
{
    for (auto& inst : _primitives)
    {
        inst.second->build_deps();
    }
}

void network_impl::execute(const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    //Wait for previous execution completion
    reset_execution(false);

    for (auto& inst : _exec_order)
    {
        execute_primitive(inst, events);
    }

    for (auto& dout : _data_outputs) //data primitives are not executed so if they are marked as output we need to add them valid events manually
    {
        _events[dout->id()] = get_engine().create_user_event(true);
    }

    for (auto& prim : _primitives)
    {
        prim.second->reset_output_change();
    }

    // Using output of previouse network as input to another one may cause hazard (in OOOQ mode) if user would not 
    // provide proper event to execution. Flushing pipeline should prevent this kind of issues. 
    // In scenarios with a big number of very small networks it can provide performance drop.
    get_engine().flush_network();
}

std::vector<primitive_id> network_impl::get_output_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_outputs.size());
    for (auto const& output : _outputs)
        ret.push_back(output->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_executed_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_exec_order.size());
    for (auto const& executed_primitive : _exec_order)
        ret.push_back(executed_primitive->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        if(primitive.second->can_be_optimized())
            ret.push_back("_optimized_");
        else
            ret.push_back(primitive.second->id());
    return ret;
}

std::vector<primitive_id> network_impl::get_all_primitive_org_ids() const
{
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        ret.push_back(primitive.second->org_id());
    return ret;
}

std::shared_ptr<primitive_inst> network_impl::get_primitive(const primitive_id& id)
{
    if (!_primitives.count(id))
        allocate_primitive_instance(_program->get_node(id));

    return _primitives.at(id);
}

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<primitive_id>& ids)
{
    std::vector<std::shared_ptr<primitive_inst>> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) { return get_primitive(id); });
    return result;
}

std::vector<std::shared_ptr<primitive_inst>> network_impl::get_primitives(const std::vector<program_node*>& nodes)
{
    std::vector<std::shared_ptr<primitive_inst>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const program_node* node) { return get_primitive(node->id()); });
    return result;
}

refcounted_obj_ptr<event_impl> network_impl::execute_primitive(const std::shared_ptr<primitive_inst>& primitive, const std::vector<refcounted_obj_ptr<event_impl>>& events)
{
    auto id = primitive->id();
    auto it = _events.find(id);
    if(it != _events.end())
    {
        return it->second;
    }

    event_impl::ptr ev;
    if (!get_engine().get_context()->enabled_single_kernel() || get_engine().get_context()->single_kernel_name() == id)
        ev = primitive->execute(events);
    else
        ev = get_engine().create_user_event(true);

    _events.insert({ id, ev });
    return ev;
}

void network_impl::allocate_primitive_instance(program_node const& node)
{
    if (_primitives.count(node.id()))
        return;

    auto inst = node.type()->create_instance(*this, node);
    _primitives[node.id()] = inst;
    if (node.is_input())
        _inputs.push_back(inst);
    if (node.is_output())
    {
        _outputs.push_back(inst);
        if (node.is_type<data>())
            _data_outputs.push_back(inst);
    }
}

}
