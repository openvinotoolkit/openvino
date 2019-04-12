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
#pragma once

#include "api/CPP/network.hpp"

#include "api_impl.h"
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "refcounted_obj.h"

#include <map>
#include <vector>
#include <unordered_map>

namespace cldnn
{

class primitive_inst;

struct network_impl : public refcounted_obj<network_impl>
{
public:
    network_impl(const program_impl& program, bool is_internal = false);
    network_impl(engine_impl& engine, const topology_impl& topo, const build_options& options = build_options(), bool is_internal = false);
    network_impl(engine_impl& engine, const std::set<std::shared_ptr<program_node>>& nodes, const build_options & options, bool is_internal);

    const program_impl& get_program() const { return *_program; }
    engine_impl& get_engine() const { return _program->get_engine(); }

    void reset_execution(bool wait = true);
    void set_input_data(const primitive_id& id, memory_impl& data);

    void set_learning_rate(const float lr);
    float get_learning_rate();

    std::vector<std::shared_ptr<primitive_inst>> const& get_outputs() { return _outputs; }

    const std::vector<std::shared_ptr<const primitive_inst>>& get_outputs() const
    {
        return reinterpret_cast<const std::vector<std::shared_ptr<const primitive_inst>>&>(_outputs);
    }

    std::vector<primitive_id> get_output_ids() const;
    std::vector<primitive_id> get_executed_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_org_ids() const;
    void execute(const std::vector<event_impl::ptr>& events);
    void validate_primitives();
    // Implementation specific calls
    std::shared_ptr<primitive_inst> get_primitive(const primitive_id& id);
    std::string get_primitive_info(const primitive_id& id) const;
    const event_impl::ptr& get_primitive_event(const primitive_id& id) const { return _events.at(id); }
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<primitive_id>& ids);
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<program_node*>& nodes);
    void execute_primitive(const std::shared_ptr<primitive_inst>& primitive, const std::vector<event_impl::ptr>& events);
    void allocate_primitives();
    void build_insts_deps();
    uint32_t get_id() const { return net_id; }
    void build_exec_order();    
    bool is_internal() const { return _internal; }
private:
    uint32_t net_id = 0; 
    const program_impl::cptr _program;
    bool _internal;
    float _learning_rate = float(0.00001);

    std::map<primitive_id, std::shared_ptr<primitive_inst>> _primitives;
    std::vector<std::shared_ptr<primitive_inst>> _inputs;
    std::vector<std::shared_ptr<primitive_inst>> _outputs;
    std::list<std::shared_ptr<primitive_inst>> _exec_order;
    std::list<std::shared_ptr<primitive_inst>> _data_outputs;

    std::unordered_map<primitive_id, event_impl::ptr> _events;

    void allocate_primitive_instance(program_node const& node);
    void add_to_exec_order(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_in_internal_networks(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_primitive(const primitive_id& id);
    void check_names();
};
}

API_CAST(::cldnn_network, cldnn::network_impl)
