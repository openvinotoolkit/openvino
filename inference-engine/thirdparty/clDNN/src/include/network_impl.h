// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "api/network.hpp"

#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "refcounted_obj.h"

#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <list>
#include <set>

namespace cldnn {

class primitive_inst;

struct network_impl : public refcounted_obj<network_impl> {
public:
    explicit network_impl(const program_impl& program, uint16_t stream_id, bool is_internal = false);
    network_impl(engine_impl& engine,
                 const topology_impl& topo,
                 const build_options& options = build_options(),
                 uint16_t stream_id = 0,
                 bool is_internal = false);
    network_impl(engine_impl& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const build_options& options,
                 bool is_internal);
    ~network_impl();

    const program_impl& get_program() const { return *_program; }
    engine_impl& get_engine() const { return _program->get_engine(); }

    void reset_execution(bool wait = true);
    void set_input_data(const primitive_id& id, memory_impl& data);
    void set_output_memory(const primitive_id& id, memory_impl& mem);

    void set_learning_rate(const float lr);
    float get_learning_rate();
    uint16_t get_stream_id() const { return _stream_id; }

    std::vector<std::shared_ptr<primitive_inst>> const& get_outputs() { return _outputs; }

    const std::vector<std::shared_ptr<const primitive_inst>>& get_outputs() const {
        return reinterpret_cast<const std::vector<std::shared_ptr<const primitive_inst>>&>(_outputs);
    }

    std::vector<primitive_id> get_output_ids() const;
    std::vector<primitive_id> get_input_ids() const;
    std::vector<primitive_id> get_executed_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_org_ids() const;
    const program_impl::primitives_info& get_primitives_info() const;
    const program_impl::graph_optimizer_info& get_optimizer_passes_info() const;
    void execute(const std::vector<event_impl::ptr>& events);
    void validate_primitives();
    void set_arguments();
    // Implementation specific calls
    std::shared_ptr<primitive_inst> get_primitive(const primitive_id& id);
    std::string get_primitive_info(const primitive_id& id) const;
    const event_impl::ptr& get_primitive_event(const primitive_id& id) const { return _events.at(id); }
    bool has_event(const primitive_id& id) const { return _events.count(id); }
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<primitive_id>& ids);
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<program_node*>& nodes);
    void execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                           const std::vector<event_impl::ptr>& events);
    void allocate_primitives();
    void build_insts_deps();
    uint32_t get_id() const { return net_id; }
    void build_exec_order();
    bool is_internal() const { return _internal; }
    bool is_primary_stream();
    bool is_secondary_stream();

private:
    uint32_t net_id = 0;
    const program_impl::cptr _program;
    uint16_t _stream_id;
    bool _internal;
    bool _reset_arguments;
    float _learning_rate = static_cast<float>(0.00001);

    std::map<primitive_id, std::shared_ptr<primitive_inst>> _primitives;
    std::vector<std::shared_ptr<primitive_inst>> _inputs;
    std::vector<std::shared_ptr<primitive_inst>> _outputs;
    std::list<std::shared_ptr<primitive_inst>> _exec_order;
    std::list<std::shared_ptr<primitive_inst>> _data_outputs;

    std::unordered_map<primitive_id, event_impl::ptr> _events;

    void allocate_primitive_instance(program_node const& node);
    void transfer_memory_to_device(std::shared_ptr<primitive_inst> instance, program_node const& node);
    void allocate_mutable_data_for_streams(std::vector<std::shared_ptr<program_node>>& mutable_data_nodes);
    void add_to_exec_order(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_in_internal_networks(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_primitive(const primitive_id& id);
    void check_names();
};
}  // namespace cldnn
