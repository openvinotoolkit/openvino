// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/graph/network.hpp"
#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/stream.hpp"
#include "program_impl.h"
#include "topology_impl.h"
#include "impls/implementation_map.hpp"

#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <list>
#include <set>

namespace cldnn {

class primitive_inst;

struct network_impl {
public:
    using ptr = std::shared_ptr<network_impl>;
    explicit network_impl(program_impl::ptr program, stream::ptr stream, bool is_internal = false, bool is_primary_stream = false);
    network_impl(engine& engine,
                 const topology_impl& topo,
                 const build_options& options = build_options(),
                 bool is_internal = false);
    network_impl(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const build_options& options,
                 bool is_internal);
    ~network_impl();


    static ptr build_network(engine& engine,
                             const topology_impl& topology,
                             const build_options& options,
                             bool is_internal = false);
    static ptr build_network(engine& engine,
                             const std::set<std::shared_ptr<program_node>>& nodes,
                             const build_options& options,
                             bool is_internal);

    static ptr allocate_network(stream::ptr stream,
                                program_impl::ptr program,
                                bool is_internal = false,
                                bool is_primary_stream = false);

    static ptr allocate_network(engine& engine,
                                program_impl::ptr program,
                                bool is_internal = false,
                                bool is_primary_stream = false);
    program_impl::cptr get_program() const { return _program; }
    program_impl::ptr get_program() { return _program; }
    engine& get_engine() const { return _program->get_engine(); }

    void reset_execution(bool wait = true);
    void set_input_data(const primitive_id& id, memory::ptr data);
    void set_output_memory(const primitive_id& id, memory::ptr mem);

    void set_learning_rate(const float lr);
    float get_learning_rate();

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
    void execute(const std::vector<event::ptr>& events);
    void validate_primitives();
    void set_arguments();
    // Implementation specific calls
    std::shared_ptr<primitive_inst> get_primitive(const primitive_id& id);
    std::string get_primitive_info(const primitive_id& id) const;
    const event::ptr& get_primitive_event(const primitive_id& id) const { return _events.at(id); }
    bool has_event(const primitive_id& id) const { return _events.count(id); }
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<primitive_id>& ids);
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<program_node*>& nodes);
    void execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                           const std::vector<event::ptr>& events);
    void allocate_primitives();
    void build_insts_deps();
    uint32_t get_id() const { return net_id; }
    stream& get_stream() const { return *_stream; }
    stream::ptr get_stream_ptr() const { return _stream; }
    void build_exec_order();
    bool is_internal() const { return _internal; }
    bool is_primary_stream() { return _is_primary_stream; }

    /// Create memory object with specified @p layout and allocation @p type for primitive with @p id
    /// Underlying memory handle can be reused with other primitives from memory pool based on @p dependencies
    memory_ptr get_memory_from_pool(const layout& layout,
                                    primitive_id id,
                                    std::set<primitive_id> dependencies,
                                    allocation_type type,
                                    bool reusable = true);

private:
    uint32_t net_id = 0;
    program_impl::ptr _program;
    stream::ptr _stream;
    std::unique_ptr<memory_pool> _memory_pool;
    bool _internal;
    bool _is_primary_stream;
    bool _reset_arguments;
    float _learning_rate = static_cast<float>(0.00001);

    std::map<primitive_id, std::shared_ptr<primitive_inst>> _primitives;
    std::vector<std::shared_ptr<primitive_inst>> _inputs;
    std::vector<std::shared_ptr<primitive_inst>> _outputs;
    std::list<std::shared_ptr<primitive_inst>> _exec_order;
    std::list<std::shared_ptr<primitive_inst>> _data_outputs;

    std::unordered_map<primitive_id, event::ptr> _events;

    void allocate_primitive_instance(program_node const& node);
    void transfer_memory_to_device(std::shared_ptr<primitive_inst> instance, program_node const& node);
    void add_to_exec_order(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_in_internal_networks(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_primitive(const primitive_id& id);
    void check_names();
};
}  // namespace cldnn
