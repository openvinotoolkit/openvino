// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/compounds.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/shape_predictor.hpp"

#include <map>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <list>
#include <set>

namespace cldnn {

/// @brief Represents network output returned by @ref network::get_output().
struct network_output {
    /// @brief Returns @ref event associated with the output.
    event::ptr get_event() const { return _event; }

    /// @brief Returns @ref memory object of the output. Blocked until associated @ref event is not complete.
    memory::ptr get_memory(bool do_sync = true) const {
        // TODO: in_order queue doesn't create proper output event in some cases which leads to syncronization issues with user app
        // So call finish for associated stream to enusre that the output data is ready.
        if (do_sync) {
            if (_stream->get_queue_type() == QueueTypes::in_order) {
                _stream->finish();
            } else {
                _event->wait();
            }
        }
        return _result;
    }

    layout get_layout() const { // Last tensor memory might be null (e.g., {N, 0} shape) but we should be able to get the layout
        return _layout;
    }

private:
    event::ptr _event;
    memory::ptr _result;
    stream::ptr _stream;
    layout _layout;
    network_output(event::ptr evt, memory::ptr mem, stream::ptr stream, const layout& layout) : _event(evt), _result(mem), _stream(stream), _layout(layout) {}
    friend struct network;
};

class primitive_inst;
class ICompilationContext;

struct network {
public:
    using ptr = std::shared_ptr<network>;

    struct VariableState {
        using Ptr = std::shared_ptr<VariableState>;

        VariableState(cldnn::memory_ptr mem = nullptr) :
            memory { std::move(mem) }, is_set { false } {
        }
        void set_memory(cldnn::memory_ptr new_mem) {
            memory = new_mem;
        }
        cldnn::memory_ptr memory;
        bool is_set;
    };
    using variables_states_map = std::map<std::string, VariableState::Ptr>;

    explicit network(program::ptr program, const ExecutionConfig& config, stream::ptr stream, bool is_internal = false, bool is_primary_stream = true);
    network(engine& engine,
            const topology& topo,
            const ExecutionConfig& config = {},
            bool is_internal = false,
            std::shared_ptr<ov::threading::IStreamsExecutor> task_executor = nullptr);

    network(engine& engine,
            const std::set<std::shared_ptr<program_node>>& nodes,
            const ExecutionConfig& config,
            std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
            bool is_internal);

    network(program::ptr program, uint16_t stream_id = 0);

    network(program::ptr program, stream::ptr stream, uint16_t stream_id);

    network(cldnn::BinaryInputBuffer& ifs, stream::ptr stream, engine& engine, bool is_primary_stream, uint32_t local_net_id);
    network(cldnn::BinaryInputBuffer& ifs, const ExecutionConfig& config, stream::ptr stream, engine& engine, bool is_primary_stream, uint32_t local_net_id);

    ~network();

    void save(cldnn::BinaryOutputBuffer& ob);

    static ptr build_network(engine& engine,
                             const topology& topology,
                             const ExecutionConfig& config = {},
                             std::shared_ptr<ov::threading::IStreamsExecutor> task_executor = nullptr,
                             bool is_internal = false);

    static ptr build_network(engine& engine,
                             const std::set<std::shared_ptr<program_node>>& nodes,
                             const ExecutionConfig& config,
                             std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                             bool is_internal);

    static ptr allocate_network(stream::ptr stream,
                                program::ptr program,
                                bool is_internal = false,
                                bool is_primary_stream = false);

    static ptr allocate_network(engine& engine,
                                program::ptr program,
                                bool is_internal = false,
                                bool is_primary_stream = false);
    program::cptr get_program() const { return _program; }
    program::ptr get_program() { return _program; }
    engine& get_engine() const { return _engine; }

    void reset_execution(bool wait = true);
    event::ptr set_input_data(const primitive_id& id, memory::ptr data);
    std::vector<event::ptr> set_output_memory(const primitive_id& id, memory::ptr mem);

    std::vector<std::shared_ptr<primitive_inst>> const& get_outputs() { return _outputs; }

    const std::vector<std::shared_ptr<const primitive_inst>>& get_outputs() const {
        return reinterpret_cast<const std::vector<std::shared_ptr<const primitive_inst>>&>(_outputs);
    }

    network_output get_output(const primitive_id& output_id) {
        event::ptr evt;
        if (get_stream().get_queue_type() == QueueTypes::out_of_order)
            evt = get_primitive_event(output_id);
        return network_output(evt, get_output_memory(output_id), get_stream_ptr(), get_output_layout(output_id));
    }
    layout get_node_output_layout(const primitive_id& output_id) const;
    memory::ptr get_output_memory(const primitive_id& output_id);
    layout get_output_layout(const primitive_id& output_id) const;
    std::vector<layout> get_input_layouts() const;

    /// @brief Returns the list of primitive ids before and after graph optimization.
    /// @details If primitive was not optimized, the old and actual id will be the same.
    /// @n If primitive was optimized during graph optimization, the actual id will be "_optimized_".
    std::map<primitive_id, primitive_id> get_all_primitives() const {
        auto primitive_ids = get_all_primitive_ids();
        auto primitive_org_ids = get_all_primitive_org_ids();
        std::map<primitive_id, primitive_id> result;
        for (decltype(primitive_org_ids.size()) i = 0; i < primitive_org_ids.size(); i++) {
            result.emplace(primitive_org_ids[i], primitive_ids[i]);
        }
        return result;
    }

    /// @brief Returns the list of @ref event for the primitives that were executed in network.
    std::map<primitive_id, event::ptr> get_executed_primitives() const {
        auto primitive_ids = get_executed_primitive_ids();
        auto all_primitive_ids = get_all_primitive_ids();
        auto all_primitive_org_ids = get_all_primitive_org_ids();
        // Get list of optimized prmitives
        std::vector<primitive_id> optimized_primitives;
        for (decltype(all_primitive_org_ids.size()) i = 0; i < all_primitive_org_ids.size(); i++) {
            if (all_primitive_ids[i] == "_optimized_")
                optimized_primitives.push_back(all_primitive_org_ids[i]);
        }
        std::map<primitive_id, event::ptr> result;
        for (auto& id : primitive_ids) {
            if (std::find(optimized_primitives.begin(), optimized_primitives.end(), id) == optimized_primitives.end()) {
                if (has_event(id))
                    result.emplace(id, get_primitive_event(id));
                else
                    result.emplace(id, nullptr);
            }
        }
        return result;
    }

    std::vector<primitive_id> get_output_ids() const;
    std::vector<primitive_id> get_input_ids() const;
    std::vector<primitive_id> get_executed_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_ids() const;
    std::vector<primitive_id> get_all_primitive_org_ids() const;
    const program::primitives_info& get_primitives_info() const;
    const program::graph_optimizer_info& get_optimizer_passes_info() const;
    std::map<primitive_id, primitive_id> get_ext_id_mapping() const;
    void execute_impl(const std::vector<event::ptr>& events);

    /// @brief Executes network and returns the list of @ref network_output.
    /// @param dependencies List of @ref event objects to be waited before network execution.
    /// @note User should call set_input_data() for every @ref input_layout defined in source @ref topology
    /// before network execution.
    std::map<primitive_id, network_output> execute(const std::vector<event::ptr>& dependencies = {});

    void validate_primitives();
    void set_arguments();
    // Implementation specific calls
    bool is_cpu_impl(const primitive_id& id) const;
    std::shared_ptr<primitive_inst> get_primitive(const primitive_id& id);
    std::shared_ptr<const primitive_inst> get_primitive(const primitive_id& id) const;
    std::string get_primitive_info(const primitive_id& id) const;
    std::string get_implementation_info(const primitive_id& id) const;
    const event::ptr& get_primitive_event(const primitive_id& id) const { return _events.at(id); }
    bool has_event(const primitive_id& id) const { return _events.count(id); }
    std::vector<std::shared_ptr<primitive_inst>> get_primitives(const std::vector<primitive_id>& ids);
    std::vector<std::pair<std::shared_ptr<primitive_inst>, int>> get_primitives(const std::vector<std::pair<program_node*, int>>& nodes);
    void execute_primitive(const std::shared_ptr<primitive_inst>& primitive,
                           const std::vector<event::ptr>& events);
    void allocate_primitives();
    void configure_primitives_second_output();
    void build_insts_deps();
    uint32_t get_id() const { return net_id; }
    uint32_t get_local_id() const { return _local_net_id; }
    stream& get_stream() const { return *_stream; }
    stream::ptr get_stream_ptr() const { return _stream; }
    bool is_internal() const { return _internal; }
    bool is_primary_stream() const { return _is_primary_stream; }
    bool is_dynamic() const { return _is_dynamic; }
    size_t get_weights_cache_capacity() const { return _weights_cache_capacity; }

    memory_pool& get_memory_pool() {
        return *_memory_pool;
    }

    void allocate_variables_memories();
    void assign_variables_memories();
    /// Assigns memory state locations
    void assign_variables_memories(variables_states_map &&variables_memories);
    void update_variable_memory(const std::string& variable_id, const cldnn::layout& layout);

    /// Returns memory state @p variable_id of stateful network
    VariableState& get_variable_memory(const std::string &variable_id);
    const variables_states_map& get_variable_memories() const { return _variables_states; }

    using variables_state_info_map = std::map<std::string, cldnn::layout>;
    void set_variables_state_info(const std::string& variable_id, const cldnn::layout& layout);
    const variables_state_info_map& get_variables_state_info() const;
    const ExecutionConfig& get_config() const { return _config; }

    std::shared_ptr<ShapePredictor> get_shape_predictor() { return _shape_predictor; }
    void set_shape_predictor(std::shared_ptr<ShapePredictor> shape_predictor) { _shape_predictor = shape_predictor; }

#ifdef GPU_DEBUG_CONFIG
    int64_t get_current_iteration_num() { return iteration; }
#endif

private:
    using output_chains_map = std::map<primitive_id, std::vector<std::shared_ptr<primitive_inst>>>;
    uint32_t net_id = 0;
    program::ptr _program;
    ExecutionConfig _config;
    engine& _engine;
    stream::ptr _stream;
    std::unique_ptr<memory_pool> _memory_pool;
    bool _internal;
    bool _is_primary_stream;
    bool _is_dynamic = false;
    bool _enable_profiling = false;
    bool _reset_arguments;
    uint32_t _local_net_id = 0;     // This is for thread-safe deserialization. 'net_id' is globally unique,
                                    // but '_local_net_id' is unique only in each intel_gpu::Graph.

    std::unordered_map<primitive_id, std::shared_ptr<primitive_inst>> _primitives;
    std::vector<shared_mem_type> _in_out_shared_mem_types;
    std::vector<std::shared_ptr<primitive_inst>> _inputs;
    std::vector<std::shared_ptr<primitive_inst>> _outputs;
    std::list<std::shared_ptr<primitive_inst>> _exec_order;
    std::list<std::shared_ptr<primitive_inst>> _data_outputs;
    variables_states_map _variables_states;
    std::vector<std::shared_ptr<primitive_inst>> _variable_state_primitives;
    variables_state_info_map _variables_state_info;
    program::primitives_info _prims_info;
    std::map<primitive_id, primitive_id> _ext_id_mapping;
    size_t _weights_cache_capacity = 1;

    std::unordered_map<primitive_id, event::ptr> _events;
    // This map is used to temporarily hold events that will be deallocated later
    std::unordered_map<primitive_id, event::ptr> _old_events;
    output_chains_map _output_chains;

    std::shared_ptr<ShapePredictor> _shape_predictor;

    void build_exec_order();
    void allocate_primitive_instance(program_node const& node);
    void transfer_memory_to_device(std::shared_ptr<primitive_inst> instance, program_node const& node);
    void add_to_exec_order(const primitive_id& id);
    std::shared_ptr<primitive_inst> find_in_internal_networks(const primitive_id& id) const;
    std::shared_ptr<primitive_inst> find_primitive(const primitive_id& id) const;
    void check_names();
    void add_default_output_chains();
    void calculate_weights_cache_capacity();
    output_chains_map::iterator add_output_chain(std::shared_ptr<primitive_inst>& p_inst);

#ifdef GPU_DEBUG_CONFIG
    int64_t iteration = 0;
#endif
};
}  // namespace cldnn
