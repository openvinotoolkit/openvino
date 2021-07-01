// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cldnn/runtime/compounds.hpp"
#include "cldnn/runtime/memory.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/stream.hpp"
#include "program.hpp"

#include <cstdint>
#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <string>

namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @defgroup cpp_network Network Execution
/// @{

/// @brief Represents network output returned by @ref network::get_output().
struct network_output {
    /// @brief Returns @ref event associated with the output.
    event::ptr get_event() const { return _event; }

    /// @brief Returns @ref memory object of the output. Blocked until associated @ref event is not complete.
    memory::ptr get_memory() const {
        // TODO: in_order queue doesn't create proper output event in some cases which leads to syncronization issues with user app
        // So call finish for associated stream to enusre that the output data is ready.
        if (_stream->get_queue_type() == queue_types::in_order) {
            _stream->finish();
        } else {
            _event->wait();
        }
        return _result;
    }

private:
    event::ptr _event;
    memory::ptr _result;
    stream::ptr _stream;
    network_output(event::ptr evt, memory::ptr mem, stream::ptr stream) : _event(evt), _result(mem), _stream(stream) {}
    friend struct network;
};

struct network_impl;

/// @brief Executable network allocated from @ref program.
struct network {
    /// @brief Allocate network
    /// @param program The program object which contains compiled primitives this network should allocate memory for.
    /// @param stream_id Stream ID of this network. 0 is for primary stream, the others are secondary.
    /// Used to determine whether an extra copy of primitive's memory needed.
    explicit network(program const& program, uint16_t stream_id);

    /// @brief Constructs network object from implicitly created program object. This is a shorthand for network(program(engine, topology, options))
    /// @param engine
    /// @param topology
    /// @param options
    /// @param options
    network(engine& engine,
            const topology& topology,
            const build_options& options = build_options(),
            uint16_t stream_id = 0)
        : network(program(engine, topology, options), stream_id) {}

    /// @brief Constructs network object from C API @ref cldnn_network.
    explicit network(std::shared_ptr<network_impl> impl) : _impl(impl) {
        if (_impl == nullptr)
            throw std::invalid_argument("implementation pointer should not be null");
    }

    /// @brief Copy construction.
    network(const network& other) : _impl(other._impl) { }

    /// @brief Copy assignment.
    network& operator=(const network& other) {
        if (_impl == other._impl)
            return *this;
        _impl = other._impl;
        return *this;
    }

    friend bool operator==(const network& lhs, const network& rhs) { return lhs._impl == rhs._impl; }
    friend bool operator!=(const network& lhs, const network& rhs) { return !(lhs == rhs); }

    /// @brief Returns @ref engine by which network was built.
    engine& get_engine() const;

    /// @brief Returns network internal @ref program.
    program get_program() const;

    /// @brief Provides @ref memory for @ref input_layout primitives defined by user in source @ref topology.
    void set_input_data(const primitive_id& id, memory::ptr mem) const;

    /// @brief Provides user-supplied @ref memory for output primitives defined by user in source @ref topology.
    void set_output_memory(const primitive_id& id, memory::ptr mem) const;

    /// @brief Return stream id.
    uint16_t get_stream_id();

    stream& get_stream() const;

    stream::ptr get_stream_ptr() const;

    /// @brief Return internal network id.
    uint32_t get_id();

    std::string get_primitive_info(const primitive_id& id) const;

    /// @brief Returns description of final runtime graph
    std::vector<primitive_info> get_primitives_info();

    /// @brief Returns description of all optimization stages
    std::vector<std::pair<std::string, std::vector<primitive_info>>> get_optimization_steps_info();

    /// @brief Returns the list of executed primitives.
    std::vector<primitive_id> get_executed_primitive_ids() const;

    /// @brief Returns the list of all primitives ids in network.
    std::vector<primitive_id> get_all_primitive_ids() const;

    /// @brief Returns the list of all primitives ids in network before graph optimization.
    std::vector<primitive_id> get_all_primitive_org_ids() const;

    /// @brief Returns the list of network inputs.
    std::vector<primitive_id> get_input_ids() const;

    /// @brief Returns the list of available network outputs.
    std::vector<primitive_id> get_output_ids() const;

    /// @brief Returns @ref memory object for particular @p output. Can be called before network execution
    memory::ptr get_output_memory(const primitive_id& output_id) const;

    /// @brief Returns @ref event object for particular @p primitive. Can't be called before network execution
    event::ptr get_primitive_event(const primitive_id& output_id) const;

    /// @brief Returns @ref network_output object for particular @p output. Can't be called before network execution
    network_output get_output(const primitive_id& output_id) const {
        return network_output(get_primitive_event(output_id), get_output_memory(output_id), get_stream_ptr());
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
            if (std::find(optimized_primitives.begin(), optimized_primitives.end(), id) == optimized_primitives.end())
                result.emplace(id, get_primitive_event(id));
        }
        return result;
    }

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

    /// @brief Executes network and returns the list of @ref network_output.
    /// @param dependencies List of @ref event objects to be waited before network execution.
    /// @note User should call set_input_data() for every @ref input_layout defined in source @ref topology
    /// before network execution.
    std::map<primitive_id, network_output> execute(const std::vector<event::ptr>& dependencies = {}) const;

    /// @brief Returns wrapped C API @ref cldnn_network handler.
    network_impl* get() const { return _impl.get(); }

private:
    std::shared_ptr<network_impl> _impl;
};
CLDNN_API_CLASS(network)
/// @}
/// @}
}  // namespace cldnn
