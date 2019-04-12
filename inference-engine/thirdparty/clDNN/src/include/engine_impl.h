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
#include "api/CPP/memory.hpp"
#include "api_impl.h"
#include "event_impl.h"
#include "refcounted_obj.h"
#include "implementation_map.h"
#include "memory_pool.h"
#include "gpu/engine_info.h"

#include <memory>
#include <set>

namespace cldnn {
namespace gpu { 
    class gpu_toolkit;
}

class build_options;
using gpu_toolkit = gpu::gpu_toolkit;

struct memory_impl;
struct event_impl;
struct topology_impl;
struct program_impl;
struct network_impl;
struct program_node;

template <class>
struct typed_program_node;

struct engine_impl : public refcounted_obj<engine_impl>
{
public:
    engine_impl(const engine_configuration& conf);
    ~engine_impl();
    engine_types type() const { return engine_types::ocl; }
    refcounted_obj_ptr<memory_impl> allocate_memory(layout layout);
    refcounted_obj_ptr<memory_impl> allocate_memory(layout layout, primitive_id, uint32_t, std::set<primitive_id>, bool reusable = true);
    refcounted_obj_ptr<memory_impl> reinterpret_buffer(const memory_impl& memory, layout new_layout);
    bool is_the_same_buffer(const memory_impl& mem1, const memory_impl& mem2);

    refcounted_obj_ptr<event_impl> create_user_event(bool set = false);
    void wait_for_events(std::vector<event_impl::ptr> const& events);

    refcounted_obj_ptr<program_impl> build_program(const topology_impl& topology, const build_options& options, bool is_internal = false, bool no_optimizations = false);
    refcounted_obj_ptr<program_impl> build_program(const std::set<std::shared_ptr<program_node>>& nodes, const build_options & options, bool is_internal);
    void compile_program(program_impl& prog);

    refcounted_obj_ptr<network_impl> allocate_network(const program_impl& program, bool is_internal = false);
    refcounted_obj_ptr<network_impl> build_network(const topology_impl& topology, const build_options& options, bool is_internal = false);
    refcounted_obj_ptr<network_impl> build_network(const std::set<std::shared_ptr<program_node>>& nodes, const build_options & options, bool is_internal);
    void flush_network();
    void release_pending_memory();

    template <class T>
    std::unique_ptr<primitive_impl> create_primitive_impl(typed_program_node<T> const& node)
    {
        if (&node.get_program().get_engine() != this)
            throw std::invalid_argument("engine_impl::create_primitive_impl: program's engine does not match called engine");

        auto factory = implementation_map<T>::get(type(), node);
        return std::move(std::unique_ptr<primitive_impl>(factory(node)));
    }

    template <class T>
    bool does_an_implementation_exist(typed_program_node<T> const& node)
    {
        if (&node.get_program().get_engine() != this)
          throw std::invalid_argument("engine_impl::create_primitive_impl: program's engine does not match called engine");
        return implementation_map<T>::check(type(), node);
    }

    template <class T>
    bool does_possible_implementation_exist(typed_program_node<T> const& node)
    {
        if (&node.get_program().get_engine() != this)
            throw std::invalid_argument("engine_impl::create_primitive_impl: program's engine does not match called engine");
        return implementation_map<T>::check_io_eq(type(), node);
    }

    const engine_configuration& configuration() const { return _configuration; }
    void set_mem_pool(bool flag) { _configuration.enable_memory_pool = flag; }
    std::shared_ptr<gpu_toolkit> get_context() const { return _context; }
    gpu::engine_info_internal get_engine_info() const;
    memory_pool& get_memory_pool() { return _memory_pool; }

    uint64_t get_max_used_device_memory() const { return _memory_pool.get_max_peak_device_memory_used(); }
    uint64_t get_used_device_memory() const { return _memory_pool.get_temp_memory_used(); }

    void dump_memory_pool(const program_impl& program, std::string path, std::string dependencies) { _memory_pool.dump_memory_pool(program, path, dependencies); }
    bool use_memory_pool() const;

private:
    engine_configuration _configuration;
    std::shared_ptr<gpu_toolkit> _context;
	memory_pool _memory_pool;
};
}

API_CAST(::cldnn_engine, cldnn::engine_impl)