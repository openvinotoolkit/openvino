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
#include "api/memory.hpp"
#include "api/profiling.hpp"
#include "event_impl.h"
#include "refcounted_obj.h"
#include "implementation_map.h"
#include "memory_pool.h"
#include "device_impl.h"

#include <memory>
#include <set>
#include <vector>
#include <utility>
#include <string>
#ifdef ENABLE_CLDNN_PROFILING_PTRACE
#include <unordered_set>
#include <atomic>
#endif

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

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
enum class et : int8_t {
    mark,
    begin,
    end,
    async_start,
    async_finish
};

struct stored_event {
    stored_event(const char* _name, uint64_t _val, et _type) : name(_name), value(_val), type(_type){};

    const char* name;
    uint64_t value;
    et type;
};

struct mem_event {
    uint64_t vm_peak_rss;
    uint64_t vm_rss;
    uint64_t mp_temp_memory_used;
    uint64_t mp_max_peak_device_memory_used;
    uint64_t ts;
};

struct time_logger {
    time_logger();
    double ticks_per_usec();

    double start_ns;
    uint64_t start_ticks;
    int tid;
    int pid;
};
#endif

struct engine_impl : public refcounted_obj<engine_impl> {
public:
    explicit engine_impl(const device_impl& dev, const engine_configuration& conf);
    ~engine_impl();
    engine_types type() const { return engine_types::ocl; }
    refcounted_obj_ptr<memory_impl> allocate_memory(const layout& layout, uint32_t net_id, bool reset = true);
    refcounted_obj_ptr<memory_impl> allocate_memory(const layout& layout, allocation_type type, uint32_t net_id = 0, bool reset = true);
    refcounted_obj_ptr<memory_impl> allocate_memory(const layout& layout,
                                                    primitive_id,
                                                    uint32_t network_id,
                                                    std::set<primitive_id>,
                                                    allocation_type type,
                                                    bool reusable = true);
    refcounted_obj_ptr<memory_impl> reinterpret_buffer(const memory_impl& memory, const layout& new_layout);
    refcounted_obj_ptr<memory_impl> reinterpret_handle(const layout& new_layout,
                                                       const shared_mem_params* params,
                                                       uint32_t net_id);
    bool is_the_same_buffer(const memory_impl& mem1, const memory_impl& mem2);

    refcounted_obj_ptr<event_impl> create_user_event(uint32_t net_id, bool set = false);
    void wait_for_events(std::vector<event_impl::ptr> const& events);

    refcounted_obj_ptr<program_impl> build_program(const topology_impl& topology,
                                                   const build_options& options,
                                                   bool is_internal = false,
                                                   bool no_optimizations = false);
    refcounted_obj_ptr<program_impl> build_program(const std::set<std::shared_ptr<program_node>>& nodes,
                                                   const build_options& options,
                                                   bool is_internal);
    void compile_program(program_impl& prog);

    refcounted_obj_ptr<network_impl> allocate_network(const program_impl& program,
                                                      uint16_t stream_id,
                                                      bool is_internal = false);
    refcounted_obj_ptr<network_impl> build_network(const topology_impl& topology,
                                                   const build_options& options,
                                                   uint16_t stream_id,
                                                   bool is_internal = false);
    refcounted_obj_ptr<network_impl> build_network(const std::set<std::shared_ptr<program_node>>& nodes,
                                                   const build_options& options,
                                                   bool is_internal);
    void flush_network(uint32_t net_id);
    void release_pending_memory(uint32_t net_id);

    template <class T>
    std::unique_ptr<primitive_impl> create_primitive_impl(typed_program_node<T> const& node) {
        if (&node.get_program().get_engine() != this)
            throw std::invalid_argument(
                "engine_impl::create_primitive_impl: program's engine does not match called engine");

        auto factory = implementation_map<T>::get(type(), node);
        return std::move(std::unique_ptr<primitive_impl>(factory(node)));
    }

    template <class T>
    bool does_an_implementation_exist(typed_program_node<T> const& node) {
        if (&node.get_program().get_engine() != this)
            throw std::invalid_argument(
                "engine_impl::create_primitive_impl: program's engine does not match called engine");
        return implementation_map<T>::check(type(), node);
    }

    template <class T>
    bool does_possible_implementation_exist(typed_program_node<T> const& node) {
        if (&node.get_program().get_engine() != this)
            throw std::invalid_argument(
                "engine_impl::create_primitive_impl: program's engine does not match called engine");
        return implementation_map<T>::check_io_eq(type(), node);
    }

    const engine_configuration& configuration() const { return _configuration; }
    void set_mem_pool(bool flag) { _configuration.enable_memory_pool = flag; }
    std::shared_ptr<gpu_toolkit> get_context() const { return _context; }
    gpu::device_info_internal get_device_info() const;
    void* get_user_context() const;
    memory_pool& get_memory_pool() { return _memory_pool; }

    uint64_t get_max_used_device_memory() const { return _memory_pool.get_max_peak_device_memory_used(); }
    uint64_t get_used_device_memory() const { return _memory_pool.get_temp_memory_used(); }

    void dump_memory_pool(const program_impl& program, std::string& path, std::string& dependencies) {
        _memory_pool.dump_memory_pool(program, path, dependencies);
    }
    bool use_memory_pool() const;
    bool use_unified_shared_memory() const;
    bool supports_allocation(allocation_type type) const;
    allocation_type get_lockable_preffered_memory_allocation_type(bool is_image_layout = false) const;

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
    void event_mark(const std::string name);
    void event_begin(const std::string name);
    void event_end(const std::string name);
    void async_start(const std::string name);
    void async_finish(const std::string name);
    void logger_flush();
    void mem_tick();
#endif
private:
    engine_configuration _configuration;
    std::shared_ptr<gpu_toolkit> _context;
    memory_pool _memory_pool;

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
    std::unordered_set<std::string> name_set;
    std::vector<stored_event> ev_store;
    std::vector<mem_event> mem_ev_store;
    std::atomic_flag lock;
    time_logger _timer;
#endif

    void acquire_lock() {
        while (lock.test_and_set(std::memory_order_acquire)) {}
    }

    void release_lock() {
        lock.clear(std::memory_order_release);
    }
};

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
struct logger_scope_internal {
    explicit logger_scope_internal(engine_impl* logger, std::string name) :
        _logger(logger), _name(name) {
        if (_logger != nullptr)  _logger->event_begin(_name);
    }
    ~logger_scope_internal() { if (_logger != nullptr)  _logger->event_end(_name); }
private:
    std::string _name;
    engine_impl* _logger;
};
#define CLDNN_TRACE_IR_METHOD_INTERNAL(name) \
logger_scope_internal fscope ## __LINE__(CLDNN_TRACE_IR_ENGINE, name);

#define CLDNN_TRACE_IR_SCOPE_INTERNAL_BEGIN(name) \
{ logger_scope_internal lscope ## __LINE__(CLDNN_TRACE_IR_ENGINE, name);

#define CLDNN_TRACE_IR_SCOPE_INTERNAL_END }

#define CLDNN_TRACE_IR_MARK_INTERNAL(name) CLDNN_TRACE_IR_ENGINE->event_mark(name);

#define CLDNN_TRACE_IR_MEM_INTERNAL CLDNN_TRACE_IR_ENGINE->mem_tick();
#else
    #define CLDNN_TRACE_IR_METHOD_INTERNAL(name)
    #define CLDNN_TRACE_IR_SCOPE_INTERNAL_BEGIN(name)
    #define CLDNN_TRACE_IR_SCOPE_INTERNAL_END
    #define CLDNN_TRACE_IR_MARK_INTERNAL(name)
    #define CLDNN_TRACE_IR_MEM_INTERNAL
#endif
}  // namespace cldnn
