/*
// Copyright (c) 2016-2019 Intel Corporation
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
#include "engine_impl.h"
#include "event_impl.h"
#include "program_impl.h"
#include "network_impl.h"
#include "gpu/ocl_toolkit.h"
#include "gpu/memory_gpu.h"
#include "gpu/ocl_user_event.h"
#include "gpu/register_gpu.hpp"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
#ifdef _MSC_VER
#include <intrin.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <x86intrin.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>
#include "document.h"
#include "ostreamwrapper.h"
#include "writer.h"
#endif

namespace cldnn {

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
#define CLDNN_TRACE_IR_ENGINE (this)

void engine::event_mark(const std::string name) {
    _impl->event_mark(name);
}
void engine::event_begin(const std::string name) {
    _impl->event_begin(name);
}
void engine::event_end(const std::string name) {
    _impl->event_end(name);
}
void engine::async_start(const std::string name) {
    _impl->async_start(name);
}
void engine::async_finish(const std::string name) {
    _impl->async_finish(name);
}
void engine::mem_tick() {
    _impl->mem_tick();
}
#endif

engine::engine(engine_types type, const device& dev, const engine_configuration& configuration)
    : _impl(new engine_impl(*dev.get(), configuration)) {
    if (type != engine_types::ocl)
        throw std::invalid_argument("Invalid engine type, should be ocl.");
}

uint32_t engine::engine_count(engine_types type) {
    if (type == engine_types::ocl) {
        return 1;
    } else {
        return 0;
    }
}

void engine::release_pending_memory(uint32_t net_id) const {
    _impl->release_pending_memory(net_id);
}

device_info engine::get_info() const {
    auto info = _impl->get_device_info();
    return info.convert_to_api();
}

void* engine::get_context() const {
    return _impl->get_user_context();
}

uint64_t engine::get_max_used_device_memory_size() const {
    return _impl->get_max_used_device_memory();
}

uint64_t engine::get_temp_used_device_memory_size() const {
    return _impl->get_used_device_memory();
}

engine_types engine::get_type() const {
    return _impl->type();
}

void engine::retain() {
    _impl->add_ref();
}
void engine::release() {
#ifdef ENABLE_CLDNN_PROFILING_PTRACE
    _impl->logger_flush();
#endif
    _impl->release();
}

using gpu_toolkit_config = gpu::configuration;

gpu_toolkit_config convert_configuration(const engine_configuration conf) {
    gpu_toolkit_config result;
    result.compiler_options = conf.compiler_options;
    result.enable_profiling = conf.enable_profiling != 0;
    result.meaningful_kernels_names = conf.meaningful_kernels_names != 0;
    result.dump_custom_program = conf.dump_custom_program != 0;
    result.single_kernel_name = conf.single_kernel_name;
    result.host_out_of_order = true;
    result.use_unifed_shared_memory = true;  // Switch on/off USM.
    result.log = conf.engine_log;
    result.ocl_sources_dumps_dir = conf.sources_dumps_dir;
    result.priority_mode = conf.priority_mode;
    result.throttle_mode = conf.throttle_mode;
    result.queues_num = conf.n_streams;
    result.tuning_cache_path = conf.tuning_cache_path;
    return result;
}

engine_impl::engine_impl(const device_impl& dev, const engine_configuration& conf)
    : _configuration(conf), _context(gpu_toolkit::create(dev, convert_configuration(conf))), _memory_pool(*this) {
    gpu::register_implementations_gpu();
}

engine_impl::~engine_impl() {
    /*
        Engine, which is main owner of context deallocate events pool manually, because
        of the event_impl <-> gpu_toolkit dependencies.
    */
    _context->release_all_events_pools();
}

memory_impl::ptr engine_impl::allocate_memory(const layout& layout, uint32_t net_id, bool reset) {
    allocation_type type = get_lockable_preffered_memory_allocation_type(layout.format.is_image_2d());
    return _memory_pool.get_memory(layout, type, net_id, reset);
}

memory_impl::ptr engine_impl::allocate_memory(const layout& layout, allocation_type type, uint32_t net_id, bool reset) {
    return _memory_pool.get_memory(layout, type, net_id, reset);
}

memory_impl::ptr engine_impl::allocate_memory(const layout& layout,
                                              primitive_id id,
                                              uint32_t network_id,
                                              std::set<primitive_id> dependencies,
                                              allocation_type type,
                                              bool reusable) {
    if (use_memory_pool())
        return _memory_pool.get_memory(layout, id, network_id, dependencies, type, reusable);
    return _memory_pool.get_memory(layout, type, network_id);
}

memory_impl::ptr engine_impl::reinterpret_buffer(const memory_impl& memory, const layout& new_layout) {
    if (memory.get_engine() != (const refcounted_obj_ptr<engine_impl>) this)
        throw std::runtime_error("trying to reinterpret buffer allocated by a different engine");

    if (new_layout.format.is_image() && !memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret non-image buffer as image");

    if (!new_layout.format.is_image() && memory.get_layout().format.is_image())
        throw std::runtime_error("trying to reinterpret image buffer as non-image buffer");

    try {
        if (new_layout.format.is_image_2d()) {
           memory_impl::ptr mem_impl {
                new gpu::gpu_image2d((refcounted_obj_ptr<engine_impl>) this,
                                     new_layout,
                                     reinterpret_cast<const gpu::gpu_image2d&>(memory).get_buffer(),
                                     memory.get_net_id()),
                                     false };
            return mem_impl;
        } else if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
            memory_impl::ptr mem_impl{
                    new gpu::gpu_usm((refcounted_obj_ptr<engine_impl>) this,
                                        new_layout,
                                        reinterpret_cast<const gpu::gpu_usm&>(memory).get_buffer(),
                                        memory.get_allocation_type(),
                                        memory.get_net_id()),
                                        false };
            return mem_impl;
        } else {
           memory_impl::ptr mem_impl {
                new gpu::gpu_buffer((refcounted_obj_ptr<engine_impl>) this,
                                    new_layout,
                                    reinterpret_cast<const gpu::gpu_buffer&>(memory).get_buffer(),
                                    memory.get_net_id()),
                                    false};
            return mem_impl;
        }
    } catch (cl::Error const& err) {
        throw gpu::ocl_error(err);
    }
}

memory_impl::ptr engine_impl::reinterpret_handle(const layout& new_layout,
                                    const shared_mem_params* params,
                                    uint32_t net_id) {
    return _memory_pool.get_memory(new_layout, params, net_id);
}


bool engine_impl::is_the_same_buffer(const memory_impl& mem1, const memory_impl& mem2) {
    if (mem1.get_engine() != (refcounted_obj_ptr<engine_impl>)this || mem2.get_engine() != (refcounted_obj_ptr<engine_impl>) this)
        return false;
    if (mem1.get_net_id() != mem2.get_net_id())
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    if (!memory_capabilities::is_usm_type(mem1.get_allocation_type()))
        return (reinterpret_cast<const gpu::gpu_buffer&>(mem1).get_buffer() ==
            reinterpret_cast<const gpu::gpu_buffer&>(mem2).get_buffer());
    else
        return (reinterpret_cast<const gpu::gpu_usm&>(mem1).get_buffer() ==
            reinterpret_cast<const gpu::gpu_usm&>(mem2).get_buffer());
}

event_impl::ptr engine_impl::create_user_event(uint32_t net_id, bool set) {
    try {
        return _context->create_user_event(net_id, set);
    } catch (cl::Error const& err) {
        throw gpu::ocl_error(err);
    }
}

void engine_impl::flush_network(uint32_t net_id) { get_context()->flush(net_id); }

void engine_impl::release_pending_memory(uint32_t net_id) { get_context()->release_pending_memory(net_id); }

program_impl::ptr engine_impl::build_program(const topology_impl& topology,
                                            const build_options& options,
                                            bool is_internal,
                                            bool no_optimizations) {
    program_impl::ptr progr_impl{ new program_impl(*this, topology, options, is_internal, no_optimizations), false };
    return progr_impl;
}

program_impl::ptr engine_impl::build_program(const std::set<std::shared_ptr<program_node>>& nodes,
                                            const build_options& options,
                                            bool is_internal) {
    program_impl::ptr progr_impl{ new program_impl(*this, nodes, options, is_internal), false };
    return progr_impl;
}

network_impl::ptr engine_impl::build_network(const topology_impl& topology,
                                            const build_options& options,
                                            uint16_t stream_id,
                                            bool is_internal) {
    network_impl::ptr netw_impl{ new network_impl(*this, topology, options, stream_id, is_internal), false };
    return netw_impl;
}

network_impl::ptr engine_impl::build_network(const std::set<std::shared_ptr<program_node>>& nodes,
                                            const build_options& options,
                                            bool is_internal) {
    network_impl::ptr netw_impl{ new network_impl(*this, nodes, options, is_internal), false };
    return netw_impl;
}

network_impl::ptr engine_impl::allocate_network(const program_impl& program, uint16_t stream_id, bool is_internal) {
    if (stream_id >= _configuration.n_streams)
        throw std::invalid_argument("Unable to create network with stream_id=" + std::to_string(stream_id));
    network_impl::ptr netw_impl{ new network_impl(program, stream_id, is_internal), false };
    return netw_impl;
}

void engine_impl::wait_for_events(std::vector<event_impl::ptr> const& events) {
    CLDNN_TRACE_IR_METHOD_INTERNAL("engine_impl::wait_for_events");
    if (!events.empty())
        _context->wait_for_events(events);
}

gpu::device_info_internal engine_impl::get_device_info() const { return _context->get_device_info(); }

void* engine_impl::get_user_context() const { return static_cast<void*>(_context->context().get()); }

void engine_impl::compile_program(program_impl& program) {
    CLDNN_TRACE_IR_MEM_INTERNAL
    CLDNN_TRACE_IR_METHOD_INTERNAL("engine_impl::compile_program");
    auto& cache = _context->get_kernels_cache(program.get_id());
    if (!program.get_options().get<build_option_type::serialize_network>()->serialization_network_name.empty())
        cache.get_context().set_serialization_flag(true);
    // TODO: better compilation logic instead of a simple 'compile all'?
    CLDNN_TRACE_IR_SCOPE_INTERNAL_BEGIN("compile_program - cache.build_all()")
    cache.build_all();
    CLDNN_TRACE_IR_SCOPE_INTERNAL_END
    CLDNN_TRACE_IR_MEM_INTERNAL
}

bool engine_impl::use_memory_pool() const {
    if (configuration().enable_memory_pool && get_context()->is_neo_driver()) {
        return true;
    }
    return false;
}

bool engine_impl::use_unified_shared_memory() const {
    if (get_context()->memory_caps().supports_usm() && get_context()->get_configuration().use_unifed_shared_memory) {
        return true;
    }
    return false;
}

bool engine_impl::supports_allocation(allocation_type type) const {
    if (memory_capabilities::is_usm_type(type) && !use_unified_shared_memory())
        return false;
    if (allocation_type::usm_shared == type)
        return false;
    return get_context()->memory_caps().support_allocation_type(type);
}

allocation_type engine_impl::get_lockable_preffered_memory_allocation_type(bool is_image_layout) const {
    if (!use_unified_shared_memory() || is_image_layout)
        return allocation_type::cl_mem;

    /*
        We do not check device allocation here.
        Device allocation is reserved for buffers of hidden layers.
        Const buffers are propagated to device if possible.
    */

    bool support_usm_host = supports_allocation(allocation_type::usm_host);
    bool support_usm_shared = supports_allocation(allocation_type::usm_shared);

    if (support_usm_shared)
        return allocation_type::usm_shared;
    if (support_usm_host)
        return allocation_type::usm_host;

    throw std::runtime_error("[clDNN internal error] Could not find proper allocation type!");
}

#ifdef ENABLE_CLDNN_PROFILING_PTRACE
time_logger::time_logger() {
#ifdef _WIN32
    tid = (int)GetCurrentThreadId();
    pid = (int)GetCurrentProcessId();
    uint64_t _starttime;
    QueryPerformanceCounter((LARGE_INTEGER*)&_starttime);
    start_ticks = __rdtsc();
    start_ns = (double)_starttime;
#else
    tid = (int)(intptr_t)pthread_self();
    pid = (int)getpid();
    struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
    start_ticks = __rdtsc();
	start_ns = (double)time.tv_sec * 1.0e9 + (double)time.tv_nsec;
#endif
}

double time_logger::ticks_per_usec() {
#ifdef _WIN32
    uint64_t _frequency, _time;
    QueryPerformanceCounter((LARGE_INTEGER*)&_time);
    QueryPerformanceFrequency((LARGE_INTEGER*)&_frequency);
    uint64_t ticks = __rdtsc();
    double ns = ((double)_time - start_ns) * 1.0e9 / (double)_frequency;
#else
    struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
    uint64_t ticks = __rdtsc();
	double ns = (double)time.tv_sec * 1.0e9 + (double)time.tv_nsec - start_ns;
#endif
    return ((double)ticks - (double)start_ticks) * 1.0e3 / ns;
}

void engine_impl::event_mark(const std::string name) {
    uint64_t ticks = __rdtsc();
    acquire_lock();
    auto name_itr = name_set.find(name);
    if (name_itr == name_set.end()) {
        name_itr = name_set.insert(name).first;
    }
    const char* ev_name = name_itr->c_str();
    ev_store.emplace_back(ev_name, ticks, et::mark);
    release_lock();
}

void engine_impl::event_begin(const std::string name) {
    uint64_t ticks = __rdtsc();
    acquire_lock();
    auto name_itr = name_set.find(name);
    if (name_itr == name_set.end()) {
        name_itr = name_set.insert(name).first;
    }
    const char* ev_name = name_itr->c_str();
    ev_store.emplace_back(ev_name, ticks, et::begin);
    release_lock();
}

void engine_impl::event_end(const std::string name) {
    uint64_t ticks = __rdtsc();
    acquire_lock();
    auto name_itr = name_set.find(name);
    if (name_itr == name_set.end()) {
        name_itr = name_set.insert(name).first;
    }
    const char* ev_name = name_itr->c_str();
    ev_store.emplace_back(ev_name, ticks, et::end);
    release_lock();
}

void engine_impl::async_start(const std::string name) {
    uint64_t ticks = __rdtsc();
    acquire_lock();
    auto name_itr = name_set.find(name);
    if (name_itr == name_set.end()) {
        name_itr = name_set.insert(name).first;
    }
    const char* ev_name = name_itr->c_str();
    ev_store.emplace_back(ev_name, ticks, et::async_start);
    release_lock();
}

void engine_impl::async_finish(const std::string name) {
    uint64_t ticks = __rdtsc();
    acquire_lock();
    auto name_itr = name_set.find(name);
    if (name_itr == name_set.end()) {
        name_itr = name_set.insert(name).first;
    }
    const char* ev_name = name_itr->c_str();
    ev_store.emplace_back(ev_name, ticks, et::async_finish);
    release_lock();
}

void engine_impl::mem_tick(){
    mem_event ev = {0};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    ev.vm_peak_rss = (uint64_t)(pmc.PeakWorkingSetSize);
    ev.vm_rss = (uint64_t)(pmc.WorkingSetSize);
#elif
    std::ifstream status("/proc/self/status");
    if (!status.is_open())
        throw std::runtime_error("Can't read self status!");

    std::string line, title;
    while (std::getline(status, line)) {
        std::istringstream iss(line);
        iss >> title;
        if (title == "VmHWM:")
            iss >> ev.vm_peak_rss;
        else if (title == "VmRSS:")
            iss >> ev.vm_rss;
    }
    ev.vm_peak_rss *= 1024;
    ev.vm_rss *= 1024;
#endif
    ev.mp_temp_memory_used = _memory_pool.get_temp_memory_used();
    ev.mp_max_peak_device_memory_used = _memory_pool.get_max_peak_device_memory_used();
    ev.ts = __rdtsc();

    acquire_lock();
    mem_ev_store.push_back(ev);
    release_lock();
}

void engine_impl::logger_flush() {
    if (ev_store.empty() && mem_ev_store.empty())
        return;
    std::sort(ev_store.begin(), ev_store.end(), [](const stored_event& ev1, const stored_event& ev2)->bool {
        return ev1.value < ev2.value;
        });
    double tpus = _timer.ticks_per_usec() ;

    // flush all stored events to the JSON stream
    gpu::device_info_internal dinfo = get_device_info();
    std::string fname = "trace_";
    fname += dinfo.dev_name;
    fname += dinfo.dev_type == device_type::discrete_gpu? "_dGPU_" : "_iGPU_";
    fname += std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    fname += ".json";
    std::ofstream ofs(fname);
    rapidjson::OStreamWrapper osw(ofs);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
    writer.StartArray();
    for(auto& ev : ev_store) {
        // iterate all events with this name
        writer.StartObject();
        writer.String("cat");
        writer.String("cldnn");

        writer.String("pid");
        writer.Int(_timer.pid);

        writer.String("tid");
        writer.Int(_timer.tid);

        writer.String("ts");
        writer.Double( (double)(ev.value - _timer.start_ticks) / tpus);

        writer.String("ph");
        switch(ev.type) {
        case et::mark:
            writer.String("i");
        break;
        case et::begin:
            writer.String("B");
        break;
        case et::end:
            writer.String("E");
        break;
        case et::async_start:
            writer.String("b");
        break;
        case et::async_finish:
            writer.String("e");
        break;
        }
        writer.String("name");
        writer.String(ev.name);
        writer.EndObject();
    }
    //writer.EndArray();

    // cleanup event storage memory
    ev_store.clear();

    auto to_hex = [] (uint64_t num) {
        std::stringstream ss;
        ss << std::hex << num;
        return ss.str();
    };

    //writer.StartArray();
    size_t i = 0;
    for (auto& mev : mem_ev_store) {
        writer.StartObject();
            writer.String("cat");
            writer.String("disabled-by-default-memory-infra");

            writer.String("id");
            writer.String(to_hex(i++).c_str());

            writer.String("name");
            writer.String("periodic_interval");

            writer.String("ph");
            writer.String("v");

            writer.String("pid");
            writer.Int(_timer.pid);

            writer.String("tid");
            writer.Int(-1);

            writer.String("ts");
            writer.Double((double)(mev.ts - _timer.start_ticks) / tpus);

            //memory dump metrics
            writer.String("args");
            writer.StartObject();
                writer.String("dumps");
                writer.StartObject();
                    writer.String("process_totals");
                    writer.StartObject();
                        writer.String("peak_resident_set_size");
                        writer.String(to_hex(mev.vm_peak_rss).c_str());
                        writer.String("private_footprint_bytes");
                        writer.String(to_hex(mev.vm_rss).c_str());
                    writer.EndObject();

                    writer.String("allocators");
                    writer.StartObject();

                        writer.String("memory_pool temp_memory_used");
                        writer.StartObject();
                            writer.String("attrs");
                            writer.StartObject();
                                writer.String("size");
                                writer.StartObject();
                                    writer.String("type");
                                    writer.String("scalar");
                                    writer.String("units");
                                    writer.String("bytes");
                                    writer.String("value");
                                    writer.String(to_hex(mev.mp_temp_memory_used).c_str());
                                writer.EndObject();
                            writer.EndObject();
                            writer.String("guid");
                            writer.String("0000000000000002");
                        writer.EndObject();

                        writer.String("memory_pool max_peak_device_memory_used");
                        writer.StartObject();
                            writer.String("attrs");
                            writer.StartObject();
                                writer.String("size");
                                writer.StartObject();
                                    writer.String("type");
                                    writer.String("scalar");
                                    writer.String("units");
                                    writer.String("bytes");
                                    writer.String("value");
                                    writer.String(to_hex(mev.mp_max_peak_device_memory_used).c_str());
                                writer.EndObject();
                            writer.EndObject();
                            writer.String("guid");
                            writer.String("0000000000000003");
                        writer.EndObject();

                    writer.EndObject();
                writer.EndObject();
            writer.EndObject();
        writer.EndObject();
    }

    mem_ev_store.clear();
    writer.EndArray();
}
#endif
}  // namespace cldnn
