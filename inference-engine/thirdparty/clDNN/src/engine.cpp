// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

namespace cldnn {

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
    result.kernels_cache_path = conf.kernels_cache_path;
    result.tuning_cache_path = conf.tuning_cache_path;
    result.n_threads = conf.n_threads;
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
                                            bool no_optimizations,
                                            bool is_body_program) {
    program_impl::ptr progr_impl{ new program_impl(*this, topology, options, is_internal, no_optimizations, is_body_program), false };
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
    if (!events.empty())
        _context->wait_for_events(events);
}

gpu::device_info_internal engine_impl::get_device_info() const { return _context->get_device_info(); }

void* engine_impl::get_user_context() const { return static_cast<void*>(_context->context().get()); }

void engine_impl::compile_program(program_impl& program) {
    auto& cache = _context->get_kernels_cache(program.get_id());
    if (!program.get_options().get<build_option_type::serialize_network>()->serialization_network_name.empty())
        cache.get_context().set_serialization_flag(true);
    // TODO: better compilation logic instead of a simple 'compile all'?
    cache.build_all();
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
}  // namespace cldnn
