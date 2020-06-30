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
#include "ocl_toolkit.h"
#include "ocl_base_event.h"
#include "ocl_user_event.h"
#include "command_queues_builder.h"
#include "events_pool.h"

#include <cassert>
#include <iomanip>
#include <ios>

#include <fstream>
#include <thread>
#include <string>
#include <vector>
#include <memory>
#include <utility>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {
std::string ndrange_to_string(cl::NDRange const& range) {
    std::string ret = "(";
    for (cl::size_type i = 0; i < range.dimensions(); ++i) ret += (!i ? "" : ", ") + std::to_string(range.get()[i]);

    ret += ")";
    return ret;
}

std::string events_list_to_string(std::vector<cldnn::event_impl::ptr> events) {
    std::string ret = "(";
    bool empty = true;
    for (auto& ev : events) {
        std::string id = "unk";
        if (auto* ocl_ev = dynamic_cast<cldnn::gpu::base_event*>(ev.get()))
            id = std::to_string(ocl_ev->get_queue_stamp());

        ret += (empty ? "" : ", ") + id;
        empty = false;
    }

    ret += ")";
    return ret;
}
}  // namespace

// static class memebers - pointers to dynamically obtained OpenCL extension functions
cl::PFN_clEnqueueAcquireMediaSurfacesINTEL cl::SharedSurfLock::pfn_acquire = NULL;
cl::PFN_clEnqueueReleaseMediaSurfacesINTEL cl::SharedSurfLock::pfn_release = NULL;
cl::PFN_clCreateFromMediaSurfaceINTEL cl::ImageVA::pfn_clCreateFromMediaSurfaceINTEL = NULL;
#ifdef WIN32
cl::PFN_clCreateFromD3D11Buffer cl::BufferDX::pfn_clCreateFromD3D11Buffer = NULL;
#endif

namespace cldnn {
namespace gpu {

ocl_error::ocl_error(cl::Error const& err)
    : std::runtime_error(err.what() + std::string(", error code: ") + std::to_string(err.err())) {}

std::mutex gpu_toolkit::cache_mutex;

std::shared_ptr<gpu_toolkit> gpu_toolkit::create(const device_impl& device, const configuration& cfg) {
    struct make_shared_wa : public gpu_toolkit {
        explicit make_shared_wa(const device_impl& device, const configuration& cfg)
            : gpu_toolkit(device, cfg) {}
    };
    try {
        auto ctx = std::make_shared<make_shared_wa>(device, cfg);
        ctx->add_network(0);
        return ctx;
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

struct gpu_toolkit::ocl_logger {
    std::ofstream _log_file;
};

gpu_toolkit::gpu_toolkit(const device_impl& device_impl, const configuration& config)
    : _configuration(config),
      _device(&device_impl),
      _neo_driver(strstr(get_device_version().c_str(), "NEO") ? true : false) {
    device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    device_cache_reader dc_reader(_configuration.tuning_cache_path);
    _device_cache = dc_reader.get();

    _logger = std::unique_ptr<ocl_logger>(new ocl_logger());
    if (logging_enabled()) {
        auto device_info = get_device_info();
        open_log() << "Engine configuration:\n"
                   << "    profiling: " << std::boolalpha << _configuration.enable_profiling << "\n"
                   << "    meaningful names: " << std::boolalpha << _configuration.meaningful_kernels_names << "\n"
                   << "    dump custom program: " << std::boolalpha << _configuration.dump_custom_program << "\n"
                   << "    device type: " << std::to_string(device_info.dev_type) << "\n"
                   << "    vendor type: " << std::hex << std::setfill('0') << std::setw(4) << std::right
                   << std::to_string(device_info.vendor_id) << "\n"
                   << std::dec << std::setfill(' ') << std::right
                   << "    compiler options: " << _configuration.compiler_options << "\n"
                   << "    single kernel name: " << _configuration.single_kernel_name << "\n"
                   << "    out-of-order: " << std::boolalpha << config.host_out_of_order << "\n"
                   << "    engine log: " << _configuration.log << "\n"
                   << "    sources dumps: " << _configuration.ocl_sources_dumps_dir << "\n"
                   << "\nEngine info:\n"
                   << "    cores count: " << device_info.cores_count << "\n"
                   << "    core frequencey: " << device_info.core_frequency << "\n"
                   << "    max work group size: " << device_info.max_work_group_size << "\n"
                   << "    local memory size: " << device_info.max_local_mem_size << "\n"
                   << "    fp16: " << std::boolalpha << (device_info.supports_fp16 != 0) << "\n"
                   << "    fp16 denorms: " << std::boolalpha << (device_info.supports_fp16_denorms != 0) << "\n"
                   << "    subgroups short: " << std::boolalpha << (device_info.supports_subgroups_short != 0) << "\n"
                   << "    local block io: " << std::boolalpha << device_info.supports_local_block_io << "\n"
                   << "    optimization hints: " << std::boolalpha << device_info.supports_optimization_hints << std::endl;
    }
}

gpu_queue& gpu_toolkit::get_command_queue(uint32_t id) {
    return _command_queues_w.at(id);
}

gpu_program_state& gpu_toolkit::get_program_state(uint32_t id) {
    std::lock_guard<std::mutex> lock(toolkit_mutex);
    return *_program_states.at(id);
}

void gpu_toolkit::add_program(uint32_t prog_id) {
    std::lock_guard<std::mutex> lock(toolkit_mutex);
    _program_states.emplace(std::make_pair(prog_id, std::make_shared<gpu_program_state>(*this, prog_id)));
}

void gpu_toolkit::remove_program(uint32_t prog_id) {
    std::lock_guard<std::mutex> lock(toolkit_mutex);
    auto state_iter = _program_states.find(prog_id);

    if (state_iter != _program_states.end()) {
        _program_states.erase(state_iter);
    }
}

kernels_cache& gpu_toolkit::get_kernels_cache(uint32_t prog_id) {
    return get_program_state(prog_id)._kernels_cache;
}

void gpu_toolkit::store_binaries(kernels_binaries_vector binaries, uint32_t prog_id) {
    get_program_state(prog_id)._binaries.push_back(binaries);
}

void gpu_toolkit::add_network(uint32_t net_id) {
    std::lock_guard<std::mutex> lock(toolkit_mutex);
    command_queues_builder queue_builder(context(), device(), _device->get_platform());
    queue_builder.set_profiling(_configuration.enable_profiling);
    queue_builder.set_out_of_order((_configuration.host_out_of_order && _neo_driver));

    bool priorty_extensions =
        extension_supported("cl_khr_priority_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_priority_mode(_configuration.priority_mode, priorty_extensions);

    bool throttle_extensions =
        extension_supported("cl_khr_throttle_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_throttle_mode(_configuration.throttle_mode, throttle_extensions);

    queue_builder.build();
    _command_queues_w.emplace(std::make_pair(net_id,
        gpu_queue(net_id, queue_builder.queue(), shared_from_this())));
}

void gpu_toolkit::remove_network(uint32_t net_id) {
    std::lock_guard<std::mutex> lock(toolkit_mutex);
    auto net_iter = _command_queues_w.find(net_id);
    if (net_iter != _command_queues_w.end()) {
        // net_iter->second.release_pending_memory();
        _command_queues_w.erase(net_iter);
    }
}

event_impl::ptr gpu_toolkit::enqueue_kernel(uint32_t queue_id,
                                            kernels_cache::kernel_type const& kern,
                                            cl::NDRange const& global,
                                            cl::NDRange const& local,
                                            std::vector<event_impl::ptr> const& deps) {
    return get_command_queue(queue_id).enqueue_kernel(kern, global, local, deps);
}

event_impl::ptr gpu_toolkit::enqueue_marker(uint32_t queue_id, std::vector<event_impl::ptr> const& deps) {
    return get_command_queue(queue_id).enqueue_marker(deps);
}

event_impl::ptr gpu_toolkit::group_events(uint32_t queue_id, std::vector<event_impl::ptr> const& deps) {
    return get_command_queue(queue_id).group_events(deps);
}

event_impl::ptr gpu_toolkit::create_user_event(uint32_t queue_id, bool set) {
    return get_command_queue(queue_id).create_user_event(set);
}

void gpu_toolkit::reset_events(uint32_t queue_id) { get_command_queue(queue_id).reset_events(); }

void gpu_toolkit::release_events_pool(uint32_t queue_id) { get_command_queue(queue_id).release_events_pool(); }

void gpu_toolkit::release_all_events_pools() {
    for (auto& queue : _command_queues_w) {
        queue.second.release_events_pool();
    }
}

void gpu_toolkit::flush(uint32_t queue_id) { get_command_queue(queue_id).flush(); }

void gpu_toolkit::release_pending_memory(uint32_t queue_id) { get_command_queue(queue_id).release_pending_memory(); }

void gpu_toolkit::wait_for_events(std::vector<event_impl::ptr> const& events) {
    std::vector<cl::Event> clevents;
    for (auto& ev : events)
        if (auto ocl_ev = dynamic_cast<base_event*>(ev.get()))
            clevents.push_back(ocl_ev->get());

    try {
        cl::WaitForEvents(clevents);
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

void gpu_toolkit::log(uint64_t id, std::string const& msg) {
    if (_configuration.log.empty())
        return;

    open_log() << "[" << id << "] " << msg << std::endl;
}

void gpu_toolkit::set_output_event(uint32_t queue_id, bool out_event) {
    get_command_queue(queue_id).set_output_event(out_event);
}

std::ofstream& gpu_toolkit::open_log() {
    if (!_logger->_log_file.is_open()) {
        _logger->_log_file.open(_configuration.log, std::ios::out | std::ios::trunc);
        if (!_logger->_log_file.good()) {
            _logger->_log_file.close();
            throw std::runtime_error("Could not initialize ocl_toolkit log file");
        }

        if (!_logger->_log_file.is_open()) {
            _logger->_log_file.close();
            throw std::runtime_error("Could not open ocl_toolkit log file '" + _configuration.log + "' for writing");
        }
    }

    return _logger->_log_file;
}

}  // namespace gpu

}  // namespace cldnn
