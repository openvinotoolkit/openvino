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

namespace cldnn {
namespace gpu {

ocl_error::ocl_error(cl::Error const& err)
    : std::runtime_error(err.what() + std::string(", error code: ") + std::to_string(err.err())) {}

std::shared_ptr<gpu_toolkit> gpu_toolkit::create(const configuration& cfg) {
    struct make_shared_wa : public gpu_toolkit {
        explicit make_shared_wa(const configuration& cfg) : gpu_toolkit(cfg) {}
    };
    try {
        auto ctx = std::make_shared<make_shared_wa>(cfg);
        ctx->build_command_queues();
        return ctx;
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

struct gpu_toolkit::ocl_logger {
    std::ofstream _log_file;
};

gpu_toolkit::gpu_toolkit(const configuration& config)
    : _configuration(config),
      _ocl_builder(config),
      _user_context(_ocl_builder.is_user_context()),
      _neo_driver(strstr(get_device_version().c_str(), "NEO") ? true : false),
      _context(_ocl_builder.get_context()),
      _platform_id(_ocl_builder.get_platform_id()),
      _engine_info(*this),
      _kernels_cache(*this) {
    _ocl_builder.get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    _logger = std::unique_ptr<ocl_logger>(new ocl_logger());
    if (logging_enabled()) {
        open_log() << "Engine configuration:\n"
                   << "    profiling: " << std::boolalpha << _configuration.enable_profiling << "\n"
                   << "    meaningful names: " << std::boolalpha << _configuration.meaningful_kernels_names << "\n"
                   << "    dump custom program: " << std::boolalpha << _configuration.dump_custom_program << "\n"
                   << "    device type: " << std::to_string(_configuration.device_type) << "\n"
                   << "    vendor type: " << std::hex << std::setfill('0') << std::setw(4) << std::right
                   << std::to_string(_configuration.device_vendor) << "\n"
                   << std::dec << std::setfill(' ') << std::right
                   << "    compiler options: " << _configuration.compiler_options << "\n"
                   << "    single kernel name: " << _configuration.single_kernel_name << "\n"
                   << "    out-of-order: " << std::boolalpha << _configuration.host_out_of_order << "\n"
                   << "    engine log: " << _configuration.log << "\n"
                   << "    sources dumps: " << _configuration.ocl_sources_dumps_dir << "\n"
                   << "\nEngine info:\n"
                   << "    device id: " << _engine_info.dev_id << "\n"
                   << "    cores count: " << _engine_info.cores_count << "\n"
                   << "    core frequencey: " << _engine_info.core_frequency << "\n"
                   << "    max work group size: " << _engine_info.max_work_group_size << "\n"
                   << "    local memory size: " << _engine_info.max_local_mem_size << "\n"
                   << "    fp16: " << std::boolalpha << (_engine_info.supports_fp16 != 0) << "\n"
                   << "    fp16 denorms: " << std::boolalpha << (_engine_info.supports_fp16_denorms != 0) << "\n"
                   << "    subgroups short: " << std::boolalpha << (_engine_info.supports_subgroups_short != 0) << "\n"
                   << "    used defined context: " << std::boolalpha << _user_context << "\n"
                   << std::endl;
    }
}

void gpu_toolkit::build_command_queues() {
    command_queues_builder queue_builder(_context, _ocl_builder.get_device(), _platform_id);
    queue_builder.set_profiling(_configuration.enable_profiling);
    queue_builder.set_out_of_order((_configuration.host_out_of_order && _neo_driver));

    bool priorty_extensions =
        extension_supported("cl_khr_priority_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_priority_mode(_configuration.priority_mode, priorty_extensions);

    bool throttle_extensions =
        extension_supported("cl_khr_throttle_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_throttle_mode(_configuration.throttle_mode, throttle_extensions);

    queue_builder.build();

    for (uint16_t s = 0; s < _configuration.queues_num; s++) {
        _command_queues_w.emplace_back(s, queue_builder.queue(), shared_from_this());
    }
}

event_impl::ptr gpu_toolkit::enqueue_kernel(uint16_t queue_id,
                                            cl::Kernel const& kern,
                                            cl::NDRange const& global,
                                            cl::NDRange const& local,
                                            std::vector<event_impl::ptr> const& deps) {
    return _command_queues_w[queue_id].enqueue_kernel(kern, global, local, deps);
}

event_impl::ptr gpu_toolkit::enqueue_marker(uint16_t queue_id, std::vector<event_impl::ptr> const& deps) {
    return _command_queues_w[queue_id].enqueue_marker(deps);
}

event_impl::ptr gpu_toolkit::group_events(uint16_t queue_id, std::vector<event_impl::ptr> const& deps) {
    return _command_queues_w[queue_id].group_events(deps);
}

event_impl::ptr gpu_toolkit::create_user_event(uint16_t queue_id, bool set) {
    return _command_queues_w[queue_id].create_user_event(set);
}

void gpu_toolkit::reset_events(uint16_t queue_id) { _command_queues_w[queue_id].reset_events(); }

void gpu_toolkit::release_events_pool(uint16_t queue_id) { _command_queues_w[queue_id].release_events_pool(); }

void gpu_toolkit::flush(uint16_t queue_id) { _command_queues_w[queue_id].flush(); }

void gpu_toolkit::release_pending_memory(uint16_t queue_id) { _command_queues_w[queue_id].release_pending_memory(); }

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

void gpu_toolkit::set_output_event(uint16_t queue_id, bool out_event) {
    _command_queues_w[queue_id].set_output_event(out_event);
}

std::ofstream& gpu_toolkit::open_log() {
    if (!_logger->_log_file.is_open()) {
        _logger->_log_file.open(_configuration.log, std::ios::out | std::ios::trunc);
        if (!_logger->_log_file.good())
            throw std::runtime_error("Could not initialize ocl_toolkit log file");
        if (!_logger->_log_file.is_open()) {
            throw std::runtime_error("Could not open ocl_toolkit log file '" + _configuration.log + "' for writing");
        }
    }

    return _logger->_log_file;
}

}  // namespace gpu

}  // namespace cldnn
