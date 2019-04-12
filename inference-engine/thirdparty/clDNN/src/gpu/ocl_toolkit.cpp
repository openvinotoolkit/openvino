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

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
    #pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {
    std::string ndrange_to_string(cl::NDRange const& range)
    {
        std::string ret = "(";
        for (cl::size_type i = 0; i < range.dimensions(); ++i)
            ret += (!i ? "" : ", ") + std::to_string(range.get()[i]);

        ret += ")";
        return ret;
    }

    std::string events_list_to_string(std::vector<cldnn::event_impl::ptr> events)
    {
        std::string ret = "(";
        bool empty = true;
        for (auto& ev : events)
        {
            std::string id = "unk";
            if (auto* ocl_ev = dynamic_cast<cldnn::gpu::base_event*>(ev.get()))
                id = std::to_string(ocl_ev->get_queue_stamp());

            ret += (empty ? "" : ", ") + id;
            empty = false;
        }

        ret += ")";
        return ret;
    }
}

namespace cldnn { namespace gpu {

ocl_error::ocl_error(cl::Error const & err) : error(err.what() + std::string(", error code: ") + std::to_string(err.err()))
{
}

std::shared_ptr<gpu_toolkit> gpu_toolkit::create(const configuration & cfg)
{
    struct make_shared_wa : public gpu_toolkit { make_shared_wa(const configuration& cfg) : gpu_toolkit(cfg) {} };
    try {
        return std::make_shared<make_shared_wa>(cfg);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

struct gpu_toolkit::ocl_logger
{
    std::ofstream _log_file;
};

gpu_toolkit::gpu_toolkit(const configuration& config)
    : _configuration(config)
    , _ocl_builder(config)
    , _user_context(_ocl_builder.is_user_context())
    , _neo_driver(strstr(get_device_version().c_str(), "NEO") ? true : false)
    , _context(_ocl_builder.get_context())
    , _platform_id(_ocl_builder.get_platform_id())
    , _engine_info(*this)
    , _kernels_cache(*this)
    , _events_pool(new events_pool())
{
    _ocl_builder.get_device().getInfo(CL_DEVICE_EXTENSIONS, &_extensions);
    build_command_queues(config);

    _logger = std::unique_ptr<ocl_logger>(new ocl_logger());
    if (logging_enabled())
    {
        open_log()
            << "Engine configuration:\n"
            << "    profiling: "           << std::boolalpha << _configuration.enable_profiling << "\n"
            << "    meaningful names: "    << std::boolalpha << _configuration.meaningful_kernels_names << "\n"
            << "    dump custom program: " << std::boolalpha << _configuration.dump_custom_program << "\n"
            << "    device type: "         << std::to_string(_configuration.device_type) << "\n"
            << "    vendor type: "         << std::hex << std::setfill('0') << std::setw(4) << std::right
                                           << std::to_string(_configuration.device_vendor) << "\n"
                                           << std::dec << std::setfill(' ') << std::right
            << "    compiler options: "    << _configuration.compiler_options << "\n"
            << "    single kernel name: "  << _configuration.single_kernel_name << "\n"
            << "    out-of-order: "        << std::boolalpha << _configuration.host_out_of_order << "\n"
            << "    engine log: "          << _configuration.log << "\n"
            << "    sources dumps: "       << _configuration.ocl_sources_dumps_dir << "\n"
            << "\nEngine info:\n"
            << "    device id: "           << _engine_info.dev_id << "\n"
            << "    cores count: "         << _engine_info.cores_count << "\n"
            << "    core frequencey: "     << _engine_info.core_frequency << "\n"
            << "    max work group size: " << _engine_info.max_work_group_size << "\n"
            << "    local memory size: "   << _engine_info.max_local_mem_size << "\n"
            << "    fp16: "                << std::boolalpha << (_engine_info.supports_fp16 != 0) << "\n"
            << "    fp16 denorms: "        << std::boolalpha << (_engine_info.supports_fp16_denorms != 0) << "\n"
            << "    subgroups short: "     << std::boolalpha << (_engine_info.supports_subgroups_short != 0) << "\n"
            << "    used defined context: "<< std::boolalpha << _user_context << "\n"
            << std::endl;
    }
}

void gpu_toolkit::build_command_queues(const configuration& config)
{
    command_queues_builder queue_builder(_context, _ocl_builder.get_device(), _platform_id);
    queue_builder.set_profiling(config.enable_profiling);
    queue_builder.set_out_of_order((config.host_out_of_order && _neo_driver));

    bool priorty_extensions = extension_supported("cl_khr_priority_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_priority_mode(config.priority_mode, priorty_extensions);

    bool throttle_extensions = extension_supported("cl_khr_throttle_hints") && extension_supported("cl_khr_create_command_queue");
    queue_builder.set_throttle_mode(config.throttle_mode, throttle_extensions);

    queue_builder.build();

    _command_queue = queue_builder.queue();
}

event_impl::ptr gpu_toolkit::enqueue_kernel(cl::Kernel const& kern, cl::NDRange const& global, cl::NDRange const& local, std::vector<event_impl::ptr> const & deps)
{
    std::vector<cl::Event> dep_events;
    auto dep_events_ptr = &dep_events;
    if (!_configuration.host_out_of_order)
    {
        for (auto& dep : deps)
            if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                dep_events.push_back(ocl_ev->get());
    }
    else
    {
        dep_events_ptr = nullptr;
        sync_events(deps);
    }

    cl::Event ret_ev;
    try {
        if (!_configuration.host_out_of_order || _output_event || _configuration.enable_profiling)
        {
            _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, &ret_ev);
        }
        else
        {
            _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, nullptr);
        }
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }

    if (logging_enabled())
    {
        auto msg = kern.getInfo<CL_KERNEL_FUNCTION_NAME>() + ", gws: " + ndrange_to_string(global) + ", lws: " + ndrange_to_string(local) + ", deps: ";
        if (_configuration.host_out_of_order)
            msg += "()";
        else
            msg += events_list_to_string(deps);

        log(_queue_counter + 1, msg);
    }
    return _events_pool->get_from_base_pool(shared_from_this(), ret_ev, ++_queue_counter);
}

event_impl::ptr gpu_toolkit::enqueue_marker(std::vector<event_impl::ptr> const& deps)
{
    if (deps.empty())
        return _events_pool->get_from_user_pool(shared_from_this(), true);

    if (!_configuration.host_out_of_order)
    {
        cl::Event ret_ev;
        if (!enabled_single_kernel())
        {
            std::vector<cl::Event> dep_events;
            for (auto& dep : deps)
                if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                    dep_events.push_back(ocl_ev->get());

            try {
                _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
            }
            catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        }
        else
        {
            try {
                _command_queue.enqueueMarkerWithWaitList(nullptr, &ret_ev);
            }
            catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        }

        if (logging_enabled())
            log(_queue_counter + 1, "Marker with dependencies: " + events_list_to_string(deps));
        return _events_pool->get_from_base_pool(shared_from_this(), ret_ev, ++_queue_counter);
    }
    else
    {
        sync_events(deps);
        return _events_pool->get_from_base_pool(shared_from_this(), _last_barrier_ev, _last_barrier);
    }
}

event_impl::ptr gpu_toolkit::group_events(std::vector<event_impl::ptr> const& deps)
{
    return _events_pool->get_from_group_pool(shared_from_this(), deps);
}

event_impl::ptr gpu_toolkit::create_user_event(bool set)
{
    return _events_pool->get_from_user_pool(shared_from_this(), set);
}

void gpu_toolkit::reset_events()
{
    _events_pool->reset_events();
}

void gpu_toolkit::release_events_pool()
{
    _events_pool.reset();
}

void gpu_toolkit::flush()
{
    if (logging_enabled())
        log(0, "Flush");
    queue().flush();
}
void gpu_toolkit::release_pending_memory()
{
    /*
    TODO: Temp. solution, untill proper API calls from OpenCL are released.
    */
    void* ptr = nullptr;
    ptr = _mm_malloc(4096, 4096);
    queue().finish();
    try
    {
        cl::Buffer flusher(_context, CL_MEM_USE_HOST_PTR, (size_t)4096, ptr);
        flusher = (cl_mem)nullptr; //clear buffer
    }
    catch (...)
    {
        _mm_free(ptr);
        throw;
    }
    _mm_free(ptr);
}

void gpu_toolkit::wait_for_events(std::vector<event_impl::ptr> const & events)
{
    if (logging_enabled())
        log(0, "Wait for events: " + events_list_to_string(events));

    std::vector<cl::Event> clevents;
    for (auto& ev : events)
        if (auto ocl_ev = dynamic_cast<base_event*>(ev.get()))
            clevents.push_back(ocl_ev->get());

    try {
        cl::WaitForEvents(clevents);
    }
    catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

void gpu_toolkit::log(uint64_t id, std::string const & msg)
{
    if (_configuration.log.empty())
        return;

    open_log() << "[" << id << "] " << msg << std::endl;
}

void gpu_toolkit::sync_events(std::vector<event_impl::ptr> const & deps)
{
    if (!_configuration.host_out_of_order)
        return;

    bool needs_barrier = false;
    for (auto& dep : deps)
    {
        auto* ocl_ev = dynamic_cast<ocl_base_event*>(dep.get());
        if (ocl_ev->get_queue_stamp() > _last_barrier)
        {
            needs_barrier = true;
        }
    }

    if (needs_barrier)
    {
        try {
            if (_output_event)
            {
                _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
            }
            else
            {
                _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
            }

        }
        catch (cl::Error const& err) {
            throw ocl_error(err);
        }

        _last_barrier = ++_queue_counter;
        if (logging_enabled())
            log(_last_barrier, "Barrier");
    }
}

std::ofstream& gpu_toolkit::open_log()
{
    if (!_logger->_log_file.is_open())
    {
        _logger->_log_file.open(_configuration.log, std::ios::out | std::ios::trunc);
        if (!_logger->_log_file.good())
            throw std::runtime_error("Could not initialize ocl_toolkit log file");
        if (!_logger->_log_file.is_open())
        {
            throw std::runtime_error("Could not open ocl_toolkit log file '" + _configuration.log + "' for writing");
        }
    }

    return _logger->_log_file;
}

}

}
