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

namespace {

    cl_device_type convert_configuration_device_type(configuration::device_types device_type)
    {
        cl_device_type device_types[] = {
                CL_DEVICE_TYPE_DEFAULT,
                CL_DEVICE_TYPE_CPU,
                CL_DEVICE_TYPE_GPU,
                CL_DEVICE_TYPE_ACCELERATOR };
        return device_types[device_type];
    }
 
    bool does_device_match_config(cl::Device const& dev, configuration const& config, std::list<std::string>& reasons)
    {
        auto dev_name = dev.getInfo<CL_DEVICE_NAME>();
        bool ok = true;

        auto dev_type = dev.getInfo<CL_DEVICE_TYPE>();

        if (dev_type != convert_configuration_device_type(config.device_type))
        {
            reasons.push_back(dev_name + ": invalid device type");
            ok = false;
        }

        auto vendor_id = dev.getInfo<CL_DEVICE_VENDOR_ID>();
        if (vendor_id != config.device_vendor)
        {
            reasons.push_back(dev_name + ": invalid vendor type");
            ok = false;
        }

        if (config.host_out_of_order)
        {
            auto queue_properties = dev.getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
            using cmp_t = std::common_type<decltype(queue_properties), typename std::underlying_type<cl::QueueProperties>::type>::type;
            if (!(static_cast<cmp_t>(queue_properties) & static_cast<cmp_t>(cl::QueueProperties::OutOfOrder)))
            {
                reasons.push_back(dev_name + ": missing out of order support");
                ok = false;
            }
        }

        return ok;
    }
}

cl::Device get_gpu_device(const configuration& config, cl_platform_id& platform_id)
{
    std::list<std::string> reasons;
    cl_uint n = 0;

    // Get number of platforms availible
    cl_int err = clGetPlatformIDs(0, NULL, &n);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
    }

    // Get platform list
    std::vector<cl_platform_id> platform_ids(n);
    err = clGetPlatformIDs(n, platform_ids.data(), NULL);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
    }

    for (auto& id : platform_ids)
    {
        cl::Platform platform = cl::Platform(id);
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (auto& d : devices)
        {
            if (does_device_match_config(d, config, reasons))
            {
                platform_id = id;
                return d;
            }
        }
    }

    if (reasons.empty())
        throw std::runtime_error("Could not find any OpenCL device");

    std::string error_msg = "No OpenCL device found which would match provided configuration:";
    for (const auto& reason : reasons)
        error_msg += "\n    " + reason;

    throw std::invalid_argument(std::move(error_msg));
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
    , _device(get_gpu_device(config, _platform_id))
    , _neo_driver(strstr(get_device_version().c_str(), "NEO") ? true : false)
    , _context(_device)
    , _command_queue(_context,
                     _device,
                     (config.enable_profiling
                        ? cl::QueueProperties::Profiling
                        : cl::QueueProperties::None) | 
                     (config.host_out_of_order && _neo_driver
                        ? cl::QueueProperties::OutOfOrder
                        : cl::QueueProperties::None))
    , _engine_info(*this)
    , _kernels_cache(*this)
{
    _device.getInfo(CL_DEVICE_EXTENSIONS, &_extensions);

    cl_command_queue_properties queue_properties =
        ((config.enable_profiling) ?
            CL_QUEUE_PROFILING_ENABLE :
            0) |
            ((config.host_out_of_order &&
                _neo_driver) ?
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE :
                0);

    if (_configuration.priority_mode != cldnn_priority_disabled)
    {
        if (extension_supported("cl_khr_priority_hints") &&
            extension_supported("cl_intelx_create_command_queue"))
            // TODO add check when caps will be availible (instead of cl_intelx_create_command_queue)
            //&& extension_supported("cl_khr_create_command_queue"))
        {
            // TODO: When cl_khr_create_command_queue will be availible the
            // function name will change to clCreateCommandQueueWithPropertiesKHR
            // in place of clCreateCommandQueueWithPropertiesINTEL.
#ifndef WIN32
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"
#endif
            pfn_clCreateCommandQueueWithPropertiesINTEL clCreateCommandQueueWithPropertiesINTEL =
                (pfn_clCreateCommandQueueWithPropertiesINTEL)clGetExtensionFunctionAddressForPlatform(
                    _platform_id,
                    "clCreateCommandQueueWithPropertiesINTEL");
#ifndef WIN32
    #pragma GCC diagnostic pop
#endif
            unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;

            switch (_configuration.priority_mode)
            {
            case cldnn_priority_high:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
                break;
            case cldnn_priority_low:
                cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
                break;
            default:
                break;
            }

            cl_int error_code = CL_SUCCESS;
            cl_queue_properties properties_low[] = {
                CL_QUEUE_PRIORITY_KHR, cl_queue_priority_value,
                CL_QUEUE_PROPERTIES, queue_properties,
                0 };

            _command_queue = clCreateCommandQueueWithPropertiesINTEL(
                _context.get(),
                _device.get(),
                properties_low,
                &error_code);

            if (error_code != CL_SUCCESS) {
                throw std::runtime_error("clCreateCommandQueueWithPropertiesINTEL error " + std::to_string(error_code));
            }
        }
        else
        {
            throw std::invalid_argument(
                "The param priority_mode is set in engine_configuration,\
                 but cl_khr_priority_hints or cl_khr_create_command_queue\
                 is not supported by current OpenCL implementation.");
        }
    }
    else
    {
        _command_queue = cl::CommandQueue(_context, _device, queue_properties);
    }

    if (_configuration.throttle_mode != cldnn_throttle_disabled)
    {
        if (extension_supported("cl_khr_throttle_hints"))
        {
            throw std::invalid_argument(
                "The param throttle_mode is set in engine_configuration,\
                 but it is placeholder for future use. It has no effect for now\
                 and should be set to cldnn_throttle_disabled");
        }
        else
        {
            throw std::invalid_argument(
                "The param throttle_mode is set in engine_configuration,\
                 but cl_khr_throttle_hints is not supported by current OpenCL implementation.");
        }
    }

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
            << "    configuration: "       << std::to_string(_engine_info.configuration) << "\n"
            << "    model: "               << std::to_string(_engine_info.model) << "\n"
            << "    architecture: "        << std::to_string(_engine_info.architecture) << "\n"
            << "    cores count: "         << _engine_info.cores_count << "\n"
            << "    core frequencey: "     << _engine_info.core_frequency << "\n"
            << "    max work group size: " << _engine_info.max_work_group_size << "\n"
            << "    local memory size: "   << _engine_info.max_local_mem_size << "\n"
            << "    fp16: "                << std::boolalpha << (_engine_info.supports_fp16 != 0) << "\n"
            << "    fp16 denorms: "        << std::boolalpha << (_engine_info.supports_fp16_denorms != 0) << "\n"
            << "    subgroups short: "     << std::boolalpha << (_engine_info.supports_subgroups_short != 0) << "\n"
            << std::endl;
    }
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

    return{ new base_event(shared_from_this(), ret_ev, ++_queue_counter), false };
}

event_impl::ptr gpu_toolkit::enqueue_marker(std::vector<event_impl::ptr> const& deps)
{
    if (deps.empty())
        return{ new user_event(shared_from_this(), true), false };

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

        return{ new base_event(shared_from_this(), ret_ev, ++_queue_counter), false };
    }
    else
    {
        sync_events(deps);
        return{ new base_event(shared_from_this(), _last_barrier_ev, _last_barrier), false };
    }
}

event_impl::ptr gpu_toolkit::group_events(std::vector<event_impl::ptr> const& deps)
{
    return{ new base_events(shared_from_this(), deps), false };
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
