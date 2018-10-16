#include "ocl_base_event.h"

#include <cassert>

using namespace cldnn;
using namespace gpu;

namespace {
    bool is_event_profiled(const cl::Event& event)
    {
        if (event() != nullptr)
        {
            auto queue = event.getInfo<CL_EVENT_COMMAND_QUEUE>();
            if (queue() != nullptr)
            {
                return (queue.getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_PROFILING_ENABLE) != 0;
            }
        }
        return false;
    }
}

void CL_CALLBACK base_event::ocl_event_completion_callback(cl_event, cl_int, void* me)
{
    reinterpret_cast<base_event*>(me)->_set = true;
    reinterpret_cast<base_event*>(me)->call_handlers();
}

void base_event::set_ocl_callback()
{
    if (_callback_set)
        return;

    if (_event.get() != nullptr)
    {
        _event.setCallback(CL_COMPLETE, ocl_event_completion_callback, this);
        _callback_set = true;
    }
}

void base_event::wait_impl()
{
    if (_event.get() != nullptr)
    {
        _event.wait();
        if (get_context()->logging_enabled())
        {
            get_context()->log(0, "Wait for event: " + std::to_string(_queue_stamp));
        }
    }
}

bool base_event::is_set_impl()
{
    if (_event.get() != nullptr)
    {
        return _event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
    }
    return true;
}

bool base_event::add_event_handler_impl(cldnn_event_handler, void*)
{
    set_ocl_callback();
    return true;
}

bool base_event::get_profiling_info_impl(std::list<cldnn_profiling_interval>& info)
{
    if (!is_event_profiled(_event))
        return true;

    static const std::vector<profiling_period_ocl_start_stop> profiling_periods
    {
        { "submission", CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT },
        { "starting",   CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START },
        { "executing",  CL_PROFILING_COMMAND_START,  CL_PROFILING_COMMAND_END },
    };


    for (auto& period : profiling_periods)
    {
        cl_ulong start;
        cl_ulong end;

        _event.getProfilingInfo(period.start, &start);
        _event.getProfilingInfo(period.stop, &end);

        info.push_back({ period.name, end - start });
    }

    return true;
}
