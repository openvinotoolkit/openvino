#include "ocl_user_event.h"

using namespace cldnn::gpu;

void user_event::set_impl()
{
    //we simulate "wrapper_cast" here to cast from cl::Event to cl::UserEvent which both wrap the same cl_event
    //casting is valid as long as cl::UserEvent does not add any members to cl::Event (which it shouldn't)
    static_assert(sizeof(cl::UserEvent) == sizeof(cl::Event) && alignof(cl::UserEvent) == alignof(cl::Event), "cl::UserEvent does not match cl::Event");
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
    _duration = std::make_unique<cldnn::instrumentation::profiling_period_basic>(_timer.uptime());
}

bool user_event::get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) {
    if (_duration == nullptr)
    {
        return false;
    }
    
    info.push_back({ "duration", static_cast<uint64_t>(_duration->value().count()) });
    return true;
}