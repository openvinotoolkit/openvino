#pragma once

#include "ocl_base_event.h"

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable: 4250) //Visual Studio warns us about inheritance via dominance but it's done intentionally so turn it off
#endif

namespace cldnn { namespace gpu {

struct user_event : public base_event, public cldnn::user_event
{
    user_event(std::shared_ptr<gpu_toolkit> ctx, bool auto_set = false) : base_event(ctx, cl::UserEvent(ctx->context())), cldnn::user_event(auto_set)
    {
        if (auto_set)
            user_event::set_impl();
    }

    void set_impl() override;

    bool get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) override;

protected:
    cldnn::instrumentation::timer<> _timer;
    std::unique_ptr<cldnn::instrumentation::profiling_period_basic> _duration;
};

#ifdef _WIN32
#pragma warning(pop)
#endif

} }