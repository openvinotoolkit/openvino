#pragma once

#include "ocl_toolkit.h"

namespace cldnn { namespace gpu {

struct profiling_period_ocl_start_stop
{
    const char* name;
    cl_profiling_info start;
    cl_profiling_info stop;
};

struct base_event : virtual public event_impl
{
public:
    base_event(std::shared_ptr<gpu_toolkit> ctx, cl::Event const& ev, uint64_t queue_stamp = 0) : _ctx(ctx), _event(ev), _queue_stamp(queue_stamp)
    {}

    auto get_context() const { return _ctx; }
    cl::Event get() { return _event; }

    uint64_t get_queue_stamp() const { return _queue_stamp; }

private:
    std::shared_ptr<gpu_toolkit> _ctx;
    cl::Event _event;
    bool _callback_set = false;
    uint64_t _queue_stamp = 0;

    void set_ocl_callback();

    static void CL_CALLBACK ocl_event_completion_callback(cl_event, cl_int, void* me);

private:
    void wait_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(cldnn_event_handler, void*) override;
    bool get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) override;
};

}}
