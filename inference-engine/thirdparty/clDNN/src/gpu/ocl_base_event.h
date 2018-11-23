#pragma once

#include "ocl_toolkit.h"

namespace cldnn { namespace gpu {

struct profiling_period_ocl_start_stop
{
    const char* name;
    cl_profiling_info start;
    cl_profiling_info stop;
};

struct ocl_base_event : virtual public event_impl
{
public:
    ocl_base_event(uint64_t queue_stamp = 0) : _queue_stamp(queue_stamp) {}
    uint64_t get_queue_stamp() const { return _queue_stamp; }
protected:
    uint64_t _queue_stamp = 0;
};

struct base_event : virtual public ocl_base_event
{
public:
    base_event(std::shared_ptr<gpu_toolkit> ctx, cl::Event const& ev, uint64_t queue_stamp = 0) : ocl_base_event(queue_stamp), _ctx(ctx), _event(ev)
    {}

    std::shared_ptr<gpu_toolkit> get_context() const { return _ctx; }
    cl::Event get() { return _event; }


private:
    std::shared_ptr<gpu_toolkit> _ctx;
    cl::Event _event;
    bool _callback_set = false;

    void set_ocl_callback();

    static void CL_CALLBACK ocl_event_completion_callback(cl_event, cl_int, void* me);

private:
    void wait_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(cldnn_event_handler, void*) override;
    bool get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) override;

    friend struct base_events;
};

struct base_events : virtual public ocl_base_event
{
public:
    base_events(std::shared_ptr<gpu_toolkit> ctx, std::vector<event_impl::ptr> const &ev) : ocl_base_event(0), _ctx(ctx), _events(ev)
    {
        uint64_t _queue_stamp_max = 0;
        for (size_t i = 0; i < ev.size(); i++)
        {
            auto * _base_event = dynamic_cast<base_event*>(ev[i].get());
            if (_base_event->get_queue_stamp() > _queue_stamp_max)
                _queue_stamp_max = _base_event->get_queue_stamp();
        }
        _queue_stamp = _queue_stamp_max;
    }

    std::shared_ptr<gpu_toolkit> get_context() const { return _ctx; }

private:
    void wait_impl() override;
    bool is_set_impl() override;

    bool get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) override;

    std::shared_ptr<gpu_toolkit> _ctx;
    std::vector<event_impl::ptr> _events;
};

}}
