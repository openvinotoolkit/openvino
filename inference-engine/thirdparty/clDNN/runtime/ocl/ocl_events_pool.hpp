// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/utils.hpp"
#include "cldnn/runtime/event.hpp"

#include <iostream>
#include <vector>
#include <memory>

namespace cldnn {
namespace ocl {

template <typename Type,
          typename U = typename std::enable_if<meta::is_any_of<Type, base_event, user_event, base_events>::value>::type>
class event_pool_impl {
protected:
    event_pool_impl() = default;

    using type = Type;

    event::ptr get_from_pool(const cl::Context& ctx) {
        for (auto& ev : _events) {
            if (!ev->is_valid()) {
                ev->reset();
                return ev;
            }
        }
        auto ev_impl = std::make_shared<Type>(ctx);
        return allocate(ev_impl);
    }

    void reset_events() {
        for (auto& ev : _events) ev->reset();
    }

private:
    std::vector<event::ptr> _events;

    event::ptr allocate(const event::ptr& obj) {
        _events.emplace_back(obj);
        return _events.back();
    }
};

struct base_event_pool : event_pool_impl<base_event> {
    event::ptr get(const cl::Context& ctx, const cl::Event& ev, const uint64_t q_stamp) {
        auto ret = get_from_pool(ctx);
        std::dynamic_pointer_cast<type>(ret)->attach_ocl_event(ev, q_stamp);
        return ret;
    }
    void reset() { reset_events(); }
};

struct user_event_pool : event_pool_impl<user_event> {
    event::ptr get(const cl::Context& ctx, bool set = false) {
        auto ret = get_from_pool(ctx);
        dynamic_cast<type*>(ret.get())->attach_event(set);
        return ret;
    }
    void reset() { reset_events(); }
};

struct group_event_pool : event_pool_impl<base_events> {
    event::ptr get(const cl::Context& ctx, const std::vector<event::ptr>& deps) {
        auto ret_ev = get_from_pool(ctx);
        dynamic_cast<type*>(ret_ev.get())->attach_events(deps);
        return ret_ev;
    }
    void reset() { reset_events(); }
};

class events_pool {
public:
    events_pool() = default;

    event::ptr get_from_base_pool(const cl::Context& ctx, const cl::Event& ev, const uint64_t q_stamp) {
        return _base_pool.get(ctx, ev, q_stamp);
    }

    event::ptr get_from_user_pool(const cl::Context& ctx, bool set = false) {
        return _user_pool.get(ctx, set);
    }

    event::ptr get_from_group_pool(const cl::Context& ctx, const std::vector<event::ptr>& deps) {
        return _group_pool.get(ctx, deps);
    }

    void reset_events() {
        _base_pool.reset();
        _user_pool.reset();
        _group_pool.reset();
    }

private:
    base_event_pool _base_pool;
    user_event_pool _user_pool;
    group_event_pool _group_pool;
};
}  // namespace ocl
}  // namespace cldnn
