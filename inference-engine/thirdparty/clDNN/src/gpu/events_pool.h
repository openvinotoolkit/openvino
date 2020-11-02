/*
// Copyright (c) 2019 Intel Corporation
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
#include "refcounted_obj.h"
#include "event_impl.h"
#include "meta_utils.h"
#include <iostream>
#include <vector>
#include <memory>

namespace cldnn {
namespace gpu {

class gpu_toolkit;

template <typename Type,
          typename U = typename std::enable_if<meta::is_any_of<Type, base_event, user_event, base_events>::value>::type>
class event_pool_impl {
protected:
    event_pool_impl() = default;

    using type = Type;

    event_impl::ptr get_from_pool(std::shared_ptr<gpu_toolkit>& ctx) {
        for (auto& ev : _events) {
            if (!ev->is_valid()) {
                ev->reset();
                return ev;
            }
        }
        const event_impl::ptr ev_impl { new Type(ctx), false };
        return allocate(ev_impl);
    }

    void reset_events() {
        for (auto& ev : _events) ev->reset();
    }

private:
    std::vector<event_impl::ptr> _events;

    event_impl::ptr allocate(const event_impl::ptr& obj) {
        _events.emplace_back(obj);
        return _events.back();
    }
};

struct base_event_pool : event_pool_impl<base_event> {
    event_impl::ptr get(std::shared_ptr<gpu_toolkit>& ctx, const cl::Event& ev, const uint64_t q_stamp) {
        auto ret = get_from_pool(ctx);
        dynamic_cast<type*>(ret.get())->attach_ocl_event(ev, q_stamp);
        return ret;
    }
    void reset() { reset_events(); }
};

struct user_event_pool : event_pool_impl<user_event> {
    event_impl::ptr get(std::shared_ptr<gpu_toolkit>& ctx, bool set = false) {
        auto ret = get_from_pool(ctx);
        dynamic_cast<type*>(ret.get())->attach_event(set);
        return ret;
    }
    void reset() { reset_events(); }
};

struct group_event_pool : event_pool_impl<base_events> {
    event_impl::ptr get(std::shared_ptr<gpu_toolkit>& ctx, const std::vector<event_impl::ptr>& deps) {
        auto ret_ev = get_from_pool(ctx);
        dynamic_cast<type*>(ret_ev.get())->attach_events(deps);
        return ret_ev;
    }
    void reset() { reset_events(); }
};

class events_pool {
public:
    events_pool() = default;

    event_impl::ptr get_from_base_pool(std::shared_ptr<gpu_toolkit> ctx, const cl::Event& ev, const uint64_t q_stamp) {
        return _base_pool.get(ctx, ev, q_stamp);
    }

    event_impl::ptr get_from_user_pool(std::shared_ptr<gpu_toolkit> ctx, bool set = false) {
        return _user_pool.get(ctx, set);
    }

    event_impl::ptr get_from_group_pool(std::shared_ptr<gpu_toolkit> ctx, const std::vector<event_impl::ptr>& deps) {
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
}  // namespace gpu
}  // namespace cldnn
