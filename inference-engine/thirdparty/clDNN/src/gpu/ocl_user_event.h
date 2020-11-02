/*
// Copyright (c) 2018 Intel Corporation
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

#pragma once

#include "ocl_base_event.h"
#include "api/profiling.hpp"
#include <memory>
#include <list>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4250)  // Visual Studio warns us about inheritance via dominance but it's done intentionally
                                 // so turn it off
#endif

namespace cldnn {
namespace gpu {

struct user_event : public base_event, public cldnn::user_event {
    explicit user_event(std::shared_ptr<gpu_toolkit> ctx) : base_event(ctx), cldnn::user_event(false) {}

    void set_impl() override;
    void attach_event(bool set) {
        _event = cl::UserEvent(get_context()->context());
        // we need to reset the timer(since attach_ocl_event is called only when this object is being reused)
        _timer = cldnn::instrumentation::timer<>();
        if (set) {
            set_impl();
            _set = set;
        }
    }
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    cldnn::instrumentation::timer<> _timer;
    std::unique_ptr<cldnn::instrumentation::profiling_period_basic> _duration;
};

#ifdef _WIN32
#pragma warning(pop)
#endif

}  // namespace gpu
}  // namespace cldnn
