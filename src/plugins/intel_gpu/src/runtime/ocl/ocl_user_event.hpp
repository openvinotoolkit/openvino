// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/profiling.hpp"
#include "openvino/util/util.hpp"
#include "ocl_base_event.hpp"
#include <memory>
#include <list>

OPENVINO_DISABLE_WARNING_MSVC_BEGIN(4250)  // Visual Studio warns us about inheritance via dominance but it's done intentionally
                                           // so turn it off

namespace cldnn {
namespace ocl {

struct ocl_user_event : public ocl_base_event {
    explicit ocl_user_event(const cl::Context& ctx, bool is_set = false)
    : ocl_base_event()
    , _ctx(ctx)
    , _event(_ctx) {
        if (is_set) {
            set();
        }
    }

    void set_impl() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;
    cl::Event& get() override { return _event; };

protected:
    cldnn::instrumentation::timer<> _timer;
    std::unique_ptr<cldnn::instrumentation::profiling_period_basic> _duration;
    const cl::Context& _ctx;
    cl::UserEvent _event;

private:
    void wait_impl() override;
    bool is_set_impl() override;
};

OPENVINO_DISABLE_WARNING_MSVC_END(4250)

}  // namespace ocl
}  // namespace cldnn
