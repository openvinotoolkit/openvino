// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "ocl_toolkit.h"
#include "event_impl.h"
#include <vector>
#include <memory>

namespace cldnn {
namespace gpu {
class events_waiter : public context_holder {
public:
    explicit events_waiter(std::shared_ptr<gpu_toolkit> context) : context_holder(context) {}

    event_impl::ptr run(uint32_t queue_id, const std::vector<event_impl::ptr>& dependencies) {
        if (dependencies.size() == 1)
            return dependencies[0];

        return context()->enqueue_marker(queue_id, dependencies);
    }
};
}  // namespace gpu
}  // namespace cldnn
