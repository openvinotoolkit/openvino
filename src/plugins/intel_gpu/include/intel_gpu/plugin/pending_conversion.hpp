// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace ov::intel_gpu {

struct PendingConversion {
    PendingConversion() = default;
    PendingConversion(const PendingConversion&) = delete;
    PendingConversion& operator=(const PendingConversion&) = delete;

    void wait();
    // Call only after GPU completion has been confirmed.
    void clear_after_completion() noexcept;

    cldnn::memory::ptr staging;
    cldnn::event::ptr done;
};

}  // namespace ov::intel_gpu
