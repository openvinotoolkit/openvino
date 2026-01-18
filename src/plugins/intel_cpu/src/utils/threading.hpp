// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

constexpr const char* threadingType() {
    constexpr auto tbb_mode = OV_THREAD;

    switch (tbb_mode) {
    case OV_THREAD_TBB:
        return "TBB";
    case OV_THREAD_OMP:
        return "OMP";
    case OV_THREAD_SEQ:
        return "SEQ";
    case OV_THREAD_TBB_AUTO:
        return "TBB_AUTO";
    case OV_THREAD_TBB_ADAPTIVE:
        return "TBB_ADAPTIVE";
    default:  // never expected, but cannot throw in constexpr context
        return "UNKNOWN";
    }
}

}  // namespace ov::intel_cpu
