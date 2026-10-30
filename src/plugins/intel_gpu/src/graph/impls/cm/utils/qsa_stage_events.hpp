// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/runtime/event.hpp"

namespace ov::intel_gpu::cm {

/// Test / benchmark hook: while armed, the QSA CM primitives (SparsePA, SparseSDPA) append the event of every
/// stage they dispatch, so a profiling-enabled queue can report the GPU time of the whole multi-kernel call
/// (the primitive itself only returns the event of its last stage).
struct QsaStageEvents {
    static bool& armed() {
        static bool value = false;
        return value;
    }
    static std::vector<cldnn::event::ptr>& events() {
        static std::vector<cldnn::event::ptr> value;
        return value;
    }
    static const cldnn::event::ptr& record(const cldnn::event::ptr& ev) {
        if (armed()) {
            events().push_back(ev);
        }
        return ev;
    }
};

}  // namespace ov::intel_gpu::cm
