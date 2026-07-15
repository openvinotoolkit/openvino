// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/zero_api.hpp"

namespace ov {
namespace tp {

// Owns a single L0 context that spans all per-rank GPUs participating in
// tensor-parallel execution. Each rank's intel_gpu RemoteContextImpl adopts
// this context (with non-owning semantics), so we must keep it alive until
// every rank's CompiledModel is released.
struct TPL0SharedContext {
    ze_driver_handle_t              driver{nullptr};
    std::vector<ze_device_handle_t> devices;
    ze_context_handle_t             context{nullptr};

    TPL0SharedContext() = default;
    TPL0SharedContext(const TPL0SharedContext&) = delete;
    TPL0SharedContext& operator=(const TPL0SharedContext&) = delete;

    ~TPL0SharedContext() {
        if (context) {
            ov::zeContextDestroy(context);
            context = nullptr;
        }
    }
};

using TPL0SharedContextPtr = std::shared_ptr<TPL0SharedContext>;

}  // namespace tp
}  // namespace ov
