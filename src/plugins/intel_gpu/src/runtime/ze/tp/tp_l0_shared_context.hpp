// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ze/ze_common.hpp"

namespace cldnn {
namespace ze {
namespace tp {

// Owns a single L0 context that spans all per-rank GPUs participating in
// tensor-parallel execution. Each rank's RemoteContextImpl adopts this context
// (with non-owning semantics) and holds a reference, as does the group's
// coordinator, so it lives until the last of them is released.
struct TPL0SharedContext {
    ze_driver_handle_t              driver{nullptr};
    std::vector<ze_device_handle_t> devices;  // rank order
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

/// The shared context over `devices` (rank order, all on `driver`), created on
/// first request and reused by every later one while anyone still holds it.
/// Every rank of a group asks for the same list, so they all land on one
/// context, which is what cross-device copies require. A different rank order
/// is a different group and gets a context of its own.
TPL0SharedContextPtr acquire_shared_context(ze_driver_handle_t driver, const std::vector<ze_device_handle_t>& devices);

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
