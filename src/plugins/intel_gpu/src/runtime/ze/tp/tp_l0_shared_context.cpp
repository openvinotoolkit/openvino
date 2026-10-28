// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_l0_shared_context.hpp"

#include <map>
#include <mutex>

namespace cldnn {
namespace ze {
namespace tp {

TPL0SharedContextPtr acquire_shared_context(ze_driver_handle_t driver, const std::vector<ze_device_handle_t>& devices) {
    OPENVINO_ASSERT(driver != nullptr && !devices.empty(), "[GPU][TP] a shared context needs a driver and devices");

    static std::mutex mutex;
    static std::map<std::vector<ze_device_handle_t>, std::weak_ptr<TPL0SharedContext>> contexts;

    std::lock_guard<std::mutex> lock(mutex);
    if (auto existing = contexts[devices].lock()) {
        OPENVINO_ASSERT(existing->driver == driver, "[GPU][TP] the devices of a group changed their driver");
        return existing;
    }

    OPENVINO_ASSERT(ov::zeContextCreateEx != nullptr,
                    "[GPU][TP] the Level Zero loader does not export zeContextCreateEx, which is required to "
                    "share one context across devices. Update the Level Zero loader.");

    auto shared = std::make_shared<TPL0SharedContext>();
    shared->driver = driver;
    shared->devices = devices;

    ze_context_desc_t desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    auto handles = devices;
    OV_ZE_EXPECT(ov::zeContextCreateEx(driver, &desc, static_cast<uint32_t>(handles.size()), handles.data(), &shared->context));

    // Expired entries are only dropped here, which keeps the map as small as
    // the number of distinct groups the process ever used.
    for (auto it = contexts.begin(); it != contexts.end();) {
        it = it->second.expired() ? contexts.erase(it) : std::next(it);
    }
    contexts[devices] = shared;
    return shared;
}

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
