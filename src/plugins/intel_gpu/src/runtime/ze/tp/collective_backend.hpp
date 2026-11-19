// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "ze/ze_common.hpp"

namespace cldnn {
namespace ze {
namespace tp {

/// The one collective backend of the GPU plugin: peer copies between the
/// devices of one Level Zero context.
inline constexpr const char* l0_p2p_backend = "L0_P2P";

/// The devices of a TP group, as far as the GPU plugin could resolve them.
struct DeviceGroup {
    std::vector<std::string> names;          // as given, e.g. "GPU.1"
    ze_driver_handle_t driver{nullptr};      // valid only without a problem
    std::vector<ze_device_handle_t> devices;  // valid only without a problem
    std::string problem;                     // why the names do not form a group; empty if they do
};

/// The backend names that serve `group`, in the order "AUTO" prefers them;
/// empty if none does.
std::vector<std::string> supported_backends(const DeviceGroup& group);

/// Empty if `backend` -- a name or "AUTO" -- serves `group`, otherwise why not.
/// Does not throw for a group or name it cannot serve.
std::string backend_support(const DeviceGroup& group, const std::string& backend);

/// The name `requested` stands for on `group` ("AUTO": the first supported
/// one). Throws with backend_support()'s reason if there is none.
std::string resolve_backend(const DeviceGroup& group, const std::string& requested);

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
