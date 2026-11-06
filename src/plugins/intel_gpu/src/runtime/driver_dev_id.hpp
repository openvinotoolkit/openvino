// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_gpu/runtime/device_info.hpp"

namespace cldnn {

/// @brief Fallback lookup of the Intel PCI device ID through OS facilities (SetupAPI on Windows, sysfs on Linux).
/// @param pci PCI location of the device the ID is requested for, if the runtime can report it.
/// @return The device ID or 0 if it cannot be determined.
/// @details If the OS entry for @p pci is found, its ID is returned. Otherwise, if all known Intel adapters share
/// one ID, that ID is returned. If they differ, a warning is logged and the last enumerated ID is returned.
int driver_dev_id(const std::optional<pci_bus_info>& pci);

}  // namespace cldnn
