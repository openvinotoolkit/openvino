//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/network_description.hpp"
#include "vm_runtime/virtual_machine.h"

#include <optional>

namespace intel_npu::vm {

std::optional<NetworkMetadata> getNetworkMetadata(npu_vm_module* module);

}  // namespace intel_npu::vm
