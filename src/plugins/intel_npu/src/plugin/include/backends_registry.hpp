// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// System
#include <memory>
#include <set>
#include <vector>

#include "openvino/runtime/so_ptr.hpp"

// Plugin
#include "intel_npu/common/npu.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

enum class AvailableBackends { LEVEL_ZERO, IMD };

class BackendsRegistry final {
public:
    BackendsRegistry();

    BackendsRegistry(const BackendsRegistry& other) = delete;
    BackendsRegistry(BackendsRegistry&& other) = delete;
    void operator=(const BackendsRegistry&) = delete;
    void operator=(BackendsRegistry&&) = delete;

    ov::SoPtr<IEngineBackend> getEngineBackend();

private:
    ov::SoPtr<IEngineBackend> initializeBackend(const AvailableBackends& backendName);
    void registerBackend(const AvailableBackends& backendName);

    std::unordered_map<AvailableBackends, ov::SoPtr<IEngineBackend>> _registeredBackends;

    Logger _logger;
};

}  // namespace intel_npu
