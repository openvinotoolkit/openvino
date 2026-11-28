// Copyright (C) 2018-2026 Intel Corporation
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

enum class AvailableBackends { LEVEL_ZERO };

class BackendsRegistry final {
public:
    BackendsRegistry();

    BackendsRegistry(const BackendsRegistry& other) = delete;
    BackendsRegistry(BackendsRegistry&& other) = delete;
    void operator=(const BackendsRegistry&) = delete;
    void operator=(BackendsRegistry&&) = delete;

    ~BackendsRegistry() = default;

    ov::SoPtr<IEngineBackend> getEngineBackend();
    /** @brief The factory that creates inference requests for the backend that getEngineBackend() returns, nullptr if
     * there is none */
    std::shared_ptr<IInferRequestFactory> getInferRequestFactory() const;

private:
    ov::SoPtr<IEngineBackend> initializeBackend(const AvailableBackends& backendName);
    void registerBackend(const AvailableBackends& backendName);

    std::unordered_map<AvailableBackends, ov::SoPtr<IEngineBackend>> _registeredBackends;
    std::unordered_map<AvailableBackends, std::shared_ptr<IInferRequestFactory>> _inferRequestFactories;

    Logger _logger;
};

}  // namespace intel_npu
