// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends_registry.hpp"

#include <fstream>
#include <memory>

#include "intel_npu/common/device_helpers.hpp"
#include "zero_backend.hpp"

using namespace intel_npu;

namespace {

std::string backendToString(const AvailableBackends& backend) {
    switch (backend) {
    case AvailableBackends::LEVEL_ZERO:
        return "npu_level_zero_backend";
    default:
        return "unsupported backend";
    }
}

}  // namespace

namespace intel_npu {

BackendsRegistry::BackendsRegistry() : _logger("BackendsRegistry", Logger::global().level()) {
#if defined(OPENVINO_STATIC_LIBRARY)
    registerBackend(AvailableBackends::LEVEL_ZERO);
#else

#    if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    registerBackend(AvailableBackends::LEVEL_ZERO);
#    endif
#endif
}

ov::SoPtr<IEngineBackend> BackendsRegistry::initializeBackend(const AvailableBackends& backendName) {
    std::string backendNameToString = backendToString(backendName);
    try {
        switch (backendName) {
        case AvailableBackends::LEVEL_ZERO: {
            return ov::SoPtr<IEngineBackend>(std::make_shared<ZeroEngineBackend>());
        }
        default:
            _logger.warning("Invalid backend '%s'", backendNameToString.c_str());
        }
    } catch (const std::exception& ex) {
        _logger.warning("Got an error during backend '%s' loading : %s", backendNameToString.c_str(), ex.what());
    } catch (...) {
        _logger.warning("Got an unknown error during backend '%s' loading", backendNameToString.c_str());
    }

    return {nullptr};
}

void BackendsRegistry::registerBackend(const AvailableBackends& backendName) {
    if (_registeredBackends.find(backendName) != _registeredBackends.end()) {
        return;
    }

    const auto& backend = initializeBackend(backendName);

    if (backend != nullptr) {
        const auto backendDevices = backend->getDeviceNames();
        if (!backendDevices.empty()) {
            std::stringstream deviceNames;
            for (const auto& device : backendDevices) {
                deviceNames << device << " ";
            }
            _logger.debug("Register '%s' with devices '%s'", backend->getName().c_str(), deviceNames.str().c_str());
            _registeredBackends.emplace(backendName, backend);
        }
    }
}

ov::SoPtr<IEngineBackend> BackendsRegistry::getEngineBackend() {
#if defined(OPENVINO_STATIC_LIBRARY)
    if (_registeredBackends.find(AvailableBackends::LEVEL_ZERO) != _registeredBackends.end()) {
        _logger.info("Use '%s' backend for inference",
                     _registeredBackends.at(AvailableBackends::LEVEL_ZERO)->getName().c_str());
        return _registeredBackends.at(AvailableBackends::LEVEL_ZERO);
    }
#else

#    if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    if (_registeredBackends.find(AvailableBackends::LEVEL_ZERO) != _registeredBackends.end()) {
        _logger.info("Use '%s' backend for inference",
                     _registeredBackends.at(AvailableBackends::LEVEL_ZERO)->getName().c_str());
        return _registeredBackends.at(AvailableBackends::LEVEL_ZERO);
    }
#    endif
#endif

    _logger.warning("None of the backends were initialized successfully."
                    "Only offline compilation can be done!");

    return {nullptr};
}

}  // namespace intel_npu
