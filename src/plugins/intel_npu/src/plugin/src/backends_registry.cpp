// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends_registry.hpp"

#include <fstream>
#include <memory>

#include "intel_npu/common/device_helpers.hpp"
#include "zero_backend.hpp"

#if !defined(OPENVINO_STATIC_LIBRARY) && defined(ENABLE_IMD_BACKEND)
#    include "openvino/util/file_util.hpp"
#    include "openvino/util/shared_object.hpp"
#endif

using namespace intel_npu;

namespace {

std::string backendToString(const AvailableBackends& backend) {
    switch (backend) {
    case AvailableBackends::LEVEL_ZERO:
        return "npu_level_zero_backend";
    case AvailableBackends::IMD:
        return "npu_imd_backend";
    default:
        return "unsupported backend";
    }
}

#if !defined(OPENVINO_STATIC_LIBRARY) && defined(ENABLE_IMD_BACKEND)
std::shared_ptr<void> loadBackendLibrary(const std::string& libpath) {
#    if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    return ov::util::load_shared_object(ov::util::string_to_wstring(libpath).c_str());
#    else
    return ov::util::load_shared_object(libpath.c_str());
#    endif
}

std::shared_ptr<IEngineBackend> getBackend(std::shared_ptr<void> so) {
    static constexpr auto CreateFuncName = "CreateNPUEngineBackend";
    auto symbol = ov::util::get_symbol(so, CreateFuncName);

    using CreateFuncT = void (*)(std::shared_ptr<IEngineBackend>&);
    const auto createFunc = reinterpret_cast<CreateFuncT>(symbol);

    std::shared_ptr<IEngineBackend> backendPtr;
    createFunc(backendPtr);
    return backendPtr;
}

ov::SoPtr<IEngineBackend> loadBackend(const std::string& libpath) {
    auto backendSO = loadBackendLibrary(libpath);
    auto backend = getBackend(backendSO);

    return ov::SoPtr<IEngineBackend>(backend, backendSO);
}
#endif

}  // namespace

namespace intel_npu {

BackendsRegistry::BackendsRegistry() : _logger("BackendsRegistry", Logger::global().level()) {
#if defined(OPENVINO_STATIC_LIBRARY)
    registerBackend(AvailableBackends::LEVEL_ZERO);
#else
#    if defined(ENABLE_IMD_BACKEND)
    if (const auto* envVar = std::getenv("IE_NPU_USE_IMD_BACKEND")) {
        if (envVarStrToBool("IE_NPU_USE_IMD_BACKEND", envVar)) {
            registerBackend(AvailableBackends::IMD);
        }
    }
#    endif

#    if defined(_WIN32) || defined(_WIN64) || (defined(__linux__) && defined(__x86_64__))
    registerBackend(AvailableBackends::LEVEL_ZERO);
#    endif
#endif
}

ov::SoPtr<IEngineBackend> BackendsRegistry::initializeBackend(const AvailableBackends& backendName) {
    std::string backendNameToString = backendToString(backendName);
    try {
        switch (backendName) {
        case AvailableBackends::IMD: {
#if !defined(OPENVINO_STATIC_LIBRARY) && defined(ENABLE_IMD_BACKEND)
            const auto path =
                ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), backendNameToString + OV_BUILD_POSTFIX);
            if (!std::ifstream(path).good()) {
                _logger.debug("Backend '%s' at '%s' doesn't exist", backendNameToString.c_str(), path.c_str());
                break;
            }
            return loadBackend(path);
#endif
        }
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
#    if defined(ENABLE_IMD_BACKEND)
    if (const auto* envVar = std::getenv("IE_NPU_USE_IMD_BACKEND")) {
        if (envVarStrToBool("IE_NPU_USE_IMD_BACKEND", envVar)) {
            if (_registeredBackends.find(AvailableBackends::IMD) != _registeredBackends.end()) {
                _logger.info("Use '%s' backend for inference",
                             _registeredBackends.at(AvailableBackends::IMD)->getName().c_str());
                return _registeredBackends.at(AvailableBackends::IMD);
            }
        }
    }
#    endif

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
