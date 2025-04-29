// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_backend.hpp"

#include <vector>

#include "intel_npu/config/options.hpp"
#include "zero_device.hpp"

namespace intel_npu {

ZeroEngineBackend::ZeroEngineBackend() : _logger("ZeroEngineBackend", Logger::global().level()) {
    _logger.debug("ZeroEngineBackend - initialize started");

    _initStruct = ZeroInitStructsHolder::getInstance();

    auto device = std::make_shared<ZeroDevice>(_initStruct);
    _devices.emplace(std::make_pair(device->getName(), device));
    _logger.debug("ZeroEngineBackend - initialize completed");
}

uint32_t ZeroEngineBackend::getDriverVersion() const {
    return _initStruct->getDriverVersion();
}

uint32_t ZeroEngineBackend::getGraphExtVersion() const {
    return _initStruct->getGraphDdiTable().version();
}

bool ZeroEngineBackend::isCommandQueueExtSupported() const {
    return _initStruct->isExtensionSupported(std::string(ZE_COMMAND_QUEUE_NPU_EXT_NAME), ZE_MAKE_VERSION(1, 0));
}

bool ZeroEngineBackend::isLUIDExtSupported() const {
    return _initStruct->isExtensionSupported(std::string(ZE_DEVICE_LUID_EXT_NAME), ZE_MAKE_VERSION(1, 0));
}

ZeroEngineBackend::~ZeroEngineBackend() = default;

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice() const {
    if (_devices.empty()) {
        _logger.debug("ZeroEngineBackend - getDevice() returning empty list");
        return {};
    } else {
        _logger.debug("ZeroEngineBackend - getDevice() returning device list");
        return _devices.begin()->second;
    }
}

const std::shared_ptr<IDevice> ZeroEngineBackend::getDevice(const std::string& name) const {
    // First let's see if its a number (for device index) or a name
    int param = 0;
    try {
        param = std::stoi(name);
    } catch (...) {
        // seems like it is not a number
        param = -1;
    }
    // if it is not a number, we search for it
    if (param < 0) {
        if (_devices.find(name) != _devices.end()) {
            // string index exists, so we can return its Idevice
            return _devices.find(name)->second;
        } else {
            // try looking for a device with this name
            for (auto it = _devices.begin(); it != _devices.end(); ++it) {
                if (it->second->getFullDeviceName() == name) {
                    return it->second;
                }
            }
            // if the loop ends w/o return = no device with this name
            OPENVINO_THROW("Could not find available NPU device with the specified name: NPU.", name);
        }
    } else {
        // parameter is a number, but can be index or arch
        // arch numbers are at least 4 digit, so we assume numbers smaller than 4 digits are indexes,
        // while numbers bigger than 3 digis are arch names
        if (param < 1000) {
            // we asume it is an index, so return the device at that index (if exists)
            if (_devices.size() < (param + 1)) {
                // index does not exist
                OPENVINO_THROW("Could not find available NPU device with the specified index: NPU.", name);
            } else {
                // returning the n-th element (param)
                auto it = _devices.begin();
                std::advance(it, param);
                return it->second;
            }
        } else {
            // we asume this is an arch number, so we search for the first one
            for (auto it = _devices.begin(); it != _devices.end(); ++it) {
                if (it->second->getName() == name) {
                    return it->second;
                }
            }
            // if we got here, it means there is no device with that arch number
            OPENVINO_THROW("Could not find available NPU device with specified arch: NPU.", name);
        };
    }
    // if we got here without returning already, it means we did not find a device with requested name/index/arch
    OPENVINO_THROW("Could not find requested NPU device: NPU.", name);
}

const std::vector<std::string> ZeroEngineBackend::getDeviceNames() const {
    _logger.debug("ZeroEngineBackend - getDeviceNames started");
    std::vector<std::string> devicesNames;
    std::for_each(_devices.cbegin(), _devices.cend(), [&devicesNames](const auto& device) {
        devicesNames.push_back(device.first);
    });
    _logger.debug("ZeroEngineBackend - getDeviceNames completed and returning result");
    return devicesNames;
}

void* ZeroEngineBackend::getContext() const {
    return _initStruct->getContext();
}

void ZeroEngineBackend::updateInfo(const Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    if (_devices.size() > 0) {
        for (auto& dev : _devices) {
            dev.second->updateInfo(config);
        }
    }
}

const std::shared_ptr<ZeroInitStructsHolder> ZeroEngineBackend::getInitStructs() const {
    return _initStruct;
}

}  // namespace intel_npu
