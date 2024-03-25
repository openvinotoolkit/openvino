//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux.hpp"

#include "openvino/util/shared_object.hpp"
#include "vpux/al/itt.hpp"

namespace vpux {

const std::shared_ptr<IDevice> IEngineBackend::getDevice() const {
    OPENVINO_THROW("Default getDevice() not implemented");
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const std::string&) const {
    OPENVINO_THROW("Specific device search not implemented");
}
const std::shared_ptr<IDevice> IEngineBackend::getDevice(const ov::AnyMap&) const {
    OPENVINO_THROW("Get device based on params not implemented");
}
const std::vector<std::string> IEngineBackend::getDeviceNames() const {
    OPENVINO_THROW("Get all device names not implemented");
}

void IEngineBackend::registerOptions(OptionsDesc&) const {}

IDevice::Uuid IDevice::getUuid() const {
    OPENVINO_THROW("Get UUID not supported");
}

uint32_t IDevice::getSubDevId() const {
    OPENVINO_THROW("Get SubDevId is not supported");
}

uint32_t IDevice::getMaxNumSlices() const {
    OPENVINO_THROW("Get MaxNumSlices is not supported");
}

uint64_t IDevice::getAllocMemSize() const {
    OPENVINO_THROW("Get AllocMemSize is not supported");
}

uint64_t IDevice::getTotalMemSize() const {
    OPENVINO_THROW("Get TotalMemSize is not supported");
}

uint32_t IDevice::getDriverVersion() const {
    OPENVINO_THROW("Get VPU driver version is not supported with this backend");
}

}  // namespace vpux
