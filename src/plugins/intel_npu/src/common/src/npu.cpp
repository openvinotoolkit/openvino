// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/npu.hpp"

#include "intel_npu/common/itt.hpp"
#include "openvino/util/shared_object.hpp"

namespace intel_npu {

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

std::shared_ptr<IDriver> IEngineBackend::getDriver() const {
    return nullptr;
}

std::shared_ptr<IContext> IEngineBackend::getContext() const {
    return nullptr;
}

void IEngineBackend::registerOptions(OptionsDesc&) const {}

IDevice::Uuid IDevice::getUuid() const {
    OPENVINO_THROW("Get UUID not supported");
}

ov::device::LUID IDevice::getLUID() const {
    OPENVINO_THROW("Get LUID not supported");
}

uint32_t IDevice::getSubDevId() const {
    OPENVINO_THROW("Get SubDevId is not supported");
}

uint32_t IDevice::getMaxNumSlices() const {
    OPENVINO_THROW("Get MaxNumSlices is not supported");
}

ov::device::PCIInfo IDevice::getPciInfo() const {
    OPENVINO_THROW("Get PCIInfo is not supported");
}

ov::device::Type IDevice::getDeviceType() const {
    OPENVINO_THROW("Get DEVICE_TYPE is not supported");
}

std::shared_ptr<IDriver> IDevice::getDriver() const {
    OPENVINO_THROW("Get driver is not supported");
}

void* IDevice::getDeviceHandle() const {
    OPENVINO_THROW("Get device handle is not supported");
}

std::map<ov::element::Type, float> IDevice::getGops() const {
    OPENVINO_THROW("Get DEVICE_GOPS is not supported");
}

}  // namespace intel_npu
