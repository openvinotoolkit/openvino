// Copyright (C) 2018-2025 Intel Corporation
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

uint32_t IEngineBackend::getDriverVersion() const {
    OPENVINO_THROW("Get NPU driver version is not supported with this backend");
}

uint32_t IEngineBackend::getGraphExtVersion() const {
    OPENVINO_THROW("Get NPU driver extension version is not supported with this backend");
}

void* IEngineBackend::getContext() const {
    OPENVINO_THROW("Get NPU context is not supported with this backend");
}

void IEngineBackend::registerOptions(OptionsDesc&) const {}

const std::shared_ptr<ZeroInitStructsHolder> IEngineBackend::getInitStructs() const {
    return nullptr;
}

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

uint64_t IDevice::getAllocMemSize() const {
    OPENVINO_THROW("Get AllocMemSize is not supported");
}

uint64_t IDevice::getTotalMemSize() const {
    OPENVINO_THROW("Get TotalMemSize is not supported");
}

ov::device::PCIInfo IDevice::getPciInfo() const {
    OPENVINO_THROW("Get PCIInfo is not supported");
}

ov::device::Type IDevice::getDeviceType() const {
    OPENVINO_THROW("Get DEVICE_TYPE is not supported");
}

std::map<ov::element::Type, float> IDevice::getGops() const {
    OPENVINO_THROW("Get DEVICE_GOPS is not supported");
}

ov::SoPtr<ov::IRemoteTensor> IDevice::createRemoteTensor(std::shared_ptr<ov::IRemoteContext>,
                                                         const ov::element::Type&,
                                                         const ov::Shape&,
                                                         const Config&,
                                                         ov::intel_npu::TensorType,
                                                         ov::intel_npu::MemType,
                                                         void*) {
    OPENVINO_THROW("Create Remote Tensor is not supported");
}

ov::SoPtr<ov::ITensor> IDevice::createHostTensor(std::shared_ptr<ov::IRemoteContext>,
                                                 const ov::element::Type&,
                                                 const ov::Shape&,
                                                 const Config&,
                                                 ov::intel_npu::TensorType) {
    OPENVINO_THROW("Create Host Tensor is not supported");
}

}  // namespace intel_npu
