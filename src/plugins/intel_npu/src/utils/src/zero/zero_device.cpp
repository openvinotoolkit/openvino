// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_device.hpp"

#include "intel_npu/npu_private_properties.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"

using namespace intel_npu;

ZeroDevice::ZeroDevice(ze_device_handle_t deviceHandle,
                       const std::shared_ptr<ZeroDriver>& driver)
    : _zero_api(ZeroApi::get_instance()),
      _device_handle(deviceHandle),
      _driver(driver),
      _log("ZeroDevice", Logger::global().level()) {
    _log.debug("ZeroDevice::ZeroDevice init");
    _device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    // Get LUID info, if supported
    if (driver->getCapabilities().luidExtSupported) {
        _device_luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
        _device_properties.pNext = &_device_luid;
    }
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_device_handle, &_device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility. For any other error, we just fall-back to device ID to assure backwards compatibilty with even
    // older drivers
    _pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_device_handle, &_pci_properties);
    if (ZE_RESULT_SUCCESS == retpci) {
        // windows driver specific backwards compatibility
        if (_pci_properties.address.device == 0) {
            _log.warning("PCI information not available in driver. Falling back to deviceId");
            _pci_properties.address.device = _device_properties.deviceId;
        }
    } else {
        // general backwards compatibility
        _log.warning("PCI information not available in driver. Falling back to deviceId");
        _pci_properties.address.device = _device_properties.deviceId;
    }

    /// Calculate and store device GOPS with formula: frequency * number of tiles * ops per tile
    /// cross-OS backwards compatibilty: only calculate gops if driver supports it (version>x)
    uint32_t gops_support_drv_version = UINT32_MAX;
#if defined(_WIN32) || defined(__CYGWIN__)
    gops_support_drv_version = 2465;  /// Windows driver version which supports Gops calculations
#else                                 // _WIN32 || __CYGWIN__
    gops_support_drv_version = 1715354569;  /// Linux driver version which supports Gops calculations
#endif                                // _WIN32 || __CYGWIN__
    if (driver->getCapabilities().driverVersion >= gops_support_drv_version) {
        float gops = (_device_properties.coreClockRate / powf(1000, 3)) * _device_properties.numSlices *
                     _device_properties.physicalEUSimdWidth;
        _device_gops[ov::element::f32] = 0;
        _device_gops[ov::element::u8] = gops;
        _device_gops[ov::element::i8] = gops;
        _device_gops[ov::element::f16] = 0.5f * gops;
    }
}

std::string ZeroDevice::getName() const {
// Blacklisted deviceIDs
#define LEGACY_NPU_3000_DEVICE_ID 0x6240

// KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D
#define NPU_4000_DEVICE_ID   0x643E
#define NPU_5010_DEVICE_ID   0xB03E
#define NPU_5020_DEVICE_ID   0xFD3E
#define NPU_6010_DEVICE_ID   0xD71D

    std::string name;
    switch (_device_properties.deviceId) {
    case NPU_3720_P_DEVICE_ID:
    case NPU_3720_S_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU3720;
        break;
    case NPU_4000_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU4000;
        break;
    case NPU_5010_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU5010;
        break;
    case NPU_5020_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU5020;
        break;
    case NPU_6010_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU6010;
        break;
    case LEGACY_NPU_3000_DEVICE_ID:
        OPENVINO_THROW("[LEGACY] NPU device ID 0x", std::hex, _device_properties.deviceId, " is not supported!");
    default:
        name = ov::intel_npu::Platform::AUTO_DETECT;
    }

    return name;
}

std::string ZeroDevice::getFullDeviceName() const {
    return _device_properties.name;
}

IDevice::Uuid ZeroDevice::getUuid() const {
    Uuid uuid{};
    static_assert(sizeof(_device_properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match intel_npu::Uuid::uuid size");

    std::copy(std::begin(_device_properties.uuid.id), std::end(_device_properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

ov::device::LUID ZeroDevice::getLUID() const {
    ov::device::LUID luidstruct;
    // incompatibility check
    static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == ov::device::LUID::MAX_LUID_SIZE, "LUID size mismatch");
    for (int i = 0; i < ZE_MAX_DEVICE_LUID_SIZE_EXT; i++) {
        luidstruct.luid[i] = _device_luid.luid.id[i];
    }
    return luidstruct;
}

uint32_t ZeroDevice::getSubDevId() const {
    return _device_properties.subdeviceId;
}

uint32_t ZeroDevice::getMaxNumSlices() const {
    return _device_properties.numSlices;
}

ze_device_graph_properties_t ZeroDevice::getCompilerProperties() {
    std::lock_guard<std::mutex> lock(_compiler_mutex);
    if (!_compiler_properties.has_value()) {
        // Keep the optional disengaged unless the driver query succeeds
        ze_device_graph_properties_t compiler_properties = {};
        compiler_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
        THROW_ON_FAIL_FOR_LEVELZERO(
            "pfnDeviceGetGraphProperties",
            lockDriver()->getGraphDdiTable().pfnDeviceGetGraphProperties(_device_handle, &compiler_properties));
        _compiler_properties = compiler_properties;
    }
    return _compiler_properties.value();
}

uint32_t ZeroDevice::getCompilerVersion() {
    const auto properties = getCompilerProperties();
    return ZE_MAKE_VERSION(properties.compilerVersion.major, properties.compilerVersion.minor);
}

ov::device::PCIInfo ZeroDevice::getPciInfo() const {
    return ov::device::PCIInfo{_pci_properties.address.domain,
                               _pci_properties.address.bus,
                               _pci_properties.address.device,
                               _pci_properties.address.function};
}

std::map<ov::element::Type, float> ZeroDevice::getGops() const {
    return _device_gops;
}

std::shared_ptr<ZeroDriver> ZeroDevice::lockDriver() const {
    auto driver = _driver.lock();
    OPENVINO_ASSERT(driver != nullptr, "The driver of the device is no longer available");
    return driver;
}

std::shared_ptr<IDriver> ZeroDevice::getDriver() const {
    return lockDriver();
}

void* ZeroDevice::getDeviceHandle() const {
    return _device_handle;
}

ov::device::Type ZeroDevice::getDeviceType() const {
    return ov::device::Type::INTEGRATED;
}

void ZeroDevice::updateInfo(const ov::AnyMap& properties) {
    if (properties.count(ov::log::level.name()) != 0) {
        _log.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }
}

bool ZeroDevice::validateCompatibilityDescriptor(const std::string& compatibilityDescriptor) const {
    if (lockDriver()->getCapabilities().zeApiVersion < ZE_MAKE_VERSION(1, 16)) {
        OPENVINO_THROW("Compatibility descriptor validation is not supported by this driver");
    }

    ze_validate_runtime_requirements_output_t output = {};
    output.stype = ZE_STRUCTURE_TYPE_RUNTIME_REQUIREMENTS_OUTPUT;
    output.pNext = nullptr;

    const ze_result_t result =
        zeDeviceValidateRuntimeRequirements(_device_handle, compatibilityDescriptor.c_str(), &output);

    if (result != ZE_RESULT_SUCCESS) {
        _log.warning("zeDeviceValidateRuntimeRequirements returned error: 0x%x", static_cast<uint32_t>(result));
        return false;
    }

    // Only REQUIREMENTS_MET and MET_RECOMPILATION_ADVISABLE are treated as compatible.
    // NOT_APPLICABLE (the descriptor does not apply to this device) and REQUIREMENTS_NOT_MET are
    // intentionally treated as incompatible, since neither guarantees the blob runs correctly here
    return output.result == ZE_VALIDATE_RUNTIME_REQUIREMENTS_RESULT_REQUIREMENTS_MET ||
           output.result == ZE_VALIDATE_RUNTIME_REQUIREMENTS_RESULT_REQUIREMENTS_MET_RECOMPILATION_ADVISABLE;
}

IDevice::DeviceProperties ZeroDevice::getDeviceProperties() const {
    return {_device_properties.deviceId, _device_properties.subdeviceId, _device_properties.numSlices};
}
