// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "zero_infer_request.hpp"

using namespace intel_npu;

ZeroDevice::ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      _log("ZeroDevice", Logger::global().level()) {
    _log.debug("ZeroDevice::ZeroDevice init");
    _device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    // Get LUID info, if supported
    if (_initStructs->isExtensionSupported(std::string(ZE_DEVICE_LUID_EXT_NAME), ZE_MAKE_VERSION(1, 0))) {
        _device_luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
        _device_properties.pNext = &_device_luid;
    }
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_initStructs->getDevice(), &_device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility. For any other error, we just fall-back to device ID to assure backwards compatibilty with even
    // older drivers
    _pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_initStructs->getDevice(), &_pci_properties);
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
    if (_initStructs->getDriverVersion() >= gops_support_drv_version) {
        float gops = (_device_properties.coreClockRate / powf(1000, 3)) * _device_properties.numSlices *
                     _device_properties.physicalEUSimdWidth;
        _device_gops[ov::element::f32] = 0;
        _device_gops[ov::element::u8] = gops;
        _device_gops[ov::element::i8] = gops;
        _device_gops[ov::element::f16] = 0.5f * gops;
    }
}

std::string ZeroDevice::getName() const {
//    KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D
#define NPU_4000_DEVICE_ID   0x643E

    std::string name;
    switch (_device_properties.deviceId) {
    case NPU_3720_P_DEVICE_ID:
    case NPU_3720_S_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU3720;
        break;
    case NPU_4000_DEVICE_ID:
        name = ov::intel_npu::Platform::NPU4000;
        break;
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

uint64_t ZeroDevice::getAllocMemSize() const {
    ze_graph_memory_query_t query{};
    ze_result_t result = _initStructs->getGraphDdiTable().pfnQueryContextMemory(_initStructs->getContext(),
                                                                                ZE_GRAPH_QUERY_MEMORY_DDR,
                                                                                &query);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryContextMemory", result, _initStructs->getGraphDdiTable());

    return query.allocated;
}

uint64_t ZeroDevice::getTotalMemSize() const {
#define LEGACY_MAX_MEM_ALLOC_SIZE_BYTES (2147483648)  // 2GB in base-2

    ze_graph_memory_query_t query{};
    ze_result_t result = _initStructs->getGraphDdiTable().pfnQueryContextMemory(_initStructs->getContext(),
                                                                                ZE_GRAPH_QUERY_MEMORY_DDR,
                                                                                &query);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnQueryContextMemory", result, _initStructs->getGraphDdiTable());

    // For drivers with graph_extension < 1.9 we report fixed 2GB max allocation size (old drivers don't support more)
    // For drivers with graph_extension > 1.9 we report the value they return
    if (_initStructs->isExtensionSupported(std::string(ZE_GRAPH_EXT_NAME), ZE_MAKE_VERSION(1, 9))) {
        // we are safe here, can return the value directly from driver
        return query.total;
    }
#if defined(_WIN32) || defined(__CYGWIN__)
    // Special case for windows drivers with graph_extension v 1.8
    if (_initStructs->isExtensionSupported(std::string("ZE_extension_graph_1_8"), ZE_MAKE_VERSION(1, 8))) {
        // query here returns total system memory in KB, which we need to
        // divide by 2 (OS limitation) and convert to bytes
        return (query.total << 9);
    }
#endif

    // Default for older drivers: return 2GB
    return LEGACY_MAX_MEM_ALLOC_SIZE_BYTES;
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

ov::device::Type ZeroDevice::getDeviceType() const {
    return ov::device::Type::INTEGRATED;
}

std::shared_ptr<SyncInferRequest> ZeroDevice::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const Config& config) {
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, config);
}

void ZeroDevice::updateInfo(const ov::AnyMap& properties) {
    if (properties.count(ov::log::level.name()) != 0) {
        _log.setLevel(properties.at(ov::log::level.name()).as<ov::log::Level>());
    }
}
