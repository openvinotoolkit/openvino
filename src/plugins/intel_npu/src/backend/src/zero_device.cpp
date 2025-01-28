// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include "intel_npu/common/itt.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_host_tensor.hpp"
#include "zero_infer_request.hpp"
#include "zero_remote_tensor.hpp"

using namespace intel_npu;

ZeroDevice::ZeroDevice(const std::shared_ptr<ZeroInitStructsHolder>& initStructs)
    : _initStructs(initStructs),
      log("ZeroDevice", Logger::global().level()) {
    log.debug("ZeroDevice::ZeroDevice init");
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

    // Get LUID info, if supported
    if (_initStructs->isExtensionSupported(std::string(ZE_DEVICE_LUID_EXT_NAME), ZE_MAKE_VERSION(1, 0))) {
        device_luid.stype = ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES;
        device_properties.pNext = &device_luid;
    }
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                zeDeviceGetProperties(_initStructs->getDevice(), &device_properties));

    // Query PCI information
    // Older drivers do not have this implementend. Linux driver returns NOT_IMPLEMENTED, while windows driver returns
    // zero values. If this is detected, we populate only device with ID from device_properties for backwards
    // compatibility. For any other error, we just fall-back to device ID to assure backwards compatibilty with even
    // older drivers
    pci_properties.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    ze_result_t retpci = zeDevicePciGetPropertiesExt(_initStructs->getDevice(), &pci_properties);
    if (ZE_RESULT_SUCCESS == retpci) {
        // windows driver specific backwards compatibility
        if (pci_properties.address.device == 0) {
            log.warning("PCI information not available in driver. Falling back to deviceId");
            pci_properties.address.device = device_properties.deviceId;
        }
    } else {
        // general backwards compatibility
        log.warning("PCI information not available in driver. Falling back to deviceId");
        pci_properties.address.device = device_properties.deviceId;
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
        float gops = (device_properties.coreClockRate / powf(1000, 3)) * device_properties.numSlices *
                     device_properties.physicalEUSimdWidth;
        device_gops[ov::element::f32] = 0;
        device_gops[ov::element::u8] = gops;
        device_gops[ov::element::i8] = gops;
        device_gops[ov::element::f16] = 0.5f * gops;
    }
}

std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, ov::SoPtr<ov::ITensor>> ZeroDevice::runInit(
    const std::shared_ptr<IGraph>& initGraph,
    const std::shared_ptr<const ov::Model>& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const Config& config) {
    std::unordered_map<size_t, std::shared_ptr<ov::ITensor>> constantIdToTensorData;
    std::vector<std::vector<std::shared_ptr<ov::ITensor>>> inputTensors;
    std::vector<std::shared_ptr<ov::ITensor>> outputTensors;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> outputTensorsMap;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_memcpy;
    std::chrono::steady_clock::time_point end_memcpy;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;
    long long memcpy_duration = 0;

    // Match the inputs of the "init" model with the Constant nodes of the original model
    begin = std::chrono::steady_clock::now();
    size_t constantIndex = 0;
    for (auto&& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        std::shared_ptr<ov::ITensor> tensor = ov::make_tensor(constantNode->get_element_type(),
                                                              constantNode->get_shape(),
                                                              const_cast<void*>(constantNode->get_data_ptr()),
                                                              constantNode->get_strides());
        constantIdToTensorData.emplace(constantIndex++, tensor);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "getting constant IDs " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;

    begin = std::chrono::steady_clock::now();
    size_t initInputsByteSize = 0;

    for (const IODescriptor& descriptor : initGraph->get_metadata().inputs) {
        initInputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initInputsTensor =
        createHostTensor(context._ptr, ov::element::Type_t::u8, ov::Shape({initInputsByteSize}), config);
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init inputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    size_t offset = 0;
    for (const IODescriptor& descriptor : initGraph->get_metadata().inputs) {
        const size_t id = std::stoi(descriptor.nameFromCompiler);
        auto currentInputBufferLocation = static_cast<unsigned char*>(initInputsTensor->data()) + offset;
        const size_t currentInputSize =
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
        OPENVINO_ASSERT(constantIdToTensorData.count(id), "Mismatch between weights IDs and parsed inputs");
        OPENVINO_ASSERT(constantIdToTensorData.at(id)->get_byte_size() == currentInputSize,
                        "Byte size mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constantIdToTensorData.at(id)->get_element_type() == descriptor.precision,
                        "Precision mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constantIdToTensorData.at(id)->get_shape() == descriptor.shapeFromCompiler.to_shape(),
                        "Shape mismatch for ",
                        descriptor.nameFromCompiler);

        begin_memcpy = std::chrono::steady_clock::now();
        std::memcpy(currentInputBufferLocation, constantIdToTensorData.at(id)->data(), currentInputSize);
        end_memcpy = std::chrono::steady_clock::now();
        memcpy_duration =
            memcpy_duration + std::chrono::duration_cast<std::chrono::microseconds>(end_memcpy - begin_memcpy).count();

        inputTensors.push_back({ov::make_tensor(constantIdToTensorData.at(id)->get_element_type(),
                                                constantIdToTensorData.at(id)->get_shape(),
                                                currentInputBufferLocation,
                                                constantIdToTensorData.at(id)->get_strides())});
        offset += currentInputSize;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Setting init inputs " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;
    std::cout << "Memcpy duration " << memcpy_duration << "[microseconds]" << std::endl;

    begin = std::chrono::steady_clock::now();
    size_t initOutputsByteSize = 0;

    for (const IODescriptor& descriptor : initGraph->get_metadata().outputs) {
        initOutputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initOutputsTensor =
        createHostTensor(context._ptr, ov::element::Type_t::u8, ov::Shape({initOutputsByteSize}), config);
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init outputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    offset = 0;
    for (const IODescriptor& descriptor : initGraph->get_metadata().outputs) {
        const auto currentOutputBufferLocation = static_cast<unsigned char*>(initOutputsTensor->data()) + offset;

        const ov::SoPtr<ov::ITensor> hostTensor =
            ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.to_shape(), currentOutputBufferLocation);

        outputTensors.push_back(hostTensor._ptr);
        outputTensorsMap.emplace(descriptor.nameFromCompiler, hostTensor._ptr);
        offset +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Creating output tensors "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
              << std::endl;

    auto progilingPool = zeroProfiling::ProfilingPool(_initStructs, initGraph, zeroProfiling::POOL_SIZE);
    auto profilingQuery = zeroProfiling::ProfilingQuery(_initStructs, 0);

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties", zeDeviceGetProperties(_initStructs->getDevice(), &properties));

    auto groupOrdinal = zeroUtils::findGroupOrdinal(_initStructs->getDevice(), properties);
    const auto pipeline = std::make_unique<Pipeline>(config,
                                                     _initStructs,
                                                     initGraph,
                                                     progilingPool,
                                                     profilingQuery,
                                                     nullptr,
                                                     inputTensors,
                                                     outputTensors,
                                                     groupOrdinal);
    begin = std::chrono::steady_clock::now();
    pipeline->push();
    pipeline->pull();
    end = std::chrono::steady_clock::now();
    std::cout << "Running the pipeline " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;

    return std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, ov::SoPtr<ov::ITensor>>(
        outputTensorsMap,
        initOutputsTensor);
}

std::string ZeroDevice::getName() const {
//    KMD is setting usDeviceID from VpuFamilyID.h
#define NPU_3720_P_DEVICE_ID 0x7D1D
#define NPU_3720_S_DEVICE_ID 0xAD1D
#define NPU_4000_DEVICE_ID   0x643E

    std::string name;
    switch (device_properties.deviceId) {
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
    return device_properties.name;
}

IDevice::Uuid ZeroDevice::getUuid() const {
    Uuid uuid{};
    static_assert(sizeof(device_properties.uuid.id) == uuid.uuid.size(),
                  "ze_device_uuid_t::id size doesn't match intel_npu::Uuid::uuid size");

    std::copy(std::begin(device_properties.uuid.id), std::end(device_properties.uuid.id), std::begin(uuid.uuid));

    return uuid;
}

ov::device::LUID ZeroDevice::getLUID() const {
    ov::device::LUID luidstruct;
    // incompatibility check
    static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == ov::device::LUID::MAX_LUID_SIZE, "LUID size mismatch");
    for (int i = 0; i < ZE_MAX_DEVICE_LUID_SIZE_EXT; i++) {
        luidstruct.luid[i] = device_luid.luid.id[i];
    }
    return luidstruct;
}

uint32_t ZeroDevice::getSubDevId() const {
    return device_properties.subdeviceId;
}

uint32_t ZeroDevice::getMaxNumSlices() const {
    return device_properties.numSlices;
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
    return ov::device::PCIInfo{pci_properties.address.domain,
                               pci_properties.address.bus,
                               pci_properties.address.device,
                               pci_properties.address.function};
}

std::map<ov::element::Type, float> ZeroDevice::getGops() const {
    return device_gops;
}

ov::device::Type ZeroDevice::getDeviceType() const {
    return ov::device::Type::INTEGRATED;
}

std::shared_ptr<SyncInferRequest> ZeroDevice::createInferRequest(
    const std::shared_ptr<const ICompiledModel>& compiledModel,
    const Config& config) {
    return std::make_shared<ZeroInferRequest>(_initStructs, compiledModel, config);
}

ov::SoPtr<ov::IRemoteTensor> ZeroDevice::createRemoteTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                            const ov::element::Type& element_type,
                                                            const ov::Shape& shape,
                                                            const Config& config,
                                                            ov::intel_npu::TensorType tensor_type,
                                                            ov::intel_npu::MemType mem_type,
                                                            const void* mem) {
    return {std::make_shared<
        ZeroRemoteTensor>(context, _initStructs, element_type, shape, config, tensor_type, mem_type, mem)};
};

ov::SoPtr<ov::ITensor> ZeroDevice::createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape,
                                                    const Config& config) {
    return {std::make_shared<ZeroHostTensor>(context, _initStructs, element_type, shape, config)};
};
