// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "zero_device.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>

#include "intel_npu/common/itt.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/zero/zero_api.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "zero_host_tensor.hpp"
#include "zero_infer_request.hpp"
#include "zero_remote_tensor.hpp"

using namespace intel_npu;

namespace {
std::vector<std::shared_ptr<ov::op::v0::Constant>> getAllConstantsInTopologicalOrder(
    const std::shared_ptr<const ov::Model>& model) {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;

    // Match the inputs of the "init" model with the Constant nodes of the original model
    begin = std::chrono::steady_clock::now();
    for (auto&& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }
        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        constants.push_back(constantNode);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "getting constant IDs " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;

    return constants;
}

struct QueueData {
    int64_t initGraphIndex = -1;
    ZeroDevice::InputData inputs;
    ZeroDevice::OutputData outputs;

    bool isTerminator() const {
        return initGraphIndex == -1;
    }
};

// very simple multi-threaded executor for 2 kinds of tasks where task 1 "calls"
// task 2
template <typename Task1Callable, typename Task2Callable>
class Parallelizer {
    std::vector<std::shared_ptr<ov::op::v0::Constant>> _modelConstants;

    std::mutex _mutex;
    std::queue<QueueData> _payloads;
    std::atomic_bool _progressTask1 = true;

    Task1Callable _task1;
    Task2Callable _task2;

public:
    Parallelizer(const std::shared_ptr<const ov::Model>& model, Task1Callable&& task1, Task2Callable&& task2)
        : _modelConstants(getAllConstantsInTopologicalOrder(model)),
          _task1(std::forward<Task1Callable>(task1)),
          _task2(std::forward<Task2Callable>(task2)) {}

    void callForAllAndWait(const std::vector<std::shared_ptr<IGraph>>& initGraphs) {
        std::condition_variable task1SyncPoint;
        std::condition_variable task2SyncPoint;

        std::thread task2Thread([&]() {
            while (true) {
                std::unique_lock lock(_mutex);
                task2SyncPoint.wait(lock, [&]() {
                    return !_payloads.empty();
                });

                auto payload = std::move(_payloads.front());
                _payloads.pop();
                if (payload.isTerminator()) {
                    return;  // Note: exit condition
                }
                lock.unlock();

                _task2(std::move(payload), task1SyncPoint, _progressTask1);  // TODO: putting sync point inside is meh
            }
        });

        std::mutex task1Mutex;
        for (int64_t i = 0; i < int64_t(initGraphs.size()); ++i) {
            {
                // TODO: task1Mutex is *only* used here. this is a poor man's
                // spinlock on an atomic boolean (without busy wait)
                std::unique_lock lock(task1Mutex);
                task1SyncPoint.wait(lock, [&]() {
                    return _progressTask1.load();
                });
            }

            // TODO: should we run task1 under mutex?
            auto payload = _task1(_modelConstants, i);
            _progressTask1.store(false);
            {
                std::lock_guard guard(_mutex);
                _payloads.push(std::move(payload));
            }
            task2SyncPoint.notify_one();
        }
        {
            std::lock_guard guard(_mutex);
            _payloads.push({});  // isTerminator()
        }
        task2SyncPoint.notify_one();

        task2Thread.join();
    }
};

// c++17 deduction guide
template <typename Task1Callable, typename Task2Callable>
Parallelizer(const std::shared_ptr<const ov::Model>&, Task1Callable&&, Task2Callable&&)
    -> Parallelizer<Task1Callable, Task2Callable>;

void merge_two_maps(std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& dst,
                    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& src) {
    dst.merge(src);
    OPENVINO_ASSERT(src.empty(), "Found weights inputs collision between different inits");
}
}  // namespace

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
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // TODO: this traverses full IR, so ideally only runs once for all inits
    const auto constants = getAllConstantsInTopologicalOrder(model);

    auto [inputTensors, initInputsTensor] = allocateInputs(initGraph, constants, context, config);

    auto [outputTensors, initOutputsTensor, outputTensorsMap] = allocateOutputs(initGraph, context, config);

    // Create zero-pipeline and run it (infer init schedule)
    begin = std::chrono::steady_clock::now();

    ze_device_properties_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties", zeDeviceGetProperties(_initStructs->getDevice(), &properties));

    begin = std::chrono::steady_clock::now();
    Pipeline pipeline(config, _initStructs, initGraph, inputTensors, outputTensors);
    end = std::chrono::steady_clock::now();
    std::cout << "Creating the pipeline " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;
    begin = std::chrono::steady_clock::now();
    pipeline.push();
    pipeline.pull();
    end = std::chrono::steady_clock::now();
    std::cout << "Running the pipeline " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;

    return {outputTensorsMap, initOutputsTensor};
}

std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, std::vector<ov::SoPtr<ov::ITensor>>>
ZeroDevice::runInitMultiThreaded(const std::vector<std::shared_ptr<IGraph>>& initGraphs,
                                 const std::shared_ptr<const ov::Model>& model,
                                 const ov::SoPtr<ov::IRemoteContext>& context,
                                 const Config& config) {
    if (initGraphs.size() == 1) {
        std::cout << "::runInitMultiThreaded() for single init - fallback to ::runInit()" << std::endl;
        auto [map, tensor] = runInit(initGraphs.front(), model, context, config);
        return {map, {tensor}};
    }

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> weightsInputs;
    std::vector<ov::SoPtr<ov::ITensor>> initTensors;

    // the pipeline:
    // allocate I/O -> create Pipeline -> run Pipeline
    //                                    allocate I/O -> create Pipeline -> run Pipeline
    Parallelizer multiThreadedRunner(
        model,
        [&](const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants, int64_t graphIndex) -> QueueData {
            const auto& initGraph = initGraphs[graphIndex];

            QueueData data{};
            data.initGraphIndex = graphIndex;
            data.inputs = allocateInputs(initGraph, constants, context, config);
            data.outputs = allocateOutputs(initGraph, context, config);
            return data;
        },
        [&](QueueData&& data, std::condition_variable& cv, std::atomic_bool& flag) {
            std::chrono::steady_clock::time_point begin;
            std::chrono::steady_clock::time_point end;

            const auto& initGraph = initGraphs[data.initGraphIndex];

            // Create zero-pipeline and run it (infer init schedule)
            begin = std::chrono::steady_clock::now();
            ze_device_properties_t properties = {};
            properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                        zeDeviceGetProperties(_initStructs->getDevice(), &properties));

            Pipeline pipeline(config, _initStructs, initGraph, data.inputs.tensors, data.outputs.tensors);
            end = std::chrono::steady_clock::now();
            std::cout << "Creating the pipeline "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                      << std::endl;

            // progress task 1:
            flag.store(true);
            cv.notify_one();

            begin = std::chrono::steady_clock::now();
            pipeline.push();
            pipeline.pull();
            end = std::chrono::steady_clock::now();
            std::cout << "Running the pipeline "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                      << std::endl;

            // TODO: pre-allocate those well in advance? (outside of this loop)
            merge_two_maps(weightsInputs, data.outputs.tensorsMap);
            initTensors.push_back(data.outputs.hostTensor);
        });

    multiThreadedRunner.callForAllAndWait(initGraphs);

    return {weightsInputs, initTensors};
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
    return {std::make_shared<ZeroRemoteTensor>(context,
                                               _initStructs,
                                               device_properties,
                                               element_type,
                                               shape,
                                               config,
                                               tensor_type,
                                               mem_type,
                                               mem)};
};

ov::SoPtr<ov::ITensor> ZeroDevice::createHostTensor(std::shared_ptr<ov::IRemoteContext> context,
                                                    const ov::element::Type& element_type,
                                                    const ov::Shape& shape,
                                                    const Config& config,
                                                    ov::intel_npu::TensorType tensor_type) {
    return {std::make_shared<ZeroHostTensor>(context,
                                             _initStructs,
                                             device_properties,
                                             element_type,
                                             shape,
                                             config,
                                             tensor_type)};
};

ZeroDevice::InputData ZeroDevice::allocateInputs(const std::shared_ptr<IGraph>& initGraph,
                                                 const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants,
                                                 const ov::SoPtr<ov::IRemoteContext>& context,
                                                 const Config& config) {
    std::vector<std::vector<std::shared_ptr<ov::ITensor>>> inputTensors;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_memcpy;
    std::chrono::steady_clock::time_point end_memcpy;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;
    long long memcpy_duration = 0;

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

        OPENVINO_ASSERT(id < constants.size(), "Mismatch between weights IDs and parsed inputs");
        const auto& constant = constants[id];
        OPENVINO_ASSERT(constant->get_byte_size() == currentInputSize,
                        "Byte size mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constant->get_element_type() == descriptor.precision,
                        "Precision mismatch for ",
                        descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constant->get_shape() == descriptor.shapeFromCompiler.to_shape(),
                        "Shape mismatch for ",
                        descriptor.nameFromCompiler);

        begin_memcpy = std::chrono::steady_clock::now();
        // TODO: should we copy the constant acknowledging strides? (if there
        // are strides, we risk copying bogus data here)
        std::memcpy(currentInputBufferLocation, constant->get_data_ptr(), currentInputSize);
        end_memcpy = std::chrono::steady_clock::now();
        memcpy_duration =
            memcpy_duration + std::chrono::duration_cast<std::chrono::microseconds>(end_memcpy - begin_memcpy).count();

        inputTensors.push_back(
            {ov::make_tensor(constant->get_element_type(), constant->get_shape(), currentInputBufferLocation)});
        offset += currentInputSize;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Creating input tensors " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << "[microseconds]" << std::endl;
    std::cout << "Memcpy duration " << memcpy_duration << "[microseconds]" << std::endl;

    return {inputTensors, initInputsTensor};
}

ZeroDevice::OutputData ZeroDevice::allocateOutputs(const std::shared_ptr<IGraph>& initGraph,
                                                   const ov::SoPtr<ov::IRemoteContext>& context,
                                                   const Config& config) {
    std::vector<std::shared_ptr<ov::ITensor>> outputTensors;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> outputTensorsMap;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;

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

    size_t offset = 0;
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

    return {outputTensors, initOutputsTensor, outputTensorsMap};
}
