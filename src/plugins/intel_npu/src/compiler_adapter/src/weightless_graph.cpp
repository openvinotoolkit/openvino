// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_graph.hpp"

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
    WeightlessGraph::InputData inputs;
    WeightlessGraph::OutputData outputs;

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

namespace intel_npu {

WeightlessGraph::WeightlessGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                                 const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                 ze_graph_handle_t mainGraphHandle,
                                 NetworkMetadata mainMetadata,
                                 std::unique_ptr<BlobContainer> mainBlobPtr,
                                 const std::vector<ze_graph_handle_t>& initGraphHandles,
                                 const std::vector<NetworkMetadata>& initMetadata,
                                 const std::vector<std::unique_ptr<BlobContainer>>& initBlobPtrs,
                                 const Config& config,
                                 const ov::SoPtr<ICompiler>& compiler = {nullptr})
    : Graph(zeGraphExt,
            zeroInitStruct,
            mainGraphHandle,
            std::move(mainMetadata),
            std::move(mainBlobPtr),
            config,
            compiler),
      _initHandles(initGraphHandles),
      _initMetadata(initMetadata),
      _initBlobPtrs(initBlobPtrs) {}

std::pair<uint64_t, std::vector<uint64_t>> WeightlessGraph::export_blob(std::ostream& stream) const {
    if (_blobIsReleased) {
        OPENVINO_THROW("Model was optimized away. Try importing it using `ov::hint::compiled_blob` property to extend "
                       "its lifetime.");
    }

    const auto writeToStream = [&](ze_graph_handle_t handle,
                                   const std::unique_ptr<BlobContainer>& blobUniquePtr) -> uint64_t {
        uint64_t blobSize;
        const uint8_t* blobRawPtr = nullptr;
        std::vector<uint8_t> blob;

        if (blobUniquePtr == nullptr) {
            // when compiling the model using Compiler in Driver, the blob is handled by the driver
            _zeGraphExt->getGraphBinary(handle, blob, blobRawPtr, blobSize);
        } else {
            // in all other cases, the blob is handled by the plugin
            blobRawPtr = static_cast<const uint8_t*>(blobUniquePtr->get_ptr());
            blobSize = blobUniquePtr->size();
        }

        stream.write(reinterpret_cast<const char*>(blobRawPtr), blobSize);

        if (!stream) {
            _logger.error("Write blob to stream failed. Blob is broken!");
            return 0;
        }

        return blobSize;
    };

    // By convention, first write the main part
    uint64_t mainBlobSize = writeToStream(_handle, _blobPtr);
    uint64_t totalBlobSize = mainBlobSize;

    // Then the init schedules
    std::vector<uint64_t> initSizes;
    for (size_t initIndex = 0; initIndex < _initHandles.size(); ++initIndex) {
        uint64_t initBlobSize = writeToStream(_initHandles.at(initIndex), _initBlobPtrs.at(initIndex));
        totalBlobSize += initBlobSize;
        initSizes.push_back(initBlobSize);
    }

    if (_logger.level() >= ov::log::Level::INFO) {
        std::uint32_t result = 1171117u;
        const uint8_t* blobRawPtr = static_cast<const uint8_t*>(_blobPtr->get_ptr());
        for (const uint8_t* it = blobRawPtr; it != blobRawPtr + mainBlobSize; ++it) {
            result = ((result << 7) + result) + static_cast<uint32_t>(*it);
        }

        std::stringstream str;
        str << "Main blob size: " << mainBlobSize << ", hash: " << std::hex << result;
        _logger.info(str.str().c_str());
    }
    _logger.info("Write blob to stream successfully.");
    return std::make_pair(totalBlobSize, initSizes);
}

void WeightlessGraph::initialize(const Config& config) {
    _logger.debug("Graph initialize start");

    if (_zeGraphExt == nullptr || _handle == nullptr) {
        return;
    }

    _logger.debug("performing pfnGetProperties");
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(_handle, &props);
    THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

    _logger.debug("performing pfnGetArgumentProperties3");
    for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
        ze_graph_argument_properties_3_t arg3{};
        arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(_handle, index, &arg3);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

        if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _input_descriptors.push_back(ArgumentDescriptor{arg3, index});
        } else {
            _output_descriptors.push_back(ArgumentDescriptor{arg3, index});
        }
    }

    _input_descriptors.shrink_to_fit();
    _output_descriptors.shrink_to_fit();

    _command_queue_group_ordinal =
        zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    uint32_t command_queue_options = 0;

    if (config.has<TURBO>() && config.get<TURBO>()) {
        if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 0)) {
            OPENVINO_THROW("Turbo is not supported by the current driver");
        }
        command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
    }

    if (_zeroInitStruct->getCommandQueueDdiTable().version() >= ZE_MAKE_VERSION(1, 1) &&
        config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
    }

    _command_queue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                    zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                    _command_queue_group_ordinal,
                                                    command_queue_options);

    if (config.has<WORKLOAD_TYPE>()) {
        set_workload_type(config.get<WORKLOAD_TYPE>());
    }

    _zeGraphExt->initializeGraph(_handle, _command_queue_group_ordinal);

    _logger.debug("Graph initialize finish");

    //  We are allowed to release the original blob because weights were loaded in NPU memory during
    //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
    //  releasing it here to avoid unnecessary memory usage.
    _blobIsReleased = release_blob(config);

    _batch_size = get_batch_size(_metadata);

    if (_zeroInitStruct->getCommandQueueDdiTable().version() < ZE_MAKE_VERSION(1, 1) &&
        config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
        auto number_of_command_lists = _batch_size.has_value() ? *_batch_size : 1;

        _last_submitted_event.resize(number_of_command_lists);
    }
}

WeightlessGraph::InputData WeightlessGraph::allocateInputs(
    const std::shared_ptr<IGraph>& initGraph,
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
    const ov::SoPtr<ov::ITensor> initInputsTensor = {
        std::make_shared<ZeroHostTensor>(context._ptr,
                                         _initStructs,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initInputsByteSize}),
                                         ov::intel_npu::TensorType::INPUT)};
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

WeightlessGraph::OutputData WeightlessGraph::allocateOutputs(const std::shared_ptr<IGraph>& initGraph,
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
    const ov::SoPtr<ov::ITensor> initOutputsTensor = {
        std::make_shared<ZeroHostTensor>(context._ptr,
                                         _initStructs,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initOutputsByteSize}),
                                         ov::intel_npu::TensorType::BINDED)};
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

std::pair<std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>, ov::SoPtr<ov::ITensor>>
WeightlessGraph::runInit(const std::shared_ptr<IGraph>& initGraph,
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
WeightlessGraph::runInitMultiThreaded(const std::vector<std::shared_ptr<IGraph>>& initGraphs,
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

WeightlessGraph::~WeightlessGraph() {
    // TODO do the same for init schedules, but in the initialize function. then remove this.
    if (_handle != nullptr) {
        auto result = _zeGraphExt->destroyGraph(_handle);

        if (ZE_RESULT_SUCCESS == result) {
            _handle = nullptr;
        }
    }

    if (!_last_submitted_event.empty()) {
        _last_submitted_event.clear();
    }

    if (_command_queue != nullptr) {
        _command_queue.reset();
    }
}

}  // namespace intel_npu
