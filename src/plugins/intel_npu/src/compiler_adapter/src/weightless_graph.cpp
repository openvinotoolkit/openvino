// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_graph.hpp"

#include "intel_npu/utils/zero/zero_host_tensor.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/make_tensor.hpp"

#define USE_SINGLE_THREADED_RUN_INIT 0

namespace {

std::vector<std::shared_ptr<ov::op::v0::Constant>> get_all_constants_in_topological_order(
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
        : _modelConstants(get_all_constants_in_topological_order(model)),
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
                                 const std::shared_ptr<ov::Model>& model,
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
      _initBlobPtrs(initBlobPtrs),
      _model(model) {}

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
    // The same operations are performed for the main schedule
    Graph::initialize(config);

    // Simplified version for init schedules
    const size_t numberOfInits = _initHandles.size();
    _initsInputDescriptors.reserve(numberOfInits);  // Can be removed after initialization?
    _initsOutputDescriptors.reserve(numberOfInits);
    _initsCommandQueues.reserve(numberOfInits);
    _initsCommandQueueOrdinals.reserve(numberOfInits);

    for (size_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
        _logger.debug("Graph initialize start, init schedule ", initIndex);

        ze_graph_handle_t initHandle = _initHandles.at(initIndex);
        std::unique_ptr<BlobContainer>& initBlobPtr = _initBlobPtrs.at(initIndex);
        std::vector<ArgumentDescriptor>& initInputDescriptors = _initsInputDescriptors.at(initIndex);
        std::vector<ArgumentDescriptor>& initOutputDescriptors = _initsOutputDescriptors.at(initIndex);
        std::shared_ptr<CommandQueue>& initCommandQueue = _initsCommandQueues.at(initIndex);
        uint32_t& initCommandQueueOrdinal = _initsCommandQueueOrdinals.at(initIndex);

        _logger.debug("performing pfnGetProperties");
        ze_graph_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        auto result = _zeroInitStruct->getGraphDdiTable().pfnGetProperties(initHandle, &props);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

        _logger.debug("performing pfnGetArgumentProperties3");
        for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
            ze_graph_argument_properties_3_t arg3{};
            arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            auto result = _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(initHandle, index, &arg3);
            THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetArgumentProperties3", result, _zeroInitStruct->getGraphDdiTable());

            if (arg3.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
                initInputDescriptors.push_back(ArgumentDescriptor{arg3, index});
            } else {
                initOutputDescriptors.push_back(ArgumentDescriptor{arg3, index});
            }
        }

        initInputDescriptors.shrink_to_fit();
        initOutputDescriptors.shrink_to_fit();

        initCommandQueueOrdinal = zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                                          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

        uint32_t command_queue_options = 0;
        if (config.has<TURBO>() && config.get<TURBO>()) {
            command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
        }

        // TODO do we want this for init schedules?
        if (config.has<RUN_INFERENCES_SEQUENTIALLY>() && config.get<RUN_INFERENCES_SEQUENTIALLY>()) {
            command_queue_options = command_queue_options | ZE_NPU_COMMAND_QUEUE_OPTION_DEVICE_SYNC;
        }

        initCommandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                          zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                          initCommandQueueOrdinal,
                                                          command_queue_options);

        if (config.has<WORKLOAD_TYPE>()) {
            set_workload_type(config.get<WORKLOAD_TYPE>(), initCommandQueue);
        }
        _zeGraphExt->initializeGraph(initHandle, initCommandQueueOrdinal);
        _logger.debug("Graph initialize finish, init schedule ", initIndex);

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        release_blob(config, initBlobPtr, initHandle);
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#if USE_SINGLE_THREADED_RUN_INIT
    for (const auto& initGraph : _initGraphs) {
        auto [weightsInputs, initOutputsTensor] = run_init_single_threaded();

        add_weights_inputs(weightsInputs);
        add_init_out_tensor(std::move(initOutputsTensor));
    }
#else
    std::tie(_weightsInputs, _initOutputsTensors) = run_init_multi_threaded();
#endif
    std::cout << "run_init() call "
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count()
              << "[ms]" << std::endl;
}

WeightlessGraph::InputData WeightlessGraph::allocate_inputs(
    const size_t initIndex,
    const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants) {
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

    for (const IODescriptor& descriptor : _initMetadata.at(initIndex).inputs) {
        initInputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initInputsTensor = {
        std::make_shared<ZeroHostTensor>(nullptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initInputsByteSize}),
                                         ov::intel_npu::TensorType::INPUT)};
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init inputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initMetadata.at(initIndex).inputs) {
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

WeightlessGraph::OutputData WeightlessGraph::allocate_outputs(const size_t initIndex) {
    std::vector<std::shared_ptr<ov::ITensor>> outputTensors;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> outputTensorsMap;

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    std::chrono::steady_clock::time_point begin_tensor_creation;
    std::chrono::steady_clock::time_point end_tensor_creation;

    begin = std::chrono::steady_clock::now();
    size_t initOutputsByteSize = 0;

    for (const IODescriptor& descriptor : _initMetadata.at(initIndex).outputs) {
        initOutputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    begin_tensor_creation = std::chrono::steady_clock::now();
    const ov::SoPtr<ov::ITensor> initOutputsTensor = {
        std::make_shared<ZeroHostTensor>(nullptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initOutputsByteSize}),
                                         ov::intel_npu::TensorType::BINDED)};
    end_tensor_creation = std::chrono::steady_clock::now();
    std::cout
        << "init outputs tensor creation "
        << std::chrono::duration_cast<std::chrono::microseconds>(end_tensor_creation - begin_tensor_creation).count()
        << "[microseconds]" << std::endl;

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initMetadata.at(initIndex).outputs) {
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

void WeightlessGraph::run_init_single_threaded() {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // TODO: this traverses full IR, so ideally only runs once for all inits
    const auto constants = get_all_constants_in_topological_order(_model);

    for (size_t initIndex = 0; initIndex < _initHandles.size(); ++initIndex) {
        auto [inputTensors, initInputsTensor] = allocate_inputs(initIndex, constants);

        auto [outputTensors, initOutputsTensor, outputTensorsMap] = allocate_outputs(initIndex);

        // Create zero-pipeline and run it (infer init schedule)
        begin = std::chrono::steady_clock::now();

        ze_device_properties_t properties = {};
        properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                    zeDeviceGetProperties(_initStructs->getDevice(), &properties));

        begin = std::chrono::steady_clock::now();
        // Pipeline pipeline(config, _initStructs, initGraph, inputTensors, outputTensors);
        create_pipeline();
        end = std::chrono::steady_clock::now();
        std::cout << "Creating the pipeline "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                  << std::endl;
        begin = std::chrono::steady_clock::now();
        // pipeline.push();
        // pipeline.pull();
        run_pipeline();
        end = std::chrono::steady_clock::now();
        std::cout << "Running the pipeline "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                  << std::endl;

        merge_two_maps(_weightsInputs, outputTensorsMap);
        _initOutputsTensors.push_back(std::move(initOutputsTensor));
    }
}

void WeightlessGraph::run_init_multi_threaded() {
    if (initGraphs.size() == 1) {
        std::cout << "::run_init_multi_threaded() for single init - fallback to ::runInit()" << std::endl;
        auto [map, tensor] = runInit(initGraphs.front(), model, context, config);
        return {map, {tensor}};
    }

    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> weightsInputs;
    std::vector<ov::SoPtr<ov::ITensor>> initTensors;

    // the pipeline:
    // allocate I/O -> create Pipeline -> run Pipeline
    //                                    allocate I/O -> create Pipeline -> run Pipeline
    Parallelizer multiThreadedRunner(
        _model,
        [&](const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants, int64_t graphIndex) -> QueueData {
            const auto& initGraph = initGraphs[graphIndex];

            QueueData data{};
            data.initGraphIndex = graphIndex;
            data.inputs = allocate_inputs(initGraph, constants, _config);
            data.outputs = allocate_outputs(initGraph, _config);
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
            merge_two_maps(_weightsInputs, data.outputs.tensorsMap);
            _initOutputsTensors.push_back(data.outputs.hostTensor);
        });

    multiThreadedRunner.callForAllAndWait(initGraphs);
}

void WeightlessGraph::create_pipeline(const size_t initIndex) {
    _logger.debug("Init Pipeline - initialize started");

    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }

    _initsCommandLists.at(initIndex) =
        std::make_unique<CommandList>(_zeroInitStruct, _initsCommandQueueOrdinals.at(initIndex));
    _initsFences.at(initIndex) = std::make_unique<Fence>(_initsCommandQueues.at(initIndex));

    size_t io_index = 0;
    for (const auto& desc : _initsInputDescriptors.at(initIndex)) {
        auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(input_tensors.at(io_index));
        void* data = remote_tensor->get_original_memory();

        _zeGraphExt->setGraphArgumentValue(_initHandles.at(initIndex), desc.idx, static_cast<unsigned char*>(data));

        ++io_index;
    }

    io_index = 0;
    for (const auto& desc : _initsOutputDescriptors.at(initIndex)) {
        auto remote_tensor = std::dynamic_pointer_cast<ZeroRemoteTensor>(output_tensors.at(io_index));
        void* data = remote_tensor->get_original_memory();

        _zeGraphExt->setGraphArgumentValue(_initHandles.at(initIndex), desc.idx, static_cast<unsigned char*>(data));
        ++io_index;
    }

    _initsCommandLists.at(initIndex)->appendGraphExecute(static_cast<ze_graph_handle_t>(_initHandles.at(initIndex)),
                                                         nullptr);

    _logger.debug("Init Pipeline - initialize completed");
}

void WeightlessGraph::run_pipeline(const size_t initIndex) {
    _logger.debug("Init Pipeline - push() started");
    _initsCommandLists.at(initIndex)->close();

    _initsCommandQueues.at(initIndex)->executeCommandList(*_initsCommandLists.at(initIndex),
                                                          *_initsFences.at(initIndex));
    _logger.debug("Init Pipeline - push() completed");

    _logger.debug("Init Pipeline - pull() started");
    _initsFences.at(initIndex)->hostSynchronize();
    _logger.debug("Init Pipeline - pull() completed");
}

void ZeroInferRequest::set_weights_inputs(
    const std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& weightsInputs) {
    // TODO can optimize this a little
    for (const auto& [weightName, weightTensor] : weightsInputs) {
        size_t inputIndex;
        for (inputIndex = 0; inputIndex < _metadata.inputs.size(); ++inputIndex) {
            const IODescriptor inputDescriptor = _metadata.inputs.at(inputIndex);
            if (inputDescriptor.isMainInputWeights && inputDescriptor.nameFromCompiler == weightName) {
                break;
            }
        }

        OPENVINO_ASSERT(inputIndex != _metadata.inputs.size(),
                        "Did not find an input bearing the ",
                        weightName,
                        " weights name");

        _userInputTensors.at(inputIndex) = {weightTensor};
        set_tensor_data(weightTensor, inputIndex, INPUT);
    }
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
