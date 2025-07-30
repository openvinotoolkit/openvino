// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weightless_graph.hpp"

#include <condition_variable>
#include <iterator>
#include <mutex>
#include <queue>

#include "intel_npu/config/options.hpp"
#include "intel_npu/prefix.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/make_tensor.hpp"

#define USE_SINGLE_THREADED_RUN_INIT 0

namespace intel_npu {

namespace {

constexpr uint8_t MAIN_SCHEDULE_INDEX = 0;

std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> get_all_constants_in_topological_order(
    const std::shared_ptr<const ov::Model>& model,
    const Logger& logger) {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> constants;

    // Match the inputs of the "init" model with the Constant nodes of the original model
    for (auto&& node : model->get_ops()) {
        if (!ov::is_type<ov::op::v0::Constant>(node)) {
            continue;
        }

        auto constantNode = std::static_pointer_cast<ov::op::v0::Constant>(node);
        ov::RTMap& runtimeInfoMap = constantNode->get_rt_info();
        const auto& weightlessCacheAttrIt = runtimeInfoMap.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (weightlessCacheAttrIt != runtimeInfoMap.end()) {
            auto& weightlessCacheAttr = weightlessCacheAttrIt->second.as<ov::WeightlessCacheAttribute>();
            constants[weightlessCacheAttr.bin_offset] = constantNode;
        }
    }

    return constants;
}

struct QueueData {
    int64_t initIndex = -1;
    WeightlessGraph::InputData inputs;
    WeightlessGraph::OutputData outputs;

    bool isTerminator() const {
        return initIndex == -1;
    }
};

// very simple multi-threaded executor for 2 kinds of tasks where task 1 "calls"
// task 2
template <typename Task1Callable, typename Task2Callable>
class Parallelizer {
    std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>> _modelConstants;

    std::mutex _mutex;
    std::queue<QueueData> _payloads;
    std::atomic_bool _progressTask1 = true;

    Task1Callable _task1;
    Task2Callable _task2;

public:
    Parallelizer(const std::shared_ptr<const ov::Model>& model,
                 Task1Callable&& task1,
                 Task2Callable&& task2,
                 const Logger& logger)
        : _modelConstants(get_all_constants_in_topological_order(model, logger)),
          _task1(std::forward<Task1Callable>(task1)),
          _task2(std::forward<Task2Callable>(task2)) {}

    void callForAllAndWait(const int64_t numberOfInits) {
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

                _task2(std::move(payload), task1SyncPoint,
                       _progressTask1);  // TODO: putting sync point inside is meh
            }
        });

        std::mutex task1Mutex;
        for (int64_t i = 0; i < numberOfInits; ++i) {
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
        _modelConstants = {};
    }
};

// c++17 deduction guide
template <typename Task1Callable, typename Task2Callable>
Parallelizer(const std::shared_ptr<const ov::Model>&, Task1Callable&&, Task2Callable&&, const Logger&)
    -> Parallelizer<Task1Callable, Task2Callable>;

void merge_two_maps(std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& dst,
                    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>>& src) {
    dst.merge(src);
    OPENVINO_ASSERT(src.empty(), "Found weights inputs collision between different inits");
}

}  // namespace

WeightlessGraph::WeightlessGraph(const std::shared_ptr<ZeGraphExtWrappers>& zeGraphExt,
                                 const std::shared_ptr<ZeroInitStructsHolder>& zeroInitStruct,
                                 const GraphDescriptor& mainGraphDesc,
                                 NetworkMetadata mainMetadata,
                                 std::optional<ov::Tensor> mainBlob,
                                 const std::vector<GraphDescriptor>& initGraphDesc,
                                 std::vector<NetworkMetadata> initMetadata,
                                 std::optional<std::vector<ov::Tensor>> initBlobs,
                                 const std::shared_ptr<const ov::Model>& model,
                                 const Config& config,
                                 const bool blobIsPersistent,
                                 const ov::SoPtr<ICompiler>& compiler)
    : Graph(zeGraphExt,
            zeroInitStruct,
            mainGraphDesc,
            std::move(mainMetadata),
            std::move(mainBlob),
            config,
            blobIsPersistent,
            compiler,
            true),
      _initsGraphDesc(initGraphDesc),
      _initBlobs(std::move(initBlobs)),
      _initsMetadata(std::move(initMetadata)),
      _model(model) {
    if (!config.get<CREATE_EXECUTOR>() || config.get<DEFER_WEIGHTS_LOAD>()) {
        _logger.info("Graph initialize is deferred from the \"WeightlessGraph\" constructor");
        return;
    }

    initialize(config);
}

std::pair<uint64_t, std::optional<std::vector<uint64_t>>> WeightlessGraph::export_blob(std::ostream& stream) const {
    if (_blobIsReleased) {
        OPENVINO_THROW("Model was optimized away. Try importing it using `ov::hint::compiled_blob` property to extend "
                       "its lifetime.");
    }

    size_t blobIndex = 0;
    std::uint32_t totalResult = 1171117u;
    totalResult = ((totalResult << 7) + totalResult);

    const auto writeToStream = [&](GraphDescriptor _graphDesc,
                                   const std::optional<ov::Tensor>& blobTensor) -> uint64_t {
        uint64_t blobSize;
        const uint8_t* blobRawPtr = nullptr;
        std::vector<uint8_t> blob;

        if (blobTensor == std::nullopt) {
            // when compiling the model using Compiler in Driver, the blob is handled by the driver
            _zeGraphExt->getGraphBinary(_graphDesc, blob, blobRawPtr, blobSize);
        } else {
            // in all other cases, the blob is handled by the plugin
            blobRawPtr = static_cast<const uint8_t*>(blobTensor->data());
            blobSize = blobTensor->get_byte_size();
        }

        if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        stream.write(reinterpret_cast<const char*>(blobRawPtr), static_cast<std::streamsize>(blobSize));

        if (!stream) {
            _logger.error("Write blob to stream failed. Blob is broken!");
            return 0;
        }

        if (_logger.level() >= ov::log::Level::INFO) {
            std::uint32_t result = 1171117u;
            for (const uint8_t* it = blobRawPtr; it != blobRawPtr + blobSize; ++it) {
                result = ((result << 7) + result) + static_cast<uint32_t>(*it);
            }

            totalResult += result;

            std::stringstream str;
            if (blobIndex == MAIN_SCHEDULE_INDEX) {
                str << "Main blob size " << blobSize << ", hash " << std::hex << result;
            } else {
                str << "Init part " << blobIndex << " blob size " << blobSize << ", hash " << std::hex << result;
            }
            _logger.info(str.str().c_str());
        }

        size_t size = utils::align_size_to_standarg_page_size(blobSize);
        size_t paddingSize = size - blobSize;
        if (paddingSize > 0) {
            std::fill_n(std::ostream_iterator<char>(stream), paddingSize, 0);

            if (!stream) {
                _logger.error("Write padding to stream failed. Blob is broken!");
                return 0;
            }

            _logger.info("Blob size with padding: %ld", size);
        }

        return size;
    };

    // By convention, first write the main part
    uint64_t mainBlobSize = writeToStream(_graphDesc, _blob);
    uint64_t totalBlobSize = mainBlobSize;
    ++blobIndex;

    // Then the init schedules
    std::vector<uint64_t> initSizes;
    for (size_t initIndex = 0; initIndex < _initsGraphDesc.size(); ++initIndex) {
        uint64_t initBlobSize = writeToStream(_initsGraphDesc.at(initIndex)._handle,
                                              _initBlobs.has_value() && _initBlobs->at(initIndex)
                                                  ? std::make_optional(_initBlobs->at(initIndex))
                                                  : std::nullopt);
        totalBlobSize += initBlobSize;
        initSizes.push_back(initBlobSize);
        ++blobIndex;
    }

    std::stringstream str;
    str << "Blob size: " << totalBlobSize << ", hash: " << std::hex << totalResult;
    _logger.info(str.str().c_str());

    _logger.info("Write blob to stream successfully.");
    return std::make_pair(totalBlobSize, initSizes);
}

void WeightlessGraph::initialize(const Config& config) {
    // Simplified version for init schedules
    const size_t numberOfInits = _initsGraphDesc.size();
    _initsInputDescriptors.resize(numberOfInits);
    _initsOutputDescriptors.resize(numberOfInits);
    _initsCommandQueueOrdinals.resize(numberOfInits);
    _initsCommandLists.resize(numberOfInits);
    _initsFences.resize(numberOfInits);

    for (size_t initIndex = 0; initIndex < numberOfInits; ++initIndex) {
        _logger.debug("WeightlessGraph initialize start, init schedule ", initIndex);
        std::vector<ArgumentDescriptor>& initInputDescriptors = _initsInputDescriptors.at(initIndex);
        std::vector<ArgumentDescriptor>& initOutputDescriptors = _initsOutputDescriptors.at(initIndex);
        uint32_t& initCommandQueueOrdinal = _initsCommandQueueOrdinals.at(initIndex);

        // Code similar to "Graph::initialize"
        _logger.debug("performing pfnGetProperties");
        ze_graph_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
        auto result =
            _zeroInitStruct->getGraphDdiTable().pfnGetProperties(_initsGraphDesc.at(initIndex)._handle, &props);
        THROW_ON_FAIL_FOR_LEVELZERO_EXT("pfnGetProperties", result, _zeroInitStruct->getGraphDdiTable());

        _logger.debug("performing pfnGetArgumentProperties3");
        for (uint32_t index = 0; index < props.numGraphArgs; ++index) {
            ze_graph_argument_properties_3_t arg3{};
            arg3.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
            auto result =
                _zeroInitStruct->getGraphDdiTable().pfnGetArgumentProperties3(_initsGraphDesc.at(initIndex)._handle,
                                                                              index,
                                                                              &arg3);
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

        _zeGraphExt->initializeGraph(_initsGraphDesc.at(initIndex), initCommandQueueOrdinal);
        _logger.debug("WeightlessGraph initialize finish, init schedule ", initIndex);

        //  We are allowed to release the original blob because weights were loaded in NPU memory during
        //  _zeGraphExt->initializeGraph(). The driver will not access the original blob from this moment on, so we are
        //  releasing it here to avoid unnecessary memory usage.
        release_init_blob(initIndex);
    }

    // Create a single command queue for all weights initialization schedules
    _initsCommandQueueGroupOrdinal =
        zeroUtils::findCommandQueueGroupOrdinal(_zeroInitStruct->getDevice(),
                                                ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE);

    uint32_t commandQueueOptions = 0;
    if (config.has<TURBO>() && config.get<TURBO>()) {
        _logger.debug("Set ZE_NPU_COMMAND_QUEUE_OPTION_TURBO in init command queue options");
        commandQueueOptions = commandQueueOptions | ZE_NPU_COMMAND_QUEUE_OPTION_TURBO;
    }

    _initsCommandQueue = std::make_shared<CommandQueue>(_zeroInitStruct,
                                                        zeroUtils::toZeQueuePriority(config.get<MODEL_PRIORITY>()),
                                                        _initsCommandQueueGroupOrdinal,
                                                        commandQueueOptions);

    if (config.has<WORKLOAD_TYPE>()) {
        switch (config.get<WORKLOAD_TYPE>()) {
        case ov::WorkloadType::DEFAULT:
            _initsCommandQueue->setWorkloadType(ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_DEFAULT);
            break;
        case ov::WorkloadType::EFFICIENT:
            _initsCommandQueue->setWorkloadType(ze_command_queue_workload_type_t::ZE_WORKLOAD_TYPE_BACKGROUND);
            break;
        default:
            OPENVINO_THROW("Unknown value for WorkloadType!");
        }
    }

#if USE_SINGLE_THREADED_RUN_INIT
    run_init_single_threaded();
#else
    run_init_multi_threaded();
#endif

    if (_initBlobs != std::nullopt) {  // Do not release the graph when compiling a model on the CiD path, and we don't
                                       // have a blob. We may need it to export later.
        release_graphs();
    }

    _initsInputDescriptors.clear();
    _initsOutputDescriptors.clear();
    _initsCommandQueueOrdinals.clear();
    _initsCommandLists.clear();
    _initsFences.clear();
    _initsMetadata.clear();
    _initsCommandQueue.reset();

    // The main schedule is initialized after the weights initialization ones in order to save some memory
    Graph::initialize(config);

    set_weights_inputs();
}

WeightlessGraph::InputData WeightlessGraph::allocate_inputs(
    const size_t initIndex,
    const std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>& constants) {
    std::vector<std::shared_ptr<ov::ITensor>> initInputsViewTensors;
    size_t initInputsByteSize = 0;

    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).inputs) {
        initInputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    // Due to the large number of init inputs, allocating a single buffer for all of them is more efficient. "View
    // tensors" are used for separating them.
    const ov::SoPtr<ZeroHostTensor> initInputsAllocatedTensor = {
        std::make_shared<ZeroHostTensor>(nullptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initInputsByteSize}),
                                         ov::intel_npu::TensorType::INPUT)};

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).inputs) {
        auto currentInputBufferLocation =
            static_cast<unsigned char*>(const_cast<void*>(initInputsAllocatedTensor->data(ov::element::Type_t::u8))) +
            offset;
        const size_t currentInputSize =
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));

        std::shared_ptr<ov::op::v0::Constant> constant;
        const size_t id = std::stoi(descriptor.nameFromCompiler);
        OPENVINO_ASSERT(constants.count(id) > 0,
                        "Weights ID ",
                        id,
                        " not found in the model constants. This may indicate a mismatch between the model and the "
                        "metadata of the compiled model.");

        constant = constants.at(id);

        std::memcpy(currentInputBufferLocation, constant->get_data_ptr(), currentInputSize);

        initInputsViewTensors.push_back(
            ov::make_tensor(constant->get_element_type(), constant->get_shape(), currentInputBufferLocation));
        offset += currentInputSize;
    }

    return {initInputsViewTensors, initInputsAllocatedTensor};
}

WeightlessGraph::OutputData WeightlessGraph::allocate_outputs(const size_t initIndex) {
    std::vector<std::shared_ptr<ov::ITensor>> initOutputsViewTensorsVector;
    std::unordered_map<std::string, std::shared_ptr<ov::ITensor>> initOutputsViewTensorsMap;
    size_t initOutputsByteSize = 0;

    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).outputs) {
        initOutputsByteSize +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    const ov::SoPtr<ZeroHostTensor> initOutputsAllocatedTensor = {
        std::make_shared<ZeroHostTensor>(nullptr,
                                         _zeroInitStruct,
                                         ov::element::Type_t::u8,
                                         ov::Shape({initOutputsByteSize}),
                                         ov::intel_npu::TensorType::BINDED)};

    size_t offset = 0;
    for (const IODescriptor& descriptor : _initsMetadata.at(initIndex).outputs) {
        const auto currentOutputBufferLocation =
            static_cast<unsigned char*>(const_cast<void*>(initOutputsAllocatedTensor->data(ov::element::Type_t::u8))) +
            offset;

        const ov::SoPtr<ov::ITensor> hostTensor =
            ov::make_tensor(descriptor.precision, descriptor.shapeFromCompiler.to_shape(), currentOutputBufferLocation);

        initOutputsViewTensorsVector.push_back(hostTensor._ptr);
        initOutputsViewTensorsMap.emplace(descriptor.nameFromCompiler, hostTensor._ptr);
        offset +=
            ov::element::get_memory_size(descriptor.precision, shape_size(descriptor.shapeFromCompiler.to_shape()));
    }

    return {initOutputsViewTensorsVector, initOutputsAllocatedTensor, initOutputsViewTensorsMap};
}

void WeightlessGraph::run_init_single_threaded() {
    auto constants = get_all_constants_in_topological_order(_model, _logger);

    for (size_t initIndex = 0; initIndex < _initsGraphDesc.size(); ++initIndex) {
        auto [initInputsViewTensors, initInputsAllocatedTensor] = allocate_inputs(initIndex, constants);

        // We don't need these anymore, potentially save some memory
        _model = nullptr;
        constants = {};
        auto [initOutputsViewTensors, initOutputsAllocatedTensor, initOutputsViewTensorsMap] =
            allocate_outputs(initIndex);

        // Create zero-pipeline and run it (infer init schedule)
        create_pipeline(initIndex, initInputsViewTensors, initOutputsViewTensors);
        run_pipeline(initIndex);

        merge_two_maps(_mainInputsViewTensors, initOutputsViewTensorsMap);
        _mainInputsAllocatedTensors.push_back(std::move(initOutputsAllocatedTensor));
    }
}

void WeightlessGraph::run_init_multi_threaded() {
    if (_initsGraphDesc.size() == 1) {
        _logger.info("::run_init_multi_threaded() for single init - fallback to ::runInit()");
        run_init_single_threaded();
        return;
    }

    std::unordered_map<std::string, std::shared_ptr<ZeroHostTensor>> weightsInputs;
    std::vector<ov::SoPtr<ZeroHostTensor>> initTensors;

    // the pipeline:
    // allocate I/O -> create Pipeline -> run Pipeline
    //                                    allocate I/O -> create Pipeline -> run Pipeline
    Parallelizer multiThreadedRunner(
        _model,
        [&](const std::unordered_map<size_t, std::shared_ptr<ov::op::v0::Constant>>& constants,
            int64_t initIndex) -> QueueData {
            QueueData data{};
            data.initIndex = initIndex;
            data.inputs = allocate_inputs(initIndex, constants);
            data.outputs = allocate_outputs(initIndex);
            return data;
        },
        [&](QueueData&& data, std::condition_variable& cv, std::atomic_bool& flag) {
            // Create zero-pipeline and run it (infer init schedule)
            ze_device_properties_t properties = {};
            properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            THROW_ON_FAIL_FOR_LEVELZERO("zeDeviceGetProperties",
                                        zeDeviceGetProperties(_zeroInitStruct->getDevice(), &properties));

            create_pipeline(data.initIndex, data.inputs.tensors, data.outputs.tensors);

            // progress task 1:
            flag.store(true);
            cv.notify_one();

            run_pipeline(data.initIndex);

            // TODO: pre-allocate those well in advance? (outside of this loop)
            merge_two_maps(_mainInputsViewTensors, data.outputs.tensorsMap);
            _mainInputsAllocatedTensors.push_back(data.outputs.hostTensor);
        },
        _logger);

    multiThreadedRunner.callForAllAndWait(_initsGraphDesc.size());
    _model = nullptr;
}

void WeightlessGraph::create_pipeline(const size_t initIndex,
                                      const std::vector<std::shared_ptr<ov::ITensor>>& inputTensors,
                                      const std::vector<std::shared_ptr<ov::ITensor>>& outputTensors) {
    _logger.debug("Init Pipeline - initialize started");

    if (_zeGraphExt == nullptr) {
        OPENVINO_THROW("Zero compiler adapter wasn't initialized");
    }

    _initsCommandLists.at(initIndex) =
        std::make_unique<CommandList>(_zeroInitStruct, _initsCommandQueueOrdinals.at(initIndex));
    _initsFences.at(initIndex) = std::make_unique<Fence>(_initsCommandQueue);

    size_t io_index = 0;
    for (const auto& desc : _initsInputDescriptors.at(initIndex)) {
        void* data = inputTensors.at(io_index++)->data();
        _zeGraphExt->setGraphArgumentValue(_initsGraphDesc.at(initIndex), desc.idx, static_cast<unsigned char*>(data));
    }

    io_index = 0;
    for (const auto& desc : _initsOutputDescriptors.at(initIndex)) {
        void* data = outputTensors.at(io_index++)->data();
        _zeGraphExt->setGraphArgumentValue(_initsGraphDesc.at(initIndex), desc.idx, static_cast<unsigned char*>(data));
    }

    _initsCommandLists.at(initIndex)->appendGraphExecute(
        static_cast<ze_graph_handle_t>(_initsGraphDesc.at(initIndex)._handle),
        nullptr);

    _logger.debug("Init Pipeline - initialize completed");
}

void WeightlessGraph::run_pipeline(const size_t initIndex) {
    _logger.debug("Init Pipeline - push() started");
    _initsCommandLists.at(initIndex)->close();

    _initsCommandQueue->executeCommandList(*_initsCommandLists.at(initIndex), *_initsFences.at(initIndex));
    _logger.debug("Init Pipeline - push() completed");

    _logger.debug("Init Pipeline - pull() started");
    _initsFences.at(initIndex)->hostSynchronize();
    _logger.debug("Init Pipeline - pull() completed");
}

void WeightlessGraph::set_weights_inputs() {
    for (const auto& desc : _input_descriptors) {
        if (!isMainInputWeightsName(desc.info.name)) {
            continue;
        }

        const std::string weightsInputName = std::string(desc.info.name).substr(MAIN_INPUT_WEIGHTS_PREFIX.size());
        OPENVINO_ASSERT(_mainInputsViewTensors.count(weightsInputName),
                        "Mismatch between main inputs and init outputs. The input of the main schedule \"",
                        weightsInputName,
                        "\" has no correspondent within the init outputs.");
        std::shared_ptr<ov::ITensor> weightsTensor = _mainInputsViewTensors.at(weightsInputName);
        set_argument_value(desc.idx, static_cast<unsigned char*>(weightsTensor->data()));
    }
}

void WeightlessGraph::release_init_blob(const size_t initIndex) {
    if (_initsGraphDesc[initIndex]._data || _blobIsPersistent || _initBlobs == std::nullopt) {
        return;
    }

    ze_graph_properties_2_t properties = {};
    properties.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    _zeroInitStruct->getGraphDdiTable().pfnGetProperties2(_initsGraphDesc.at(initIndex)._handle, &properties);

    if (~properties.initStageRequired & ZE_GRAPH_STAGE_INITIALIZE) {
        return;
    }

    _initBlobs->at(initIndex) = ov::Tensor();
    _logger.debug("Init blob is released");
}

void WeightlessGraph::release_graphs() {
    size_t initIndex = 0;
    if (_zeGraphExt != nullptr) {
        for (auto& initGraphDesc : _initsGraphDesc) {
            _zeGraphExt->destroyGraph(initGraphDesc);

            if (!_blobIsPersistent && _initBlobs != std::nullopt && _initBlobs->at(initIndex)) {
                _initBlobs->at(initIndex) = ov::Tensor();
            }

            initIndex++;
        }
    }
    _logger.debug("Init graphs are destroyed");
}

WeightlessGraph::~WeightlessGraph() {
    // make sure all the context-dependent components are destroyed before the zero context is destroyed
    release_graphs();
}

}  // namespace intel_npu
