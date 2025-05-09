// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <queue>
#include <string_view>

#include "async_infer_request.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"
#include "openvino/core/except.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "transformations/utils/utils.hpp"

#define USE_SINGLE_THREADED_RUN_INIT 0

namespace intel_npu {

namespace {

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

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
    InitInputData inputs;
    InitOutputData outputs;

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

using intel_npu::envVarStrToBool;

std::chrono::steady_clock::time_point begin;
std::chrono::steady_clock::time_point end;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             const std::shared_ptr<IDevice>& device,
                             const std::shared_ptr<IGraph>& graph,
                             const FilteredConfig& config,
                             const std::vector<std::shared_ptr<IGraph>>& initGraphs,
                             const std::shared_ptr<ov::Model>& initModel)
    : ICompiledModel(model, plugin),
      _config(config),
      _logger("CompiledModel", config.get<LOG_LEVEL>()),
      _device(device),
      _graph(graph),
      _initGraphs(initGraphs),
      _initModel(initModel) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::CompiledModel");

    OV_ITT_TASK_CHAIN(COMPILED_MODEL, itt::domains::NPUPlugin, "CompiledModel::CompiledModel", "initialize_properties");
    _properties = std::make_unique<Properties>(PropertiesType::COMPILED_MODEL, _config);
    _properties->registerProperties();

    configure_stream_executors();

    if (!_initGraphs.empty()) {
        if (_config.get<CREATE_EXECUTOR>() && !_config.get<DEFER_WEIGHTS_LOAD>()) {
            begin = std::chrono::steady_clock::now();
#if USE_SINGLE_THREADED_RUN_INIT
            runInitSingleThreaded();

#else
            runInitMultiThreaded();
#endif
            end = std::chrono::steady_clock::now();
            std::cout << "run_init() call within the \"CompiledModel\" ctor "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;
        }
    }

    OV_ITT_TASK_SKIP(COMPILED_MODEL);
}

CompiledModel::~CompiledModel() {
    _logger.debug("~CompiledModel()");
    std::dynamic_pointer_cast<ov::threading::IStreamsExecutor>(get_task_executor())->cpu_reset();
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "CompiledModel::create_infer_request");

    // sanity check
    if (_device == nullptr) {
        OPENVINO_THROW("No available devices. Failed to create infer request!");
    }

    if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
        if (_graph == nullptr) {
            OPENVINO_THROW("Invalid graph handle! Failed to create infer request!");
        }
        _graph->initialize(_config);
    }

    const std::shared_ptr<SyncInferRequest>& syncInferRequest =
        _device->createInferRequest(shared_from_this(), _config);
    syncInferRequest->initialize_states();

    if (!_initGraphs.empty()) {
        if (!_config.get<CREATE_EXECUTOR>() || _config.get<DEFER_WEIGHTS_LOAD>()) {
            begin = std::chrono::steady_clock::now();
            // TODO: in theory, initialize() could also be pipelined with runInit?
            for (const auto& initGraph : _initGraphs) {
                initGraph->initialize(_config);
            }
            end = std::chrono::steady_clock::now();
            std::cout << "Init graph(s) initialize() "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;

            begin = std::chrono::steady_clock::now();
#if USE_SINGLE_THREADED_RUN_INIT
            runInitSingleThreaded();
#else
            runInitMultiThreaded();
#endif
            end = std::chrono::steady_clock::now();
            std::cout << "run_init() call during inference request creation "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
                      << std::endl;
        }

        OPENVINO_ASSERT(_device != nullptr);

        begin = std::chrono::steady_clock::now();
        syncInferRequest->set_weights_inputs(_weightsInputs);
        end = std::chrono::steady_clock::now();
        std::cout << "set_weights_inputs() call "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    }

    return std::make_shared<AsyncInferRequest>(syncInferRequest,
                                               get_task_executor(),
                                               _resultExecutor,
                                               get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OPENVINO_THROW_NOT_IMPLEMENTED(
        "The synchronous inference request structure implemented by the NPU plugin does not inherit "
        "the \"ov::ISyncInferRequest\" class");
}

void CompiledModel::export_model(std::ostream& stream) const {
    _logger.debug("CompiledModel::export_model");

    size_t mainBlobSizeBeforeVersioning = _graph->export_blob(stream);
    size_t totalInitBlobSizesBeforeVersioning = 0;
    std::vector<uint64_t> initBlobSizes;

    for (const std::shared_ptr<IGraph>& initGraph : _initGraphs) {
        const uint64_t initBlobSize = initGraph->export_blob(stream);
        totalInitBlobSizesBeforeVersioning += initBlobSize;
        initBlobSizes.push_back(initBlobSize);
    }

    auto meta = Metadata<CURRENT_METADATA_VERSION>(totalInitBlobSizesBeforeVersioning + mainBlobSizeBeforeVersioning,
                                                   CURRENT_OPENVINO_VERSION,
                                                   initBlobSizes);
    meta.write(stream);
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (const IODescriptor& inputDescriptor : _graph->get_metadata().inputs) {
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, inputDescriptor.shapeFromCompiler);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (const IODescriptor& outputDescriptor : _graph->get_metadata().outputs) {
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy = std::make_shared<ov::op::v0::Constant>(
            outputDescriptor.precision,
            outputDescriptor.shapeFromCompiler.to_shape().empty() ? CONSTANT_NODE_DUMMY_SHAPE
                                                                  : outputDescriptor.shapeFromCompiler.to_shape());

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     outputDescriptor.shapeFromCompiler,
                                                     outputDescriptor.outputTensorNames);

        auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
        result->output(0).set_tensor_ptr(tensorDummy);
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    _logger.warning("Returning a dummy ov::Model object that contains only the given parameter and result nodes");

    return std::make_shared<ov::Model>(results, parameters);
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    // 1. Set the property via Properties interface
    _properties->set_property(properties);

    // 2. Extra hooks
    if (properties.count(std::string(WORKLOAD_TYPE::key())) != 0) {
        if (_graph != nullptr) {
            const auto workloadType = properties.at(ov::workload_type.name()).as<ov::WorkloadType>();
            _graph->set_workload_type(workloadType);
        }
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    // special cases
    if (name == ov::model_name.name()) {
        OPENVINO_ASSERT(_graph != nullptr, "Missing graph");
        return _graph->get_metadata().name;
    } else {
        // default behaviour
        return _properties->get_property(name);
    }
}

const std::shared_ptr<IGraph>& CompiledModel::get_graph() const {
    return _graph;
}

const FilteredConfig& CompiledModel::get_config() const {
    return _config;
}

void CompiledModel::configure_stream_executors() {
    std::shared_ptr<ov::threading::ITaskExecutor> task_executor;
    if (get_plugin()->get_property(ov::internal::exclusive_async_requests.name(), {}).as<bool>()) {
        task_executor = ov::threading::executor_manager()->get_executor("NPU");
    } else if (get_property(ov::hint::enable_cpu_pinning.name()).as<bool>()) {
        auto executor_config = ov::threading::IStreamsExecutor::Config{
            /* name = */ "Intel NPU plugin executor",
            /* streams = */ get_plugin()->get_property(ov::num_streams.name(), {}).as<ov::streams::Num>(),
            /* threads_per_stream = */ 1,
            /* thread_preferred_core_type = */ ov::hint::SchedulingCoreType::PCORE_ONLY,
            /* cpu_reservation = */ true};
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(executor_config);
    } else {
        task_executor = std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"NPUPlugin executor"});
    }

    set_task_executor(std::move(task_executor));
    const auto executorId = _graph->get_metadata().name + "_NPUResultExecutor";
    _resultExecutor = ov::threading::executor_manager()->get_executor(executorId);
}

void CompiledModel::runInitSingleThreaded() {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // TODO: this traverses full IR, so ideally only runs once for all inits
    const auto constants = getAllConstantsInTopologicalOrder(_initModel);

    for (const auto& initGraph : _initGraphs) {
        auto [inputTensors, initInputsTensor] = initGraph->allocateInputs(constants, get_context(), _config);

        auto [outputTensors, initOutputsTensors, weightsInputs] =
            initGraph->allocateOutputs(initGraph, get_context(), _config);

        _weightsInputs.merge(weightsInputs);
        OPENVINO_ASSERT(weightsInputs.empty(), "Found weights inputs collision between different inits");
        _initOutputsTensors.push_back(std::move(initOutputsTensors));

        // Create zero-pipeline and run it (infer init schedule)
        begin = std::chrono::steady_clock::now();
        initGraph->createPipeline(_config, inputTensors, outputTensors);
        end = std::chrono::steady_clock::now();
        std::cout << "Creating the pipeline "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                  << std::endl;
        begin = std::chrono::steady_clock::now();
        initGraph->runPipeline();
        end = std::chrono::steady_clock::now();
        std::cout << "Running the pipeline "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                  << std::endl;
    }
}

void CompiledModel::runInitMultiThreaded() {
    if (_initGraphs.size() == 1) {
        std::cout << "::runInitMultiThreaded() for single init - fallback to ::runInitSingleThreaded()" << std::endl;
        runInitSingleThreaded();
    }

    // the pipeline:
    // allocate I/O -> create Pipeline -> run Pipeline
    //                                    allocate I/O -> create Pipeline -> run Pipeline
    Parallelizer multiThreadedRunner(
        _initModel,
        [&](const std::vector<std::shared_ptr<ov::op::v0::Constant>>& constants, int64_t graphIndex) -> QueueData {
            const auto& initGraph = _initGraphs[graphIndex];

            QueueData data{};
            data.initGraphIndex = graphIndex;
            data.inputs = initGraph->allocateInputs(constants, get_context(), _config);
            data.outputs = initGraph->allocateOutputs(get_context(), _config);
            return data;
        },
        [&](QueueData&& data, std::condition_variable& cv, std::atomic_bool& flag) {
            std::chrono::steady_clock::time_point begin;
            std::chrono::steady_clock::time_point end;

            const auto& initGraph = _initGraphs[data.initGraphIndex];

            // Create zero-pipeline and run it (infer init schedule)
            begin = std::chrono::steady_clock::now();
            initGraph->createPipeline(config, data.inputs.tensors, data.outputs.tensors);
            end = std::chrono::steady_clock::now();
            std::cout << "Creating the pipeline "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[microseconds]"
                      << std::endl;

            // progress task 1:
            flag.store(true);
            cv.notify_one();

            begin = std::chrono::steady_clock::now();
            initGraph->runPipeline();
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

}  // namespace intel_npu
