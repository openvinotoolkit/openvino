// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <precision_utils.h>
#include "exec_network.h"
#include <low_precision/low_precision.hpp>

#include "async_infer_request.h"
#include "infer_request.h"
#include "memory_state.h"
#include "itt.h"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "serialize.h"
#include "ngraph/type/element_type.hpp"
#include "nodes/memory.hpp"
#include <threading/ie_executor_manager.hpp>
#define FIX_62820 0
#if FIX_62820 && ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
#include <threading/ie_tbb_streams_executor.hpp>
#endif
#include <threading/ie_cpu_streams_executor.hpp>
#include <ie_system_conf.h>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/utils/utils.hpp>
#include <ie_ngraph_utils.hpp>
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "utils/denormals.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <cstring>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {

InferenceEngine::IInferRequestInternal::Ptr
ExecNetwork::CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    if (!this->_plugin || !_plugin->IsNewAPI())
        return nullptr;
    return std::make_shared<InferRequest>(inputs, outputs, std::static_pointer_cast<ExecNetwork>(shared_from_this()));
}

InferenceEngine::IInferRequestInternal::Ptr
ExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                    InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<LegacyInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<ExecNetwork>(shared_from_this()));
}

struct ImmediateSerialExecutor : public ITaskExecutor {
    void run(InferenceEngine::Task task) override {
        std::lock_guard<std::mutex> l{_mutex};
        task();
    }
    std::mutex _mutex;
};

ExecNetwork::ExecNetwork(const InferenceEngine::CNNNetwork &network,
                         const Config &cfg,
                         const ExtensionManager::Ptr& extMgr,
                         const std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    extensionManager(extMgr),
    _network(network),
    _cfg{cfg},
    _name{network.getName()} {
    SetPointerToPlugin(plugin);
    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "CPU plug-in doesn't support not ngraph-based model!";
    }
    bool isFloatModel = !ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(function);

    _mutex = std::make_shared<std::mutex>();
    const auto& core = _plugin->GetCore();
    if (!core)
        IE_THROW() << "Unable to get API version. Core is unavailable";
    _cfg.isLegacyApi = !core->isNewAPI();


    if (cfg.exclusiveAsyncRequests) {
        // special case when all InferRequests are muxed into a single queue
        _taskExecutor = _plugin->executorManager()->getExecutor("CPU");
    } else {
        auto streamsExecutorConfig =
            is_cpu_map_available()
                ? _cfg.streamExecutorConfig
                : InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg.streamExecutorConfig,
                                                                                      isFloatModel);
        streamsExecutorConfig._name = "CPUStreamsExecutor";
        _cfg.streamExecutorConfig._threads = streamsExecutorConfig._threads;
#if FIX_62820 && (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        _taskExecutor = std::make_shared<TBBStreamsExecutor>(streamsExecutorConfig);
#else
        _taskExecutor = _plugin->executorManager()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
#endif
    }
    if (0 != cfg.streamExecutorConfig._streams) {
#if FIX_62820 && (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
        // There is no additional threads but we still need serialize callback execution to preserve legacy behaviour
        _callbackExecutor = std::make_shared<ImmediateSerialExecutor>();
#else
        _callbackExecutor = _plugin->executorManager()->getIdleCPUStreamsExecutor(
                                IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0, IStreamsExecutor::ThreadBindingType::NONE});
#endif
    } else {
        _callbackExecutor = _taskExecutor;
    }
    int streams = std::max(1, _cfg.streamExecutorConfig._streams);
    std::vector<Task> tasks; tasks.resize(streams);
    _graphs.resize(streams);
    if (_cfg.streamExecutorConfig._streams != 0) {
        auto all_graphs_ready = [&] {
            return std::all_of(_graphs.begin(), _graphs.end(), [&] (Graph& graph) {
                return graph.IsReady();
            });
        };
        do {
            for (auto&& task : tasks) {
                task = [this] {
                    if (!this->_cfg.changedDenormalsOptMode) {
                        set_denormals_optimization(this->_cfg);
                    }

                    ExecNetwork::GetGraph();
                };
            }
            _taskExecutor->runAndWait(tasks);
        } while (!all_graphs_ready());
    } else {
        ExecNetwork::GetGraph();
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    if (_graphs.size() == 1) {
        for (auto &node : GetGraph()._graph.GetNodes()) {
            if (node->getType() == Type::MemoryInput) {
                auto memoryNode = dynamic_cast<node::MemoryInput*>(node.get());
                if (!memoryNode) {
                    IE_THROW() << "Cannot cast " << node->getName() << " to MemoryInput";
                }
                auto state_store = memoryNode->getStore();
                auto state_name = memoryNode->getId();

                // Remove suffix with pair ID. Internal information.
                auto suffix_idx = state_name.find("/id=");
                if (suffix_idx != std::string::npos)
                    state_name = state_name.substr(0, suffix_idx);

                memoryStates.emplace_back(new VariableState(state_name, state_store));
            }
        }
    }
}

ExecNetwork::GraphGuard::Lock ExecNetwork::GetGraph() const {
    int streamId = 0;
    int socketId = 0;
    auto streamsExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(_taskExecutor.get());
    if (nullptr != streamsExecutor) {
        streamId = streamsExecutor->GetStreamId();
        socketId = streamsExecutor->GetSocketId();
    }
    auto graphLock = GraphGuard::Lock(_graphs[streamId % _graphs.size()]);
    if (!graphLock._graph.IsReady()) {
        std::exception_ptr exception;
        auto makeGraph = [&] {
            try {
                GraphContext::Ptr ctx;
                {
                    std::lock_guard<std::mutex> lock{*_mutex.get()};
                    // disable weights caching if graph was created only once
                    auto weightsCache = _cfg.streamExecutorConfig._streams != 1 ? _socketWeights[socketId] : nullptr;

                    auto isQuantizedFlag =
                        (_cfg.lpTransformsMode == Config::On) &&
                        ov::pass::low_precision::LowPrecision::isFunctionQuantized(_network.getFunction());

                    ctx = std::make_shared<GraphContext>(_cfg, extensionManager, weightsCache, isQuantizedFlag);
                }
                graphLock._graph.CreateGraph(_network, ctx);
            } catch (...) {
                exception = std::current_exception();
            }
        };
        if (nullptr != streamsExecutor) {
            streamsExecutor->Execute(makeGraph);
        } else {
            makeGraph();
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
    }
    return graphLock;
}

InferenceEngine::IInferRequestInternal::Ptr ExecNetwork::CreateInferRequest() {
    return CreateAsyncInferRequestFromSync<AsyncInferRequest>();
}

std::shared_ptr<ngraph::Function> ExecNetwork::GetExecGraphInfo() {
    if (_graphs.empty())
        IE_THROW() << "No graph was found";

    return GetGraph()._graph.dump();
}

Parameter ExecNetwork::GetConfigLegacy(const std::string &name) const {
    if (_graphs.empty())
        IE_THROW() << "No graph was found";
    /* legacy implementation return all the parameters which is actually not correct
     * since they are not reconfigurable. Fixed for new API */
    Config engConfig = GetGraph()._graph.getConfig();
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        return option->second;
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
}

/**
 * Only legacy parameters are supported.
 * No RW peroperties supported for new API.
 * All the RO properties are covered with GetMetric() method and
 * GetConfig() is not expected to be called by new API with params from new configuration API.
 */
Parameter ExecNetwork::GetConfig(const std::string &name) const {
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetConfigLegacy(name);
}

InferenceEngine::Parameter ExecNetwork::GetMetricLegacy(const std::string &name, const GraphGuard& graph) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, graph.dump()->get_friendly_name());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && key : graph.getConfig()._config) {
            configKeys.push_back(key.first);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = graph.getConfig();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        IE_ASSERT(option != engConfig._config.end());
        auto streams = std::stoi(option->second);
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(
            streams ? streams : 1));
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

InferenceEngine::Parameter ExecNetwork::GetMetric(const std::string &name) const {
    if (_graphs.empty())
        IE_THROW() << "No graph was found";
    // @todo Can't we just use local copy (_cfg) instead?
    auto graphLock = GetGraph();
    const auto& graph = graphLock._graph;
    const auto& config = graph.getConfig();

    if (_cfg.isLegacyApi) {
        return GetMetricLegacy(name, graph);
    }

    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };

    if (name == ov::supported_properties) {
        return std::vector<ov::PropertyName> {
            RO_property(ov::supported_properties.name()),
            RO_property(ov::model_name.name()),
            RO_property(ov::optimal_number_of_infer_requests.name()),
            RO_property(ov::num_streams.name()),
            RO_property(ov::affinity.name()),
            RO_property(ov::inference_num_threads.name()),
            RO_property(ov::enable_profiling.name()),
            RO_property(ov::hint::inference_precision.name()),
            RO_property(ov::hint::performance_mode.name()),
            RO_property(ov::hint::execution_mode.name()),
            RO_property(ov::hint::num_requests.name()),
            RO_property(ov::hint::enable_cpu_pinning.name()),
            RO_property(ov::hint::scheduling_core_type.name()),
            RO_property(ov::hint::enable_hyper_threading.name()),
            RO_property(ov::execution_devices.name()),
            RO_property(ov::intel_cpu::denormals_optimization.name()),
            RO_property(ov::intel_cpu::sparse_weights_decompression_rate.name()),
        };
    }

    if (name == ov::model_name) {
        // @todo Does not seem ok to 'dump()' the whole graph everytime in order to get a name
        const std::string modelName = graph.dump()->get_friendly_name();
        return decltype(ov::model_name)::value_type(modelName);
    } else if (name == ov::optimal_number_of_infer_requests) {
        const auto streams = config.streamExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(streams); // ov::optimal_number_of_infer_requests has no negative values
    } else if (name == ov::num_streams) {
        const auto streams = config.streamExecutorConfig._streams;
        return decltype(ov::num_streams)::value_type(streams); // ov::num_streams has special negative values (AUTO = -1, NUMA = -2)
    } else if (name == ov::affinity) {
        const auto affinity = config.streamExecutorConfig._threadBindingType;
        switch (affinity) {
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NONE:
            return ov::Affinity::NONE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::CORES:
            return ov::Affinity::CORE;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::NUMA:
            return ov::Affinity::NUMA;
        case InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE:
            return ov::Affinity::HYBRID_AWARE;
        }
        return ov::Affinity::NONE;
    } else if (name == ov::inference_num_threads) {
        const auto num_threads = config.streamExecutorConfig._threads;
        return decltype(ov::inference_num_threads)::value_type(num_threads);
    } else if (name == ov::enable_profiling.name()) {
        const bool perfCount = config.collectPerfCounters;
        return decltype(ov::enable_profiling)::value_type(perfCount);
    } else if (name == ov::hint::inference_precision) {
        return decltype(ov::hint::inference_precision)::value_type(config.inferencePrecision);
    } else if (name == ov::hint::performance_mode) {
        const auto perfHint = ov::util::from_string(config.perfHintsConfig.ovPerfHint, ov::hint::performance_mode);
        return perfHint;
    } else if (name == ov::hint::enable_cpu_pinning.name()) {
        const bool use_pin = config.enableCpuPinning;
        return decltype(ov::hint::enable_cpu_pinning)::value_type(use_pin);
    } else if (name == ov::hint::scheduling_core_type) {
        const auto core_type = config.schedulingCoreType;
        return core_type;
    } else if (name == ov::hint::enable_hyper_threading.name()) {
        const bool use_ht = config.enableHyperThreading;
        return decltype(ov::hint::enable_hyper_threading)::value_type(use_ht);
    } else if (name == ov::hint::execution_mode) {
        return config.executionMode;
    } else if (name == ov::hint::num_requests) {
        const auto perfHintNumRequests = config.perfHintsConfig.ovPerfHintNumRequests;
        return decltype(ov::hint::num_requests)::value_type(perfHintNumRequests);
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{_plugin->GetName()};
    } else if (name == ov::intel_cpu::denormals_optimization) {
        return decltype(ov::intel_cpu::denormals_optimization)::value_type(config.denormalsOptMode == Config::DenormalsOptMode::DO_On);
    } else if (name == ov::intel_cpu::sparse_weights_decompression_rate) {
        return decltype(ov::intel_cpu::sparse_weights_decompression_rate)::value_type(config.fcSparseWeiDecompressionRate);
    }
    /* Internally legacy parameters are used with new API as part of migration procedure.
     * This fallback can be removed as soon as migration completed */
    return GetMetricLegacy(name, graph);
}

void ExecNetwork::Export(std::ostream& modelStream) {
    CNNNetworkSerializer serializer(modelStream, extensionManager);
    serializer <<_network;
}

}   // namespace intel_cpu
}   // namespace ov
