// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <precision_utils.h>
#include "mkldnn_exec_network.h"

#include "mkldnn_async_infer_request.h"
#include "mkldnn_infer_request.h"
#include "mkldnn_memory_state.h"
#include "mkldnn_itt.h"
#include "nodes/mkldnn_memory_node.hpp"
#include <threading/ie_executor_manager.hpp>

#include <threading/ie_cpu_streams_executor.hpp>
#include <ie_system_conf.h>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <cstring>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/utils/utils.hpp>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

InferenceEngine::IInferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<MKLDNNExecNetwork>(shared_from_this()));
}

MKLDNNExecNetwork::MKLDNNExecNetwork(const InferenceEngine::CNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr,
                                     NumaNodesWeights &numaNodesWeights) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    extensionManager(extMgr),
    _cfg{cfg},
    _name{network.getName()},
    _numaNodesWeights(numaNodesWeights),
        _network(network) {
    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "CPU plug-in doesn't support not ngraph-based model!";
    }
    bool isFloatModel = !ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(function);

    if (_cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(_network)) {
            IE_THROW() << "MKLDNNGraph::CreateGraph: such topology cannot be compiled for dynamic batch!";
        }
    }

    if (cfg.exclusiveAsyncRequests) {
        // special case when all InferRequests are muxed into a single queue
        _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getExecutor("CPU");
    } else {
        auto streamsExecutorConfig = InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(_cfg.streamExecutorConfig, isFloatModel);
        streamsExecutorConfig._name = "CPUStreamsExecutor";
        _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    }
    if (0 != cfg.streamExecutorConfig._streams) {
        _callbackExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
            IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0, IStreamsExecutor::ThreadBindingType::NONE});
    } else {
        _callbackExecutor = _taskExecutor;
    }

    // Workaround for initializing friendly names for all the OPs
    // Otherwise they are initialized concurrently without thread safety.
    // TODO: Can be removed after 57069 is done.
    for (const auto& op : _network.getFunction()->get_ops()) {
        op->get_friendly_name();
    }

    int streams = std::max(1, _cfg.streamExecutorConfig._streams);
    std::vector<Task> tasks; tasks.resize(streams);
    _graphs.resize(streams);
    if (_cfg.streamExecutorConfig._streams != 0) {
        for (auto&& task : tasks) {
            task = [this] {
                MKLDNNExecNetwork::GetGraph();
            };
        }
        _taskExecutor->runAndWait(tasks);
    } else {
        MKLDNNExecNetwork::GetGraph();
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    if (_graphs.size() == 1) {
        for (auto &node : GetGraph()._graph.GetNodes()) {
            if (node->getType() == MemoryInput) {
                IE_THROW() << "[DS] Unimplemented";
//                auto memoryNode = dynamic_cast<MKLDNNMemoryInputNode*>(node.get());
//                auto state_store = memoryNode->getStore();
//                auto state_name = memoryNode->getId();
//
//                // Remove suffix with pair ID. Internal information.
//                auto suffix_idx = state_name.find("/id=");
//                if (suffix_idx != std::string::npos)
//                    state_name = state_name.substr(0, suffix_idx);
//
//                memoryStates.emplace_back(new MKLDNNVariableState(state_name, state_store));
            }
        }
    }
}

MKLDNNExecNetwork::Graph::Lock MKLDNNExecNetwork::GetGraph() {
    int streamId = 0;
    int numaNodeId = 0;
    auto streamsExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(_taskExecutor.get());
    if (nullptr != streamsExecutor) {
        streamId = streamsExecutor->GetStreamId();
        numaNodeId = streamsExecutor->GetNumaNodeId();
    }
    auto graphLock = Graph::Lock(_graphs[streamId % _graphs.size()]);
    if (!graphLock._graph.IsReady()) {
        std::exception_ptr exception;
        auto makeGraph = [&] {
            try {
                {
                    std::lock_guard<std::mutex> lock{_cfgMutex};
                    graphLock._graph.setConfig(_cfg);
                }
                graphLock._graph.CreateGraph(_network, extensionManager, _numaNodesWeights[numaNodeId]);
            } catch(...) {
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

void MKLDNNExecNetwork::setProperty(const std::map<std::string, std::string> &properties) {
    {
        std::lock_guard<std::mutex> lock{_cfgMutex};
        _cfg.readProperties(properties);
    }
    for (auto& g : _graphs) {
        auto graphLock = Graph::Lock(g);
        if (graphLock._graph.IsReady()) {
            graphLock._graph.setProperty(properties);
        }
    }
}

InferenceEngine::IInferRequestInternal::Ptr MKLDNNExecNetwork::CreateInferRequest() {
    return CreateAsyncInferRequestFromSync<MKLDNNAsyncInferRequest>();
}

InferenceEngine::CNNNetwork MKLDNNExecNetwork::GetExecGraphInfo() {
    if (_graphs.size() == 0)
        IE_THROW() << "No graph was found";

    return GetGraph()._graph.dump();
}

Parameter MKLDNNExecNetwork::GetConfig(const std::string &name) const {
    if (_graphs.size() == 0)
        IE_THROW() << "No graph was found";
    Config engConfig = const_cast<MKLDNNExecNetwork*>(this)->GetGraph()._graph.getProperty();
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        return option->second;
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter MKLDNNExecNetwork::GetMetric(const std::string &name) const {
    if (_graphs.size() == 0)
        IE_THROW() << "No graph was found";

    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME,
                               const_cast<MKLDNNExecNetwork*>(this)->GetGraph()._graph.dump().getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && key : const_cast<MKLDNNExecNetwork*>(this)->GetGraph()._graph.getProperty()._config) {
            configKeys.push_back(key.first);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = const_cast<MKLDNNExecNetwork*>(this)->GetGraph()._graph.getProperty();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        IE_ASSERT(option != engConfig._config.end());
        auto streams = std::stoi(option->second);
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(
            streams ? streams : 1));
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

bool MKLDNNExecNetwork::CanProcessDynBatch(const InferenceEngine::CNNNetwork &network) const {
    InputsDataMap inputs = network.getInputsInfo();

    if (inputs.empty())
        return false;

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "CPU plug-in doesn't support not ngraph-based model!";
    }

    auto ops = function->get_ordered_ops();
    for (auto op : ops) {
        auto type = TypeFromName(op->get_type_name());
        if (type == Tile) {
            const auto tile = std::dynamic_pointer_cast<const ngraph::opset1::Tile>(op);
            const auto repeatsNode = std::dynamic_pointer_cast<const ngraph::opset1::Constant>(tile->get_input_node_shared_ptr(1));
            if (!repeatsNode)
                return false;
            if (tile && repeatsNode->cast_vector<int64_t>()[0] == 1)
                continue;
        }

        if (type == Reshape) {
            if (op->get_input_shape(0)[0] == op->get_output_shape(0)[0])
                continue;
        }

        if (type != Input &&
            type != Output &&
            type != Convolution &&
            type != Deconvolution &&
            type != Lrn &&
            type != Pooling &&
            type != FullyConnected &&
            type != MatMul &&
            type != Softmax &&
            type != Split &&
            type != Concatenation &&
                type != Eltwise) {
            return false;
        }
    }

    return true;
}

IE_SUPPRESS_DEPRECATED_START
std::vector<IVariableStateInternal::Ptr> MKLDNNExecNetwork::QueryState() {
    return memoryStates;
}
IE_SUPPRESS_DEPRECATED_END
