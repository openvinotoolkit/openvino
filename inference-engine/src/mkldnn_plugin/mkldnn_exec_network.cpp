// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <precision_utils.h>
#include <net_pass.h>
#include "mkldnn_exec_network.h"

#include "mkldnn_async_infer_request.h"
#include "mkldnn_infer_request.h"
#include "mkldnn_memory_state.h"
#include <ie_util_internal.hpp>
#include <graph_tools.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/eltwise_cpu.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "low_precision_transformations/transformer.hpp"

#include <algorithm>
#include <unordered_set>

using namespace MKLDNNPlugin;
using namespace MKLDNNPlugin::cpu;
using namespace InferenceEngine;
using InferenceEngine::details::CNNNetworkInt8Normalizer;
using namespace InferenceEngine::details;

InferenceEngine::InferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    if (graphs.size() > 1)  // streams uses special requests that are not connected to graphs
        return std::make_shared<MKLDNNGraphlessInferRequest>(networkInputs, networkOutputs);
    else
        return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs);
}

MKLDNNExecNetwork::MKLDNNExecNetwork(const InferenceEngine::ICNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr) : extensionManager(extMgr) {
    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = network.getStats(&pstats, nullptr);
    // we are cloning network if we have statistics and we can transform network.
    auto clonedNetwork = cloneNet(network);

    if (Precision::FP16 == network.getPrecision()) {
        clonedNetwork->setPrecision(Precision::FP32);
    }

    // CPU Plugin doesn't natively support some precision like int64/fp16/bool
    // so will convert all layer/tensors fp16->fp32 , bool->u8.
    // Default int64->int32 conversion is already applied in IE common module.
    NetPass::ConvertPrecision(*clonedNetwork, Precision::FP16, Precision::FP32);
    NetPass::ConvertPrecision(*clonedNetwork, Precision::BOOL, Precision::U8);

    if (s == StatusCode::OK && pstats && !pstats->isEmpty()) {
        CNNNetworkInt8Normalizer cnnorm;
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
    } else {
        if (cfg.lpTransformsMode == Config::LPTransformsMode::On) {
            auto params = LayerTransformation::Params(true,  // updatePrecisions
                                                      true,  // quantizeOutputs
                                                      true,  // weightsToConst
                                                      LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
                                                      LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
                                                      true,  // roundQuantizedValues
                                                      true,  // updateBiases
                                                      true);  // supportAsymmetricQuantization
            LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
                addBranchSpecific<EltwiseCpuTransformation>(LayerTransformation::Params(params), "Eltwise").
                add<ConvolutionTransformation>(LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }), "Convolution").
                addCleanup<ScaleShiftToConvolutionTransformation>(
                    LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
                    "ScaleShift"));
            transformer.transform(*clonedNetwork);
        }
    }

    MKLDNNGraph::ApplyUnrollPasses(static_cast<ICNNNetwork&>(*clonedNetwork));

    if (cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(*clonedNetwork)) {
            THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: such topology cannot be compiled for dynamic batch!";
        }
    }
    // general #threads logic
    const int env_threads = parallel_get_env_threads();
    const auto& numa_nodes = MKLDNNPlugin::cpu::getAvailableNUMANodes();
    const auto numa_nodes_num = numa_nodes.size();
    // use logical cores only for single-socket targets in throughput mode
    const int hw_cores = cfg.throughputStreams > 1 && numa_nodes_num == 1 ? parallel_get_max_threads() : getNumberOfCPUCores();

    const int threads = cfg.threadsNum ? cfg.threadsNum : (env_threads ? env_threads : hw_cores);
    const int threads_per_stream = std::max(1, threads/cfg.throughputStreams);

    // graph(s) initialization in taskExecutor threads (streams), in parallel (in case of streams)
    std::vector<Task> tasks;
    const int workers_per_socket = std::max(1,
            static_cast<int>(std::ceil(static_cast<float>(cfg.throughputStreams)/numa_nodes_num)));
    for (int n = 0; n < cfg.throughputStreams; n++) {
        MKLDNNGraph::Ptr _graph = std::make_shared<MKLDNNGraph>();
        graphs.push_back(_graph);
        tasks.push_back([=, &cfg, &clonedNetwork]() {
        _graph->setConfig(cfg);
         const int node = n / workers_per_socket;
         if (cfg.useThreadBinding)
            pin_current_thread_to_socket(numa_nodes[node]);
        _graph->CreateArenaWithObserverAndLoadGraph(threads_per_stream, numa_nodes[node], n,
                cfg.useThreadBinding,
                clonedNetwork, extensionManager);
        if (cfg.throughputStreams > 1)  // for streams, each worker thread has it's own graph
            MKLDNNPlugin::MultiWorkerTaskExecutor::ptrContext.ptrGraph = _graph;
        });
    }

    if (cfg.throughputStreams > 1) {
        // special executor with as many threads as requested #streams, each with it's own initialization task
        _taskExecutor = std::make_shared<MultiWorkerTaskExecutor>(tasks);
    } else {
        if (cfg.exclusiveAsyncRequests) {
            // special case when all InferRequests are muxed into a single queue
            ExecutorManager *executorManager = ExecutorManager::getInstance();
            _taskExecutor = executorManager->getExecutor("CPU");
        }
        _taskExecutor->runAndWait(tasks);
    }

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    if (graphs.size() == 1) {
        for (auto &node : graphs[0]->GetNodes()) {
            if (node->getType() == MemoryInput) {
                auto state_store = node->getChildEdgeAt(0)->getMemoryPtr();
                auto state_name = node->getName();

                // Remove suffix with pair ID. Internal information.
                auto suffix_idx = state_name.find("/id=");
                if (suffix_idx != std::string::npos)
                    state_name = state_name.substr(0, suffix_idx);

                memoryStates.emplace_back(new MKLDNNMemoryState(state_name, state_store));
            }
        }
    }
}

void MKLDNNExecNetwork::setProperty(const std::map<std::string, std::string> &properties) {
    for (auto g : graphs)
        g->setProperty(properties);
}

void MKLDNNExecNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncRequestImpl = std::make_shared<MKLDNNAsyncInferRequest>(syncRequestImpl, _taskExecutor, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MKLDNNAsyncInferRequest>(asyncRequestImpl),
                       [](IInferRequest *p) { p->Release(); });

    asyncRequestImpl->SetPointerToPublicInterface(asyncRequest);

    if (graphs.size() == 1) {  // single-stream (legacy/hetero) case - single graph for all requests
        auto mkldnnSyncRequest = dynamic_cast<MKLDNNInferRequest *>(syncRequestImpl.get());
        if (!mkldnnSyncRequest)
            THROW_IE_EXCEPTION << " Cannot get mkldnn sync request.";
        mkldnnSyncRequest->SetGraph(graphs[0]);
    }
}

void MKLDNNExecNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    graphPtr = graphs[0]->dump();
}

void MKLDNNExecNetwork::GetConfig(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    Config engConfig = graphs[0]->getProperty();
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

void MKLDNNExecNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        result = IE_SET_METRIC(NETWORK_NAME, graphs[0]->dump()->getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && key : graphs[0]->getProperty()._config) {
            configKeys.push_back(key.first);
        }
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = graphs[0]->getProperty();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        IE_ASSERT(option != engConfig._config.end());
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(std::stoi(option->second)));
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
    }
}

bool MKLDNNExecNetwork::CanProcessDynBatch(const InferenceEngine::ICNNNetwork &network) const {
    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    CNNLayerSet inputLayers;
    std::unordered_set<CNNLayer *> allLayers;

    if (inputs.empty())
        return false;

    auto & secondLayers = inputs.begin()->second->getInputData()->getInputTo();
    if (secondLayers.empty())
        return false;

    bool check_result = true;
    details::UnorderedDFS(allLayers, secondLayers.begin()->second, [&](CNNLayerPtr layer) {
        auto type = TypeFromName(layer->type);
        // This is WA for Tile layer
        auto tileLayer = dynamic_cast<TileLayer *>(layer.get());
        if (tileLayer && tileLayer->axis)
            return;

        if (type != Input &&
            type != Output &&
            type != Convolution &&
            type != Deconvolution &&
            type != Activation &&
            type != Depthwise &&
            type != Lrn &&
            type != Pooling &&
            type != FullyConnected &&
            type != Gemm &&
            type != SoftMax &&
            type != Split &&
            type != Concatenation &&
            type != Power &&
            type != Eltwise &&
            type != Crop &&
            type != BatchNormalization &&
            type != Copy) {
            check_result = false;
        }
    }, false);

    return check_result;
}

std::vector<IMemoryStateInternal::Ptr> MKLDNNExecNetwork::QueryState() {
    return memoryStates;
}
