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
#include "bf16transformer.h"
#include <ie_util_internal.hpp>
#include <graph_tools.hpp>
#include <threading/ie_executor_manager.hpp>
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "low_precision_transformations/transformer.hpp"
#include <threading/ie_cpu_streams_executor.hpp>
#include <ie_system_conf.h>
#include <threading/ie_thread_affinity.hpp>
#include <algorithm>
#include <unordered_set>
#include <utility>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

InferenceEngine::InferRequestInternal::Ptr
MKLDNNExecNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                          InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<MKLDNNInferRequest>(networkInputs, networkOutputs, std::static_pointer_cast<MKLDNNExecNetwork>(shared_from_this()));
}

MKLDNNExecNetwork::MKLDNNExecNetwork(const InferenceEngine::ICNNNetwork &network,
                                     const Config &cfg,
                                     const MKLDNNExtensionManager::Ptr& extMgr,
                                     NumaNodesWeights &numaNodesWeights) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{nullptr, nullptr},
    extensionManager(extMgr),
    _cfg{cfg},
    _name{network.getName()} {
    // we are cloning network if we have statistics and we can transform network.
    _clonedNetwork = cloneNet(network);

    IE_SUPPRESS_DEPRECATED_START
    if (Precision::FP16 == network.getPrecision()) {
        _clonedNetwork->setPrecision(Precision::FP32);
    }
    IE_SUPPRESS_DEPRECATED_END

    // CPU Plugin doesn't natively support some precision like int64/fp16/bool
    // so will convert all layer/tensors fp16->fp32 , bool->u8.
    // Default int64->int32 conversion is already applied in IE common module.
    NetPass::ConvertPrecision(*_clonedNetwork, Precision::I64, Precision::I32);
    NetPass::ConvertPrecision(*_clonedNetwork, Precision::U64, Precision::I32);
    NetPass::ConvertPrecision(*_clonedNetwork, Precision::FP16, Precision::FP32);
    NetPass::ConvertPrecision(*_clonedNetwork, Precision::BOOL, Precision::U8);
    NetPass::ConvertPrecision(*_clonedNetwork, Precision::U16, Precision::I32);

    if (_cfg.lpTransformsMode == Config::LPTransformsMode::On) {
        auto params = LayerTransformation::Params(true,  // updatePrecisions
                                                    true,  // quantizeOutputs
                                                    true,  // weightsToConst
                                                    LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
                                                    LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
                                                    true,  // roundQuantizedValues
                                                    true,  // updateBiases
                                                    true);  // supportAsymmetricQuantization
        LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params).
            add<ConvolutionTransformation>(LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }), "Convolution").
            addCleanup<ScaleShiftToConvolutionTransformation>(
                LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
                "ScaleShift"));
        transformer.transform(*_clonedNetwork);

        // Check if network is INT8 or Binary.
        // BF16 transformations were disabled since CPU plug-in doesn't support mixed precision execution:
        // BF16 + INT8 or BF16 + BIN.
        bool isFloatModel = true;
        CNNNetworkIterator i(&network);
        while (i != CNNNetworkIterator()) {
            if (CaselessEq<std::string>()((*i)->type, "FakeQuantize")) {
                isFloatModel = false;
                break;
            }
            i++;
        }

        if (with_cpu_x86_bfloat16() && isFloatModel) {
            BF16Transformer bf16Transformer;
            CNNNetwork cnnetwork(_clonedNetwork);
            // If enforceBF16 flag was set, BF16 transformation applies for all layers supported by CPU plugin.
            // Overwise, only layers marked as BF16 in 'cnnetwork' will be performed in bfloat16 mode.
            // CPU plugin throws an exception, if marked as BF16 layers have not supported by CPU plugin.
            if (cfg.enforceBF16 == true)
                bf16Transformer.convertToBFloat16(cnnetwork);
        } else {
            BF16Transformer bf16Transformer;
            CNNNetwork cnnetwork(_clonedNetwork);
            bf16Transformer.convertToFloat(cnnetwork);
        }
    }

    MKLDNNGraph::ApplyUnrollPasses(static_cast<ICNNNetwork&>(*_clonedNetwork));

    if (_cfg.batchLimit > 1) {
        // check topology for applicability
        if (!CanProcessDynBatch(*_clonedNetwork)) {
            THROW_IE_EXCEPTION << "MKLDNNGraph::CreateGraph: such topology cannot be compiled for dynamic batch!";
        }
    }

    if (cfg.exclusiveAsyncRequests) {
        // special case when all InferRequests are muxed into a single queue
        _taskExecutor = ExecutorManager::getInstance()->getExecutor("CPU");
    } else {
        const int env_threads = parallel_get_env_threads();
        const auto& numa_nodes = getAvailableNUMANodes();
        const auto numa_nodes_num = numa_nodes.size();
        auto streamExecutorConfig = cfg.streamExecutorConfig;
        // use logical cores only for single-socket targets in throughput mode
        const int hw_cores = streamExecutorConfig._streams > 1 && numa_nodes_num == 1 ? parallel_get_max_threads() : getNumberOfCPUCores();
        const int threads = streamExecutorConfig._threads ? streamExecutorConfig._threads : (env_threads ? env_threads : hw_cores);
        streamExecutorConfig._threadsPerStream = streamExecutorConfig._streams
                                                ? std::max(1, threads/streamExecutorConfig._streams)
                                                : threads;
        streamExecutorConfig._name = "CPUStreamsExecutor";
        _taskExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamExecutorConfig);
    }
    if (0 != cfg.streamExecutorConfig._streams) {
        _callbackExecutor = ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
            IStreamsExecutor::Config{"CPUCallbackExecutor", 1, 0, IStreamsExecutor::ThreadBindingType::NONE});
    } else {
        _callbackExecutor = _taskExecutor;
    }

    _graphs = decltype(_graphs){[&] {
        // TODO: Remove `cloneNet` to `localNetwork` when `MKLDNNGraph::CreateGraph`
        //       is fixed and does not change content of network passed (CVS-26420)
        auto localNetwork = cloneNet(static_cast<ICNNNetwork&>(*_clonedNetwork));
        auto graph = std::make_shared<MKLDNNGraph>();
        {
            std::unique_lock<std::mutex> lock{_cfgMutex};
            graph->setConfig(_cfg);
        }
        int numaNode = 0;
        auto* streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(_taskExecutor.get());
        if (nullptr != streamExecutor) {
            numaNode = streamExecutor->GetNumaNodeId();
        }
        graph->CreateGraph(static_cast<ICNNNetwork&>(*localNetwork), extensionManager, numaNodesWeights[numaNode]);
        return graph;
    }};

    _taskExecutor->runAndWait({std::thread::hardware_concurrency(), [this] {_graphs.local();}});

    // Save all MemoryLayer data tensors. Will use insight about mechanics
    // of MemoryLayer implementation. It uses output edge of MemoryLayer
    // producer as storage for tensor to keep it between infer calls.
    if (_graphs.size() == 1) {
        for (auto &node : _graphs.begin()->get()->GetNodes()) {
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
    {
        std::lock_guard<std::mutex> lock{_cfgMutex};
        _cfg.readProperties(properties);
    }
    for (auto g : _graphs) {
        g->setProperty(properties);
    }
}

void MKLDNNExecNetwork::CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncRequestImpl = std::make_shared<MKLDNNAsyncInferRequest>(syncRequestImpl, _taskExecutor, _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MKLDNNAsyncInferRequest>(asyncRequestImpl),
                       [](IInferRequest *p) { p->Release(); });

    asyncRequestImpl->SetPointerToPublicInterface(asyncRequest);
}

void MKLDNNExecNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";

    graphPtr = _graphs.begin()->get()->dump();
}

void MKLDNNExecNetwork::GetConfig(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";
    Config engConfig = _graphs.begin()->get()->getProperty();
    auto option = engConfig._config.find(name);
    if (option != engConfig._config.end()) {
        result = option->second;
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork config key: " << name;
    }
}

void MKLDNNExecNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (_graphs.size() == 0)
        THROW_IE_EXCEPTION << "No graph was found";

    if (name == METRIC_KEY(NETWORK_NAME)) {
        if (_graphs.begin()->get()->dump() == nullptr)
            THROW_IE_EXCEPTION << "Invalid graph dump";
        result = IE_SET_METRIC(NETWORK_NAME, _graphs.begin()->get()->dump()->getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && key : _graphs.begin()->get()->getProperty()._config) {
            configKeys.push_back(key.first);
        }
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        Config engConfig = _graphs.begin()->get()->getProperty();
        auto option = engConfig._config.find(CONFIG_KEY(CPU_THROUGHPUT_STREAMS));
        IE_ASSERT(option != engConfig._config.end());
        auto streams = std::stoi(option->second);
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(
            streams ? streams : 1));
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

        auto reshapeLayer = dynamic_cast<ReshapeLayer *>(layer.get());
        if (reshapeLayer &&
            type == Reshape &&
            (reshapeLayer->outData[0]->getTensorDesc().getDims()[0] ==
             reshapeLayer->insData[0].lock()->getTensorDesc().getDims()[0])) {
            return;
        }

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
