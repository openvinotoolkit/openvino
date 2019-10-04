// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <ie_metric_helpers.hpp>
#include "cnn_network_impl.hpp"
#include "exec_graph_info.hpp"
#include <myriad_executable_network.h>
#include <vpu/blob_reader.hpp>
#include <vpu/utils/profiling.hpp>
#include <net_pass.h>

using namespace InferenceEngine;

namespace vpu {
namespace MyriadPlugin {

static void selectNumberOfExecutors(const ncDevicePlatform_t& platform,
                                    std::uint32_t numShaves, std::uint32_t numSlices, int& numExecutors) {
    const std::uint32_t maxShaves = platform == NC_MYRIAD_2 ? 12 : 16;
    const std::uint32_t maxSlices = platform == NC_MYRIAD_2 ? 15 : 19;

    if (numExecutors == MyriadConfig::UNDEFINED_THROUGHPUT_STREAMS) {
        const std::uint32_t defaultPlatformExecutors = platform == NC_MYRIAD_2 ? 1 : 2;
        auto getMaximumAvailableExecutors = [&]() { return std::min(maxShaves / numShaves, maxSlices / numSlices); };

        numExecutors = std::min(defaultPlatformExecutors, getMaximumAvailableExecutors());
    }

    if (numExecutors < 1) {
        THROW_IE_EXCEPTION << "Number of executors must be not less than 1, " << numExecutors << " provided";
    }

    auto isEnoughResources = [&]() {
        return numShaves * numExecutors <= maxShaves && numSlices * numExecutors <= maxSlices;
    };

    if (!isEnoughResources()) {
        THROW_IE_EXCEPTION << "There are no enough resources for using " << platform << " on "
                           << (platform == NC_MYRIAD_2 ? "MYRIAD_2" : "MYRIAD_X");
    }
}

ExecutableNetwork::ExecutableNetwork(std::vector<DevicePtr> &devicePool,
    const std::map<std::string, std::string> &config, ConfigMode mode) {
    VPU_PROFILE(ExecutableNetwork);
    _config = std::make_shared<MyriadConfig>(config, mode);

    _log = std::make_shared<Logger>("MyriadPlugin", _config->hostLogLevel, consoleOutput());
    _executor = std::make_shared<MyriadExecutor>(_config->forceReset, _config->deviceLogLevel, _log);
    _device = _executor->openDevice(devicePool, _config);
    _supportedMetrics = {
        METRIC_KEY(NETWORK_NAME),
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
        METRIC_KEY(DEVICE_THERMAL)
    };

    // ignore hardware optimization config for MYRIAD2, it is always disabled
    if (_device->_platform == NC_MYRIAD_2) {
        _config->compileConfig.hwOptimization = false;
    }
}

ExecutableNetwork::ExecutableNetwork(ICNNNetwork &network, std::vector<DevicePtr> &devicePool,
                                     const std::map<std::string, std::string> &config) :
                                     ExecutableNetwork(devicePool, config) {
    VPU_PROFILE(ExecutableNetwork);
    bool ti_proc_ok = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

    auto compiledGraph = compileNetwork(
        network,
        static_cast<Platform>(_device->_platform),
        _config->compileConfig,
        std::make_shared<Logger>("GraphCompiler", _config->hostLogLevel, consoleOutput()));

    selectNumberOfExecutors(_device->_platform,
                            compiledGraph->numShaves, compiledGraph->numSlices, _config->numExecutors);

    _graphBlob = std::move(compiledGraph->blob);
    _graphMetaData = std::move(compiledGraph->graphMeta);

    _inputInfo  = std::move(compiledGraph->inputInfo);
    _outputInfo = std::move(compiledGraph->outputInfo);

    if (!_device->isBooted()) {
        return;
    }

    char networkName[1024] = {};
    network.getName(networkName, sizeof(networkName));
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, compiledGraph->blobHeader,
                             compiledGraph->numActiveStages, networkName, _config->numExecutors);
    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

ExecutableNetwork::ExecutableNetwork(const std::string &blobFilename,
                           std::vector<DevicePtr> &devicePool,
                           const std::map<std::string, std::string> &config) :
                           ExecutableNetwork(devicePool, config, ConfigMode::RUNTIME_MODE) {
    VPU_PROFILE(ExecutableNetwork);
    std::ifstream blobFile(blobFilename, std::ios::binary);
    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));

    if (!_device->isBooted()) {
        return;
    }

    // TODO: better name
    char networkName[1024] = "importedNetwork";

    BlobReader blobReader;
    blobReader.parse(_graphBlob);

    selectNumberOfExecutors(_device->_platform,
                            blobReader.getNumberOfShaves(), blobReader.getNumberOfSlices(), _config->numExecutors);

    this->_networkInputs  = blobReader.getNetworkInputs();
    this->_networkOutputs = blobReader.getNetworkOutputs();
    std::size_t numStages = blobReader.getStageCount();
    auto blobHeader = blobReader.getHeader();


    _inputInfo  = blobReader.getInputInfo();
    _outputInfo = blobReader.getOutputInfo();

    _executor->allocateGraph(_device, _graphDesc, _graphBlob, blobHeader, numStages, networkName,
                             _config->numExecutors);

    _graphMetaData.stagesMeta.resize(numStages);
    for (auto &meta : _graphMetaData.stagesMeta) {
        meta.stageName = meta.stageType = meta.layerName = meta.layerType = "UNKNOWN";
        meta.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    }

    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

void ExecutableNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        result = IE_SET_METRIC(NETWORK_NAME, _graphDesc._name);
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, _supportedMetrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, std::vector<std::string>());
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(2u*_config->numExecutors));
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        result = IE_SET_METRIC(DEVICE_THERMAL, _executor->GetThermal(_device));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}

void ExecutableNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    graphPtr = buildRuntimeGraph(_graphMetaData);
}

InferenceEngine::ICNNNetwork::Ptr ExecutableNetwork::buildRuntimeGraph(GraphMetaInfo& graphMetaInfo) {
    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>();
    net->setPrecision(Precision::FP16);
    net->setName(graphMetaInfo.graphName);

    std::map<size_t, CNNLayerPtr> stageMetaIndexToLayer;

    auto createLayerFromMeta = [&](const StageMetaInfo &stageMetaInfo) -> CNNLayer::Ptr {
        auto layer = std::make_shared<CNNLayer>(LayerParams{stageMetaInfo.stageName,
                                          stageMetaInfo.layerType,
                                          Precision::FP16});

        layer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = stageMetaInfo.layerName;
        layer->params[ExecGraphInfoSerialization::IMPL_TYPE] = stageMetaInfo.stageType;
        layer->params[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(stageMetaInfo.execOrder);

        std::stringstream layoutStream;
        int ind = 0;
        for (auto &outLayout : stageMetaInfo.outLayouts) {
            if (ind == 0) {
                layoutStream << outLayout;
                ind++;
                continue;
            }
            layoutStream << ',' << outLayout;
        }
        layer->params[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = layoutStream.str();

        std::string outPrecisionsStr;
        ind = 0;
        for (auto &outPrecision : stageMetaInfo.outPrecisions) {
            if (ind == 0) {
                outPrecisionsStr += outPrecision.name();
                ind++;
                continue;
            }
            outPrecisionsStr += ',' + std::string(outPrecision.name());
        }
        layer->params[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outPrecisionsStr;

        if (stageMetaInfo.execOrder < 0) {
            layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";
        } else {
            layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(stageMetaInfo.execTime);
        }

        return layer;
    };

    //
    // Write performance counts
    //

    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);

    const auto deviceTimings = perfInfo.data();
    auto deviceTimingsCount = perfInfo.size();

    if (deviceTimingsCount > 0) {
        std::size_t timeIndex = 0;

        for (auto &stageMeta : graphMetaInfo.stagesMeta) {
            if (stageMeta.status == ie::InferenceEngineProfileInfo::EXECUTED &&
                timeIndex < deviceTimingsCount) {
                stageMeta.execTime += deviceTimings[timeIndex];
                timeIndex++;
            }
        }
    }

    //
    // Add all stages to network
    //

    for (std::size_t i = 0; i < graphMetaInfo.stagesMeta.size(); i++) {
        const auto stageMetaData = graphMetaInfo.stagesMeta[i];

        if (stageMetaData.status == ie::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT ||
            stageMetaData.stageName == "<Receive-Tensor>" ||
            stageMetaData.stageName == "<none>") {
            continue;
        }

        auto layer = createLayerFromMeta(stageMetaData);
        stageMetaIndexToLayer.insert(std::make_pair(i, layer));
        net->addLayer(layer);
    }

    //
    // Add all edges to network
    //

    for (const auto &dataMetaData : graphMetaInfo.datasMeta) {
        DataPtr data;

        auto parent = stageMetaIndexToLayer[dataMetaData.parentIndex];
        data = std::make_shared<Data>(dataMetaData.name, dataMetaData.desc);
        parent->outData.push_back(data);
        data->getCreatorLayer() = parent;

        for (auto &childMetaIndex : dataMetaData.childrenIndices) {
            auto child = stageMetaIndexToLayer[childMetaIndex];
            data->getInputTo()[child->name] = child;
            child->insData.push_back(data);
        }
    }

    //
    // Specify inputs data
    //

    for (std::size_t i = 0; i < graphMetaInfo.stagesMeta.size(); i++) {
        const auto stageMetaData = graphMetaInfo.stagesMeta[i];

        if (stageMetaData.inputsNum != 0 ||
            stageMetaData.stageName == "<Receive-Tensor>" ||
            stageMetaData.stageName == "<none>") {
            continue;
        }

        auto input = stageMetaIndexToLayer[i];
        auto inputInfo = std::make_shared<InputInfo>();
        inputInfo->setInputData(input->outData[0]);
        net->setInputInfo(inputInfo);
    }

    return net;
}

}  // namespace MyriadPlugin
}  // namespace vpu
