// Copyright (C) 2018-2020 Intel Corporation
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
#include <vpu/utils/runtime_graph.hpp>
#include <details/ie_cnn_network_tools.h>
#include <net_pass.h>
#include <vpu/compile_env.hpp>

using namespace InferenceEngine;

namespace vpu {
namespace MyriadPlugin {

ExecutableNetwork::ExecutableNetwork(
        std::vector<DevicePtr>& devicePool,
        const MyriadConfig& config) :
            _config(config) {
    VPU_PROFILE(ExecutableNetwork);

    _log = std::make_shared<Logger>(
        "MyriadPlugin",
        _config.logLevel(),
        defaultOutput(_config.pluginLogFilePath()));

    _executor = std::make_shared<MyriadExecutor>(_config.forceReset(), _config.logLevel(), _log);
    _device = _executor->openDevice(devicePool, _config);

    const auto& compileConfig = config.compileConfig();
    const auto& revision = _device->revision();
    _actualNumExecutors = compileConfig.numExecutors != -1 ? compileConfig.numExecutors : DefaultAllocation::numStreams(revision, compileConfig);

    _supportedMetrics = {
        METRIC_KEY(NETWORK_NAME),
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
        METRIC_KEY(DEVICE_THERMAL)
    };
}

ExecutableNetwork::ExecutableNetwork(
        ICNNNetwork& network, std::vector<DevicePtr>& devicePool,
        const MyriadConfig& config) :
            ExecutableNetwork(devicePool, config) {
    VPU_PROFILE(ExecutableNetwork);

    const auto compilerLog = std::make_shared<Logger>(
        "GraphCompiler",
        _config.logLevel(),
        defaultOutput(_config.compilerLogFilePath()));

    if (_device == nullptr)
        THROW_IE_EXCEPTION << "No device was detected";
    auto compiledGraph = compileNetwork(
        network,
        static_cast<Platform>(_device->_platform),
        _config.compileConfig(),
        compilerLog);

    _actualNumExecutors = compiledGraph->numExecutors;
    _graphBlob = std::move(compiledGraph->blob);
    _graphMetaData = std::move(compiledGraph->graphMeta);

    _inputInfo  = std::move(compiledGraph->inputInfo);
    _outputInfo = std::move(compiledGraph->outputInfo);

    if (!_device->isBooted()) {
        return;
    }

    auto networkName = network.getName();
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, compiledGraph->blobHeader, compiledGraph->numActiveStages, networkName, _actualNumExecutors);
    if (_config.exclusiveAsyncRequests()) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

void ExecutableNetwork::Import(std::istream& strm,
                               std::vector<DevicePtr> &devicePool,
                               const MyriadConfig& config) {
    std::ostringstream blobContentStream;
    blobContentStream << strm.rdbuf();
    const std::string& blobContentString = blobContentStream.str();
    std::copy(blobContentString.begin(), blobContentString.end(), std::back_inserter(_graphBlob));

    if (!_device->isBooted()) {
        return;
    }

    // TODO: better name
    char networkName[1024] = "importedNetwork";

    BlobReader blobReader;
    blobReader.parse(_graphBlob);

    this->_networkInputs  = blobReader.getNetworkInputs();
    this->_networkOutputs = blobReader.getNetworkOutputs();
    std::size_t numStages = blobReader.getStageCount();
    auto blobHeader = blobReader.getHeader();

    _inputInfo  = blobReader.getInputInfo();
    _outputInfo = blobReader.getOutputInfo();

    _executor->allocateGraph(_device, _graphDesc, _graphBlob, blobHeader, numStages, networkName, _actualNumExecutors);

    _graphMetaData.stagesMeta.resize(numStages);
    for (auto &meta : _graphMetaData.stagesMeta) {
        meta.stageName = meta.stageType = meta.layerName = meta.layerType = "UNKNOWN";
        meta.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    }

    if (_config.exclusiveAsyncRequests()) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor("MYRIAD");
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& strm,
                               std::vector<DevicePtr> &devicePool,
                               const MyriadConfig& config) :
    ExecutableNetwork(devicePool, config) {
    VPU_PROFILE(ExecutableNetwork);
    Import(strm, devicePool, config);
}

ExecutableNetwork::ExecutableNetwork(
        const std::string& blobFilename,
        std::vector<DevicePtr>& devicePool,
        const MyriadConfig& config) :
    ExecutableNetwork(devicePool, config) {
    VPU_PROFILE(ExecutableNetwork);
    std::ifstream blobFile{blobFilename, std::ios::binary};
    Import(blobFile, devicePool, config);
}

void ExecutableNetwork::GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        result = IE_SET_METRIC(NETWORK_NAME, _graphDesc._name);
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        result = IE_SET_METRIC(SUPPORTED_METRICS, _supportedMetrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        result = IE_SET_METRIC(SUPPORTED_CONFIG_KEYS, std::vector<std::string>());
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, static_cast<unsigned int>(2u * _actualNumExecutors));
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        result = IE_SET_METRIC(DEVICE_THERMAL, _executor->GetThermal(_device));
    } else {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
    }
}

void ExecutableNetwork::GetExecGraphInfo(InferenceEngine::ICNNNetwork::Ptr &graphPtr) {
    auto perfInfo = _executor->getPerfTimeInfo(_graphDesc._graphHandle);
    graphPtr = buildRuntimeGraph(_graphMetaData, perfInfo);
}

}  // namespace MyriadPlugin
}  // namespace vpu
