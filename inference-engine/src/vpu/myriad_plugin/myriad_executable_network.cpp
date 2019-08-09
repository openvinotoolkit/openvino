// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <myriad_executable_network.h>
#include <vpu/blob_reader.hpp>
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
    _stagesMetaData = std::move(compiledGraph->stagesMeta);

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

    _stagesMetaData.resize(numStages);
    for (auto &meta : _stagesMetaData) {
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

}  // namespace MyriadPlugin
}  // namespace vpu
