// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include <myriad_executable_network.h>
#include <vpu/blob_reader.hpp>
#include <net_pass.h>

using namespace InferenceEngine;

namespace vpu {
namespace MyriadPlugin {

ExecutableNetwork::ExecutableNetwork(ICNNNetwork &network, std::vector<DevicePtr> &devicePool,
                                     const std::map<std::string, std::string> &config) {
    _config = std::make_shared<MyriadConfig>(config);

    _log = std::make_shared<Logger>("MyriadPlugin", _config->logLevel, consoleOutput());

    _executor = std::make_shared<MyriadExecutor>(_config->forceReset, _config->vpuLogLevel, _log);
    _device = _executor->openDevice(devicePool, _config);

    // ignore hardware optimization config for MYRIAD2, it is always disabled
    if (_device->_platform == MYRIAD_2) {
        _config->compileConfig.hwOptimization = false;
    }

    bool ti_proc_ok = !NetPass::CombineRNNSeq(network) ? NetPass::UnrollTI(network) : true;
    if (!ti_proc_ok)
        THROW_IE_EXCEPTION << "Plugin doesn't support Tensor Iterator in pure form. "
                              "None TI optimization pattern has been applied successfully";

    auto compiledGraph = compileNetwork(
        network,
        static_cast<Platform>(_device->_platform),
        _config->compileConfig,
        std::make_shared<Logger>("GraphCompiler", _config->logLevel, consoleOutput()));

    _graphBlob = std::move(compiledGraph->blob);
    _stagesMetaData = std::move(compiledGraph->stagesMeta);

    _inputInfo  = std::move(compiledGraph->inputInfo);
    _outputInfo = std::move(compiledGraph->outputInfo);

    if (!_device->isBooted()) {
        return;
    }

    char networkName[1024] = {};
    network.getName(networkName, sizeof(networkName));
    _executor->allocateGraph(_device, _graphDesc, _graphBlob, compiledGraph->blobHeader, compiledGraph->numActiveStages, networkName);
    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor(
                TargetDeviceInfo::name(TargetDevice::eMYRIAD));
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

ExecutableNetwork::ExecutableNetwork(const std::string &blobFilename,
                           std::vector<DevicePtr> &devicePool,
                           const std::map<std::string, std::string> &config) {
    _config = std::make_shared<MyriadConfig>(config, ConfigMode::RUNTIME_MODE);

    _log = std::make_shared<Logger>("MyriadPlugin", _config->logLevel, consoleOutput());

    _executor = std::make_shared<MyriadExecutor>(_config->forceReset, _config->vpuLogLevel, _log);
    _device = _executor->openDevice(devicePool, _config);

    // ignore hardware optimization config for MYRIAD2, it is always disabled
    if (_device->_platform == MYRIAD_2) {
        _config->compileConfig.hwOptimization = false;
    }

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

    this->_networkInputs  = blobReader.getNetworkInputs();
    this->_networkOutputs = blobReader.getNetworkOutputs();
    std::size_t numStages = blobReader.getStageCount();
    auto blobHeader = blobReader.getHeader();


    _inputInfo  = blobReader.getInputInfo();
    _outputInfo = blobReader.getOutputInfo();

    _executor->allocateGraph(_device, _graphDesc, _graphBlob, blobHeader, numStages, networkName);

    _stagesMetaData.resize(numStages);
    for (auto &meta : _stagesMetaData) {
        meta.stageName = meta.stageType = meta.layerName = meta.layerType = "UNKNOWN";
        meta.status = InferenceEngineProfileInfo::LayerStatus::EXECUTED;
    }

    if (_config->exclusiveAsyncRequests) {
        ExecutorManager *executorManager = ExecutorManager::getInstance();
        _taskExecutor = executorManager->getExecutor(
                TargetDeviceInfo::name(TargetDevice::eMYRIAD));
    }

    for (size_t i = 0; i < _maxTaskExecutorGetResultCount; i++) {
        std::stringstream idStream;
        idStream << networkName << "_TaskExecutorGetResult" << i;
        _taskExecutorGetResultIds.emplace(idStream.str());
    }
}

}  // namespace MyriadPlugin
}  // namespace vpu
