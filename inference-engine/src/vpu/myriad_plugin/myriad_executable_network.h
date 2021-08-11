// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <sstream>
#include <fstream>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <threading/ie_executor_manager.hpp>

#include <vpu/graph_transformer.hpp>

#include "myriad_executor.h"
#include "myriad_infer_request.h"
#include "myriad_async_infer_request.h"

namespace vpu {
namespace MyriadPlugin {

class ExecutableNetwork : public ie::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecutableNetwork> Ptr;

    ExecutableNetwork(const InferenceEngine::CNNNetwork& network, std::shared_ptr<IMvnc> mvnc, std::vector<DevicePtr> &devicePool,
                      const PluginConfiguration& configuration, const std::shared_ptr<ie::ICore> core);

    ExecutableNetwork(std::istream& strm, std::shared_ptr<IMvnc> mvnc, std::vector<DevicePtr> &devicePool, const PluginConfiguration& configuration,
                      const std::shared_ptr<ie::ICore> core);

    ExecutableNetwork(const std::string &blobFilename, std::shared_ptr<IMvnc> mvnc, std::vector<DevicePtr> &devicePool,
                      const PluginConfiguration& configuration, const std::shared_ptr<ie::ICore> core);

    virtual ~ExecutableNetwork() {
        try {
            if (_device != nullptr) {
                _executor->deallocateGraph(_device, _graphDesc);
            }
        }
        catch (...) {
            std::cerr << "ERROR ~ExecutableNetwork():\n"
                      << "Some errors occurred during the calling of the deallocateGraph() method";
        }
    }

    ie::IInferRequestInternal::Ptr CreateInferRequestImpl(ie::InputsDataMap networkInputs,
                                                         ie::OutputsDataMap networkOutputs) override {
        if (!_isNetworkConstant && (_device == nullptr || !_device->isBooted())) {
            IE_THROW() << "Can not create infer request: there is no available devices with platform "
                               << _device->_platform;
        }

        return std::make_shared<MyriadInferRequest>(_graphDesc, networkInputs, networkOutputs,
                                                    _inputInfo, _outputInfo,
                                                    _graphMetaData.stagesMeta, _config, _log, _executor,
                                                    _constDatas, _isNetworkConstant);
    }

    ie::IInferRequestInternal::Ptr CreateInferRequest() override {
        if (!_isNetworkConstant && (_device == nullptr || !_device->isBooted())) {
            IE_THROW() << "Can not create infer request: there is no available devices with platform "
                               << _device->_platform;
        }

        auto syncRequestImpl = std::make_shared<MyriadInferRequest>(_graphDesc, _networkInputs, _networkOutputs,
                                                                    _inputInfo, _outputInfo,
                                                                    _graphMetaData.stagesMeta, _config, _log,
                                                                    _executor, _constDatas, _isNetworkConstant);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutor();
        return std::make_shared<MyriadAsyncInferRequest>(
                syncRequestImpl, _taskExecutor, _callbackExecutor, taskExecutorGetResult);
    }

    void Export(std::ostream& model) override {
        model.write(_graphBlob.data(), _graphBlob.size());
    }

    void Export(const std::string &modelFileName) override {
        std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

        if (modelFile.is_open()) {
            Export(modelFile);
        } else {
            IE_THROW() << "The " << modelFileName << " file can not be opened for export";
        }
    }

    ie::Parameter GetMetric(const std::string &name) const override;

    ie::CNNNetwork GetExecGraphInfo() override;

    void Import(std::istream& strm, std::vector<DevicePtr> &devicePool, const PluginConfiguration& configuration);

private:
    Logger::Ptr _log;
    MyriadExecutorPtr _executor;
    std::vector<char> _graphBlob;
    GraphDesc _graphDesc;
    DevicePtr _device;
    GraphMetaInfo _graphMetaData;
    PluginConfiguration _config;
    bool _isNetworkConstant = false;
    const std::shared_ptr<ie::ICore> _core = nullptr;
    int _actualNumExecutors = 0;
    std::vector<std::string> _supportedMetrics;
    std::map<std::string, ie::Blob::Ptr> _constDatas;

    DataInfo _inputInfo;
    DataInfo _outputInfo;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    ExecutableNetwork(std::shared_ptr<IMvnc> mvnc,
        const PluginConfiguration& config,
        const std::shared_ptr<ie::ICore> core);

    ie::ITaskExecutor::Ptr getNextTaskExecutor() {
        std::string id = _taskExecutorGetResultIds.front();

        _taskExecutorGetResultIds.pop();
        _taskExecutorGetResultIds.push(id);

        ie::ExecutorManager *executorManager = ie::ExecutorManager::getInstance();
        ie::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

        return taskExecutor;
    }

    void openDevice(std::vector<DevicePtr>& devicePool);
};

}  // namespace MyriadPlugin
}  // namespace vpu
