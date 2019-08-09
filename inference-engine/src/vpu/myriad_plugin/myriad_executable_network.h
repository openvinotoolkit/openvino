// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <sstream>
#include <fstream>

#include <ie_common.h>
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include <cpp_interfaces/ie_executor_manager.hpp>

#include <vpu/graph_transformer.hpp>
#include <vpu/parsed_config.hpp>

#include "myriad_executor.h"
#include "myriad_executable_network.h"
#include "myriad_infer_request.h"
#include "myriad_async_infer_request.h"
#include "myriad_config.h"

namespace vpu {
namespace MyriadPlugin {

class ExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<ExecutableNetwork> Ptr;

    explicit ExecutableNetwork(InferenceEngine::ICNNNetwork &network,
                               std::vector<DevicePtr> &devicePool,
                               const std::map<std::string, std::string> &config);


    explicit ExecutableNetwork(const std::string &blobFilename,
                               std::vector<DevicePtr> &devicePool,
                               const std::map<std::string, std::string> &config);

    virtual ~ExecutableNetwork() {
        try {
            _executor->deallocateGraph(_device, _graphDesc);
        }
        catch (...) {
            std::cerr << "ERROR ~ExecutableNetwork():\n"
                      << "Some errors occurred during the calling of the deallocateGraph() method";
        }
    }

    InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                      InferenceEngine::OutputsDataMap networkOutputs) override {
        return std::make_shared<MyriadInferRequest>(_graphDesc, networkInputs, networkOutputs,
                                                    _inputInfo, _outputInfo,
                                                    _stagesMetaData, _config, _log, _executor);
    }

    void CreateInferRequest(InferenceEngine::IInferRequest::Ptr &asyncRequest) override {
        if (!_device->isBooted()) {
            THROW_IE_EXCEPTION << "Can not create infer request: there is no available devices with platform "
                               << _device->_platform;
        }

        auto syncRequestImpl = std::make_shared<MyriadInferRequest>(_graphDesc, _networkInputs, _networkOutputs,
                                                                    _inputInfo, _outputInfo,
                                                                    _stagesMetaData, _config, _log,
                                                                    _executor);
        syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
        auto taskExecutorGetResult = getNextTaskExecutor();
        auto asyncTreadSafeImpl = std::make_shared<MyriadAsyncInferRequest>(
                syncRequestImpl, _taskExecutor, taskExecutorGetResult, _taskSynchronizer, _callbackExecutor);
        asyncRequest.reset(new InferenceEngine::InferRequestBase<InferenceEngine::AsyncInferRequestThreadSafeDefault>(
                           asyncTreadSafeImpl),
                           [](InferenceEngine::IInferRequest *p) { p->Release(); });
        asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    }

    void Export(const std::string &modelFileName) override {
        std::ofstream modelFile(modelFileName, std::ios::out | std::ios::binary);

        if (modelFile.is_open()) {
            modelFile.write(_graphBlob.data(), _graphBlob.size());
        } else {
            THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
        }
    }

    void GetMetric(const std::string &name, InferenceEngine::Parameter &result, InferenceEngine::ResponseDesc *resp) const override;

    void GetMappedTopology(
            std::map<std::string, std::vector<InferenceEngine::PrimitiveInfo::Ptr>> &deployedTopology) override {
        THROW_IE_EXCEPTION << "GetMappedTopology is not implemented\n";
    }

private:
    Logger::Ptr _log;
    MyriadExecutorPtr _executor;
    std::vector<char> _graphBlob;
    GraphDesc _graphDesc;
    DevicePtr _device;
    std::vector<StageMetaInfo> _stagesMetaData;
    std::shared_ptr<MyriadConfig> _config;
    std::vector<std::string> _supportedMetrics;

    DataInfo _inputInfo;
    DataInfo _outputInfo;

    const size_t _maxTaskExecutorGetResultCount = 1;
    std::queue<std::string> _taskExecutorGetResultIds;

    ExecutableNetwork(std::vector<DevicePtr> &devicePool,
                      const std::map<std::string, std::string> &config,
                      ConfigMode mode = ConfigMode::DEFAULT_MODE);

    InferenceEngine::ITaskExecutor::Ptr getNextTaskExecutor() {
        std::string id = _taskExecutorGetResultIds.front();

        _taskExecutorGetResultIds.pop();
        _taskExecutorGetResultIds.push(id);

        InferenceEngine::ExecutorManager *executorManager = InferenceEngine::ExecutorManager::getInstance();
        InferenceEngine::ITaskExecutor::Ptr taskExecutor = executorManager->getExecutor(id);

        return taskExecutor;
    }
};

}  // namespace MyriadPlugin
}  // namespace vpu
