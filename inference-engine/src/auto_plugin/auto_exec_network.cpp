// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <memory>
#include <map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include "auto_plugin.hpp"
#include "auto_exec_network.hpp"
#include "auto_infer_request.hpp"
#include "auto_async_infer_request.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "transformations/utils/utils.hpp"

namespace AutoPlugin {
using namespace InferenceEngine;

thread_local AutoExecutableNetwork::WorkerInferRequest* AutoExecutableNetwork::_thisWorkerInferRequest = nullptr;

namespace {
std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
    auto nGraphFunc = network.getFunction();
    bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
    if (isINTModel) {
        return METRIC_VALUE(INT8);
    }
    for (auto & node : nGraphFunc->get_ordered_ops()) {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node) ||
            std::dynamic_pointer_cast<ngraph::op::DeconvolutionIE>(node)) {
          auto layerType = node->input(1).get_element_type().get_type_name();
          if (layerType == "f32")
              return METRIC_VALUE(FP32);
          if (layerType == "f16")
              return METRIC_VALUE(FP16);
        }
    }
    return METRIC_VALUE(FP32);
}
}  // namespace

struct IdleGuard {
    explicit IdleGuard(AutoExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       AutoExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    AutoExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    AutoExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    AutoExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

AutoExecutableNetwork::AutoExecutableNetwork(const std::string& modelPath,
                                             const InferenceEngine::CNNNetwork& network,
                                             const ConfigType& config,
                                             AutoInferencePlugin* plugin)
                                             : _autoPlugin(plugin) {
    if (_autoPlugin->GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    auto metaDevices = _autoPlugin->GetDeviceList(config);
    auto core = _autoPlugin->GetCore(); // shared_ptr that holds the Core while the lambda below (which captures that by val) works
    auto LoadNetworkAsync =
        [this, core, modelPath, network](const std::string& device) -> IE::SoExecutableNetworkInternal {
            IE::SoExecutableNetworkInternal executableNetwork;
            // std::cout << "!!! DEBUG: Starting Async loading to the " << device << " !!!" << std::endl;
            if (!modelPath.empty()) {
                executableNetwork = core->LoadNetwork(modelPath, device, {});
            } else {
                executableNetwork = core->LoadNetwork(network, device, {});
            }
            // std::cout << "!!! DEBUG: " << device << " was loaded !!!" << std::endl;

            uint32_t optimalNum {0};
            try {
                optimalNum = executableNetwork->GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<uint32_t>();
            } catch (const InferenceEngine::Exception &iie) {
                IE_THROW()
                    << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" << iie.what();
            }
            auto& workerRequests = _workerRequests[device];
            workerRequests.resize(optimalNum);
            auto& idleWorkerRequests = _idleWorkerRequests[device];
            idleWorkerRequests.set_capacity(optimalNum);
            auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
            for (auto&& workerRequest : workerRequests) {
                workerRequest._inferRequest = { executableNetwork, executableNetwork->CreateInferRequest() };
                auto* workerRequestPtr = &workerRequest;
                IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr));
                workerRequest._inferRequest->SetCallback(
                    [this, workerRequestPtr, device, idleWorkerRequestsPtr] (std::exception_ptr exceptionPtr) mutable {
                        IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                        workerRequestPtr->_exceptionPtr = exceptionPtr;
                        {
                            // this is the last task in pipeline
                            auto capturedTask = std::move(workerRequestPtr->_task);
                            capturedTask();
                        }
                        // try to return the request to the idle list (fails if the overall object destruction has began)
                        if (idleGuard.Release()->try_push(workerRequestPtr)) {
                            // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                            // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                            Task t;
                            if (_inferPipelineTasks.try_pop(t)) {
                                ScheduleToWorkerInferRequest(std::move(t));
                            }
                        }
                    });
            }

            if (device.find("CPU") == std::string::npos) {
                _networkActualNeeded = executableNetwork;
                _alreadyActualNetwork = true;
            }
            return executableNetwork;
        };

    // start CPU task
    const auto CPUIter = std::find_if(metaDevices.begin(), metaDevices.end(),
                                      [=](const std::string& d)->bool{return d.find("CPU") != std::string::npos;});
    if (CPUIter != metaDevices.end()) {
        _cpuDeviceName = *CPUIter;
        // init worker queue
        _idleWorkerRequests[_cpuDeviceName];
        _cpuFuture = std::async(std::launch::async, LoadNetworkAsync, _cpuDeviceName);
    }

    // start accelerator task, like GPU
    auto networkPrecision = GetNetworkPrecision(network);
    _acceleratorDeviceName = _autoPlugin->SelectDevice(metaDevices, networkPrecision);
    bool isAccelerator = _acceleratorDeviceName.find("CPU") == std::string::npos;
    if (isAccelerator) {
        // init worker queue
        _idleWorkerRequests[_acceleratorDeviceName];
        _acceleratorFuture = std::async(std::launch::async, LoadNetworkAsync, _acceleratorDeviceName);
    }

    _enablePerfCount = config.find(IE::PluginConfigParams::KEY_PERF_COUNT) != config.end()
                       && config.at(IE::PluginConfigParams::KEY_PERF_COUNT) == IE::PluginConfigParams::YES;

    // both are valid, like AUTO:CPU,GPU
    if (_cpuFuture.valid() && _acceleratorFuture.valid()) {
        try {
            _networkFirstReady = _cpuFuture.get();
            _alreadyActualNetwork = false;
        } catch (const std::exception& e) {
            printf("Warning: load network to CPU failed: %s\n", e.what());
            _networkActualNeeded = _acceleratorFuture.get();
            _alreadyActualNetwork = true;
        }
    } else if (_acceleratorFuture.valid()) {  // only accelerator is valid, like AUTO:GPU
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
    } else if (_cpuFuture.valid()) {  // only CPU is valid, like AUTO:CPU
        _networkActualNeeded = _cpuFuture.get();
        _alreadyActualNetwork = true;
    } else {
        IE_THROW() << "No device task available";
    }
}

AutoExecutableNetwork::~AutoExecutableNetwork() {
    // this is necessary to guarantee member destroyed after getting future
    if (!_alreadyActualNetwork) {
        // printf("!!! DEBUG: actual network is still not ready, wait that\n");
        _acceleratorFuture.get();
    }
    /* NOTE: The only threads that use `AutoExecutableNetwork` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& idleWorker : _idleWorkerRequests) {
        // stop accepting any idle requests back (for re-scheduling)
        idleWorker.second.set_capacity(0);
    }
    _workerRequests.clear();
}

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                                          OutputsDataMap networkOutputs) {
    InferenceEngine::SoExecutableNetworkInternal network;
    SoIInferRequestInternal inferRequest;
    if (TryGetActualNetwork(network)) {
        inferRequest = {_networkActualNeeded, _networkActualNeeded->CreateInferRequest()};
    } else {
        inferRequest = {_networkFirstReady, _networkFirstReady->CreateInferRequest()};
    }
    return std::make_shared<AutoInferRequest>(_networkInputs, _networkOutputs, inferRequest);
}

InferenceEngine::IInferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequest() {
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return std::make_shared<AutoAsyncInferRequest>(std::static_pointer_cast<AutoInferRequest>(syncRequestImpl),
                                                   std::static_pointer_cast<AutoExecutableNetwork>(shared_from_this()),
                                                   std::make_shared<InferenceEngine::ImmediateExecutor>(),
                                                   _enablePerfCount);
}

void AutoExecutableNetwork::run(InferenceEngine::Task inferTask) {
    ScheduleToWorkerInferRequest(std::move(inferTask), "");
}

bool AutoExecutableNetwork::TryGetActualNetwork(InferenceEngine::SoExecutableNetworkInternal& soExecNetwork) {
    // if already get actual network
    if (_alreadyActualNetwork) {
        soExecNetwork = _networkActualNeeded;
        // reapply config to actual network
        // fixme: GPU doesn't support SetConfig and throw exception
        try {
            _networkActualNeeded->SetConfig(_cacheConfig);
        } catch (...) {
        }
        return true;
    }
    return false;
}

void AutoExecutableNetwork::WaitForActualDevice() const {
    if (_alreadyActualNetwork) {
        return;
    }

    if (_acceleratorFuture.valid()) {
        _networkActualNeeded = _acceleratorFuture.get();
        _alreadyActualNetwork = true;
    } else {
        IE_THROW() << "Export failed due to no valid executable network";
    }
}

void AutoExecutableNetwork::Export(std::ostream& networkModel) {
    //fixme: the Export  should work with actual device, so we have to wait!!!
    WaitForActualDevice();
    _networkActualNeeded->Export(networkModel);
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
    // fixme: the GetContext  should work with actual device, so we have to wait!!!
    WaitForActualDevice();
    return _networkActualNeeded->GetContext();
}

InferenceEngine::CNNNetwork AutoExecutableNetwork::GetExecGraphInfo() {
    WaitForActualDevice();
    return _networkActualNeeded->GetExecGraphInfo();
}

Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    // fixme: should we wait actual device? meanwhile it will block inference, how to fix?
//    WaitForActualDevice();
    if (_alreadyActualNetwork) {
        return _networkActualNeeded->GetMetric(name);
    } else {
        return _networkFirstReady->GetMetric(name);
    }
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, Parameter>& config) {
    //fixme: have to store the config and reapply when the networks swapped
    _cacheConfig = config;
    if (_alreadyActualNetwork) {
        _networkActualNeeded->SetConfig(config);
    } else {
        _networkFirstReady->SetConfig(config);
    }
}

Parameter AutoExecutableNetwork::GetConfig(const std::string& name) const {
    //fixme: carefuly select between FirstLoaded and ActuallyNeeded
    return _cacheConfig;
}

void AutoExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask, DeviceName preferred_device) {
    // printf("%s:%d\n", __FUNCTION__, __LINE__);
    WorkerInferRequest* workerRequestPtr = nullptr;
    // _acceleratorDeviceName could be the same as _cpuDeviceName, such as AUTO:CPU
    auto& idleWorkerRequests = _alreadyActualNetwork ? _idleWorkerRequests[_acceleratorDeviceName]: _idleWorkerRequests[_cpuDeviceName];
    if (idleWorkerRequests.try_pop(workerRequestPtr)) {
        IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
        _thisWorkerInferRequest = workerRequestPtr;
        {
          auto capturedTask = std::move(inferPipelineTask);
          capturedTask();
        }
        idleGuard.Release();
        return;
    }

    // no vacant requests this time, storing the task to the respective queue
    _inferPipelineTasks.push(std::move(inferPipelineTask));
    // printf("!!! DEBUG: _inferPipelineTasks size = %zu\n", _inferPipelineTasks.unsafe_size());
}

}  // namespace AutoPlugin
