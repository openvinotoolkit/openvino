// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <atomic>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>


#include "ie_metric_helpers.hpp"
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>
#include <multi-device/multi_device_config.hpp>
#include <ie_plugin_config.hpp>
#include "multi_device_exec_network.hpp"
#include "multi_device_async_infer_request.hpp"
#include "multi_device_plugin.hpp"

// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
    using namespace InferenceEngine;

thread_local MultiDeviceExecutableNetwork::WorkerInferRequest* MultiDeviceExecutableNetwork::_thisWorkerInferRequest = nullptr;

struct IdleGuard {
    explicit IdleGuard(MultiDeviceExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       MultiDeviceExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    MultiDeviceExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    MultiDeviceExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

MultiDeviceExecutableNetwork::MultiDeviceExecutableNetwork(const DeviceMap<InferenceEngine::ExecutableNetwork>&                 networksPerDevice,
                                                           const std::vector<DeviceInformation>&                                networkDevices,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _devicePriorities{networkDevices},
    _devicePrioritiesInitial{networkDevices},
    _networksPerDevice{networksPerDevice},
    _config{config},
    _needPerfCounters{needPerfCounters} {
    _taskExecutor.reset();
    for (auto&& networkValue : _networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;

        auto itNumRequests = std::find_if(_devicePriorities.cbegin(), _devicePriorities.cend(),
                [&device](const DeviceInformation& d){ return d.deviceName == device;});
        unsigned int optimalNum = 0;
        try {
            optimalNum = network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        } catch (const details::InferenceEngineException &iie) {
            THROW_IE_EXCEPTION
                    << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" << iie.what();
        }
        const auto numRequests = (_devicePriorities.end() == itNumRequests ||
            itNumRequests->numRequestsPerDevices == -1) ? optimalNum : itNumRequests->numRequestsPerDevices;
        auto& workerRequests = _workerRequests[device];
        auto& idleWorkerRequests = _idleWorkerRequests[device];
        workerRequests.resize(numRequests);
        auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
        idleWorkerRequests.set_capacity(numRequests);
        for (auto&& workerRequest : workerRequests) {
            workerRequest._inferRequest = network.CreateInferRequest();
            auto* workerRequestPtr = &workerRequest;
            IE_ASSERT(idleWorkerRequests.try_push(workerRequestPtr) == true);
            workerRequest._inferRequest.SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
                [workerRequestPtr, this, device, idleWorkerRequestsPtr] (InferRequest , StatusCode status) mutable {
                    IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                    workerRequestPtr->_status = status;
                    {
                        auto capturedTask = std::move(workerRequestPtr->_task);
                        capturedTask();
                    }
                    // try to return the request to the idle list (fails if the overall object destruction has began)
                    if (idleGuard.Release()->try_push(workerRequestPtr)) {
                        Task t;
                        // try pop the task, as we know there is at least one idle request
                        if (_inferPipelineTasks.try_pop(t)) {
                            // if succeeded, let's schedule that
                            ScheduleToWorkerInferRequest(std::move(t));
                        }
                    }
                });
        }
    }
}

void MultiDeviceExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask) {
    auto devices = [&] {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devicePriorities;
    }();
    for (auto&& device : devices) {
        WorkerInferRequest* workerRequestPtr = nullptr;
        NotBusyWorkerRequests& idleWorkerRequests = _idleWorkerRequests[device.deviceName];
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
    }
    // no vacant requests this time, storing the task to the queue
    _inferPipelineTasks.push(std::move(inferPipelineTask));
}

void MultiDeviceExecutableNetwork::run(Task inferPipelineTask) {
    ScheduleToWorkerInferRequest(std::move(inferPipelineTask));
}

MultiDeviceExecutableNetwork::~MultiDeviceExecutableNetwork() {
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _devicePriorities.clear();
    }
    /* NOTE: The only threads that use `MultiDeviceExecutableNetwork` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& networkValue : _networksPerDevice) {
        // stop accepting any idle requests back (for re-scheduling)
        _idleWorkerRequests.at(networkValue.first).set_capacity(0);
    }
    _workerRequests.clear();
}

InferenceEngine::InferRequestInternal::Ptr MultiDeviceExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                                InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    InferenceEngine::InferRequest request_to_share_blobs_with;
    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequest::Ptr MultiDeviceExecutableNetwork::CreateInferRequest() {
    IInferRequest::Ptr asyncRequest;
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncTreadSafeImpl = std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                                             _needPerfCounters,
                                                                             std::static_pointer_cast<MultiDeviceExecutableNetwork>(shared_from_this()),
                                                                             _callbackExecutor);
    asyncRequest.reset(new InferRequestBase<MultiDeviceAsyncInferRequest>(asyncTreadSafeImpl), [](IInferRequest *p) { p->Release(); });
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    return asyncRequest;
}

void MultiDeviceExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    auto priorities = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == config.end() || config.size() > 1) {
        THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str <<
            "The only config supported for the Network's SetConfig is MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES";
    } else {
        auto multiPlugin = std::dynamic_pointer_cast<MultiDeviceInferencePlugin>(this->_plugin);
        assert(multiPlugin != nullptr);
        auto metaDevices = multiPlugin->ParseMetaDevices(priorities->second, {});

        if (std::any_of(metaDevices.begin(), metaDevices.end(), [](const DeviceInformation& kvp) {
                return kvp.numRequestsPerDevices != -1;
            })) {
            THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str << "You can only change device priorities but not number of requests"
                     <<" with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES!";
        }

        {
            std::lock_guard<std::mutex> lock{_mutex};
            for (auto && device : metaDevices) {
                if (_networksPerDevice.find(device.deviceName) == _networksPerDevice.end()) {
                    THROW_IE_EXCEPTION << NOT_FOUND_str << "You can only change device priorities but not add new devices with"
                        << " the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES. "
                        << device.deviceName <<
                            " device was not in the original device list!";
                }
            }
            _devicePriorities = metaDevices;

            // update value in config
            _config[MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = priorities->second;
        }
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetConfig(const std::string &name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        THROW_IE_EXCEPTION << NOT_FOUND_str << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter MultiDeviceExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
        for (auto n : _networksPerDevice) {
            try {
                res += n.second.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const details::InferenceEngineException &iie) {
                  THROW_IE_EXCEPTION
                        << "Every device used with the Multi-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << n.first << " with error:" << iie.what();
           }
        }
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        auto it = _networksPerDevice.begin();
        IE_ASSERT(it != _networksPerDevice.end());
        IE_SET_METRIC_RETURN(NETWORK_NAME, it->second.GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported Network metric: " << name;
    }
}

}  // namespace MultiDevicePlugin
