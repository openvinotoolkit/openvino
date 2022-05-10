// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "base_async_infer_request.hpp"
#include "plugin.hpp"
#include "multi_schedule.hpp"
#include "multi_executable_network.hpp"
// ------------------------------MultiSchedule----------------------------
namespace MultiDevicePlugin {

thread_local WorkerInferRequest* MultiSchedule::_thisWorkerInferRequest =
    nullptr;
// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* MultiSchedule::_thisPreferredDeviceName = "";

void MultiSchedule::init(const ScheduleContext::Ptr& sContext) {
    Schedule::init(sContext);
    _multiSContext = std::dynamic_pointer_cast<MultiScheduleContext>(_sContext);
    for (auto&& networkValue : _multiSContext->_networksPerDevice) {
        auto& device  = networkValue.first;
        auto& network = networkValue.second;
        GenerateWorkers(device, network);
    }
}

Pipeline MultiSchedule::GetPipeline(const IInferPtr& syncInferRequest,
    WorkerInferRequest** workerInferRequest) {
    Pipeline pipeline = {
        // if the request is coming with device-specific remote blobs make sure it is scheduled to the specific device only:
        Stage {
            /*TaskExecutor*/ std::make_shared<IE::ImmediateExecutor>(),
            /*task*/ [this, &syncInferRequest]() {
                // by default, no preferred device:
                _thisPreferredDeviceName = "";
                auto execNetwork = _multiSContext->_executableNetwork.lock();
                // if any input is remote (e.g. was set with SetBlob), let' use the corresponding device
                for (const auto& it : execNetwork->GetInputsInfo()) {
                    auto b = syncInferRequest->GetBlob(it.first);
                    auto r = b->as<IE::RemoteBlob>();
                    if (r) {
                        const auto name = r->getDeviceName();
                        const auto res = std::find_if(
                            _multiSContext->_devicePrioritiesInitial.cbegin(),
                            _multiSContext->_devicePrioritiesInitial.cend(),
                        [&name](const MultiDevicePlugin::DeviceInformation & d) {
                            return (d.defaultDeviceID.empty() ? d.deviceName : (d.deviceName + "." +
                                    d.defaultDeviceID)) == name;
                        });
                        if (_multiSContext->_devicePrioritiesInitial.cend() == res) {
                            IE_THROW() <<
                                "None of the devices (for which current MULTI-device configuration was "
                                "initialized) supports a remote blob created on the device named " << name;
                        } else {
                            // it is ok to take the c_str() here (as pointed in the executable_network.hpp we need to use const char*)
                            // as the original strings are from the "persistent" vector (with the right lifetime)
                            _thisPreferredDeviceName = res->deviceName.c_str();
                            break;
                        }
                    }
                }
            }},
        // as the scheduling algo may select any device, this stage accepts the scheduling decision (actual workerRequest)
        // then sets the device-agnostic blobs to the actual (device-specific) request
        Stage {
            /*TaskExecutor*/std::dynamic_pointer_cast<IE::ITaskExecutor>(shared_from_this()),
            /*task*/ [this, &syncInferRequest, workerInferRequest]() {
                *workerInferRequest = _thisWorkerInferRequest;
                auto multiSyncInferRequest = std::dynamic_pointer_cast<MultiDeviceInferRequest>
                (syncInferRequest);
                multiSyncInferRequest->SetBlobsToAnotherRequest(
                    _thisWorkerInferRequest->_inferRequest);
            }},
        // final task in the pipeline:
        Stage {
            /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(workerInferRequest),
            /*task*/ [this, &syncInferRequest, workerInferRequest]() {
                if (nullptr != (*workerInferRequest)->_exceptionPtr) {
                    std::rethrow_exception((*workerInferRequest)->_exceptionPtr);
                }
                if (_multiSContext->_needPerfCounters) {
                    auto multiSyncInferRequest = std::dynamic_pointer_cast<MultiDeviceInferRequest>
                        (syncInferRequest);
                    multiSyncInferRequest->_perfMap =
                        (*workerInferRequest)->_inferRequest->GetPerformanceCounts();
                }
                (*workerInferRequest)->_inferCount++;
            }}
    };
    return pipeline;
}

void MultiSchedule::GenerateWorkers(const std::string& device,
    const SoExecNetwork& executableNetwork) {
    auto itNumRequests = std::find_if(_multiSContext->_devicePriorities.cbegin(),
            _multiSContext->_devicePriorities.cend(),
    [&device](const DeviceInformation & d) {
        return d.deviceName == device;
    });
    unsigned int optimalNum = 0;
    try {
        optimalNum = executableNetwork->GetMetric(METRIC_KEY(
                    OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const IE::Exception& iie) {
        IE_THROW()
                << "Every device used with the Multi-Device should "
                    << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                    << "Failed to query the metric for the " << device << " with error:" <<
                    iie.what();
    }
    const auto numRequests = (_multiSContext->_devicePriorities.end() ==
            itNumRequests ||
            itNumRequests->numRequestsPerDevices == -1) ? optimalNum :
        itNumRequests->numRequestsPerDevices;
    auto& workerRequests = _workerRequests[device];
    auto& idleWorkerRequests = _idleWorkerRequests[device];
    workerRequests.resize(numRequests);
    _inferPipelineTasksDeviceSpecific[device] =
        std::unique_ptr<IE::ThreadSafeQueue<IE::Task>>(new
            IE::ThreadSafeQueue<IE::Task>);
    auto* idleWorkerRequestsPtr = &(idleWorkerRequests);
    idleWorkerRequests.set_capacity(numRequests);
    int num = 0;
    for (auto&& workerRequest : workerRequests) {
        workerRequest._inferRequest = {executableNetwork->CreateInferRequest(), executableNetwork._so};
        auto* workerRequestPtr = &workerRequest;
        workerRequestPtr->_index = num++;
        IE_ASSERT(idleWorkerRequests.try_push(std::make_pair(workerRequestPtr->_index,
                    workerRequestPtr)) == true);
        workerRequest._inferRequest->SetCallback(
            [workerRequestPtr, this, device,
                          idleWorkerRequestsPtr](std::exception_ptr exceptionPtr) mutable {
            IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
            workerRequestPtr->_exceptionPtr = exceptionPtr;
            {
                auto capturedTask = std::move(workerRequestPtr->_task);
                capturedTask();
            }
            // try to return the request to the idle list (fails if the overall object destruction has began)
            if (idleGuard.Release()->try_push(std::make_pair(workerRequestPtr->_index, workerRequestPtr))) {
                // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                IE::Task t;
                if (_inferPipelineTasks.try_pop(t)) {
                    ScheduleToWorkerInferRequest(std::move(t));
                } else if (_inferPipelineTasksDeviceSpecific[device]->try_pop(t)) {
                    ScheduleToWorkerInferRequest(std::move(t), device);
                }
            }
        });
    }
}

void MultiSchedule::ScheduleToWorkerInferRequest(IE::Task inferPipelineTask,
    DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    devices = [&] {
        std::lock_guard<std::mutex> lock(_multiSContext->_mutex);
        return _multiSContext->_devicePriorities;
    }();
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device)) {
            continue;
        }
        if (RunPipelineTask(inferPipelineTask, _idleWorkerRequests[device.deviceName],
                preferred_device)) {
            return;
        }
    }
    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty()) {
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(
                inferPipelineTask));
    } else {
        _inferPipelineTasks.push(std::move(inferPipelineTask));
    }
}

bool MultiSchedule::RunPipelineTask(IE::Task& inferPipelineTask,
    NotBusyWorkerRequests& idleWorkerRequests,
    const DeviceName& preferred_device) {
    WorkerInferRequest* workerRequestPtr = nullptr;
    std::pair<int, WorkerInferRequest*> worker;
    if (idleWorkerRequests.try_pop(worker)) {
        workerRequestPtr = worker.second;
        IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
        _thisWorkerInferRequest = workerRequestPtr;
        {
            auto capturedTask = std::move(inferPipelineTask);
            capturedTask();
        }
        idleGuard.Release();
        return true;
    }
    return false;
}

void MultiSchedule::run(IE::Task inferPipelineTask) {
    ScheduleToWorkerInferRequest(std::move(inferPipelineTask),
        _thisPreferredDeviceName);
}

MultiSchedule::~MultiSchedule() {
    {
        std::lock_guard<std::mutex> lock(_multiSContext->_mutex);
        _multiSContext->_devicePriorities.clear();
    }
    /* NOTE: The only threads that use `MultiSchedule` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& idleWorker : _idleWorkerRequests) {
        // stop accepting any idle requests back (for re-scheduling)
        idleWorker.second.set_capacity(0);
    }
    for (auto&& _workerRequest : _workerRequests) {
        unsigned int count = 0;
        for (auto& request : _workerRequest.second) {
            count += request._inferCount;
        }
        LOG_INFO("[AUTOPLUGIN]%s:infer:%ld", _workerRequest.first.c_str(), count);
    }
    _workerRequests.clear();
}

IE::IInferRequestInternal::Ptr MultiSchedule::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    SoInfer request_to_share_blobs_with;
    IE::RemoteContext::Ptr ctx = nullptr;
    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _multiSContext->_devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs,
            request_to_share_blobs_with);
}

IInferPtr
MultiSchedule::CreateInferRequestImpl(IE::InputsDataMap networkInputs,
    IE::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    SoInfer request_to_share_blobs_with;
    IE::RemoteContext::Ptr ctx = nullptr;
    // borrowing device-specific blobs from the underlying requests for the device-agnostic, user-facing requests
    // this allows to potentially save on the data-copy later (if the requests are scheduled in the same order)
    for (const auto& device : _multiSContext->_devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            break;
        }
        sum += dev_requests.size();
    }
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs,
            request_to_share_blobs_with);
}

IInferPtr MultiSchedule::CreateInferRequest() {
    IInferPtr syncRequestImpl;
    auto execNetwork = std::dynamic_pointer_cast<MultiExecutableNetwork>(
            _multiSContext->_executableNetwork.lock());
    if (_multiSContext->_core && _multiSContext->_core->isNewAPI())
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_parameters,
                execNetwork->_results);
    if (!syncRequestImpl)
        syncRequestImpl = CreateInferRequestImpl(execNetwork->_networkInputs,
                execNetwork->_networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(execNetwork);
    return std::make_shared<BaseAsyncInferRequest>(shared_from_this(),
            syncRequestImpl,
            execNetwork->_callbackExecutor);
}

}  // namespace MultiDevicePlugin

