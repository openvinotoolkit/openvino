// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "async_infer_request.hpp"
#include "plugin.hpp"
#include "bind_multi_schedule.hpp"
#include "multi_executable_network.hpp"
// ------------------------------MultiSchedule----------------------------
namespace MultiDevicePlugin {

thread_local IE::IInferRequestInternal* BinderMultiSchedule::_sharedRequest = nullptr;

void BinderMultiSchedule::init(const ScheduleContext::Ptr& sContext) {
     MultiSchedule::init(sContext);
}

Pipeline BinderMultiSchedule::GetPipeline(const IInferPtr& syncInferRequest, WorkerInferRequest** workerInferRequest) {
    Pipeline pipeline = {
        // if the request is coming with device-specific remote blobs make sure it is scheduled to the specific device only:
        Stage {
            /*TaskExecutor*/ std::make_shared<IE::ImmediateExecutor>(), /*task*/ [this, &syncInferRequest, workerInferRequest]() {
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
                _thisWorkerInferRequest = *workerInferRequest;
                _sharedRequest = std::dynamic_pointer_cast<MultiDeviceInferRequest>(syncInferRequest)->GetSharedRequest()._ptr.get();
            }},
        // as the scheduling algo may select any device, this stage accepts the scheduling decision (actual workerRequest)
        // then sets the device-agnostic blobs to the actual (device-specific) request
        Stage {
            /*TaskExecutor*/std::dynamic_pointer_cast<IE::ITaskExecutor>(shared_from_this()), /*task*/ [this, &syncInferRequest, workerInferRequest]() {
                *workerInferRequest = _thisWorkerInferRequest;
                auto multiSyncInferRequest = std::dynamic_pointer_cast<MultiDeviceInferRequest>(syncInferRequest);
                multiSyncInferRequest->SetBlobsToAnotherRequest(_thisWorkerInferRequest->_inferRequest);
                INFO_RUN([workerInferRequest]() {
                    (*workerInferRequest)->_startTimes.push_back(std::move(std::chrono::steady_clock::now()));
                });
            }},
        // final task in the pipeline:
        Stage {
            /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(workerInferRequest), /*task*/ [this, &syncInferRequest, workerInferRequest]() {
                if (nullptr != (*workerInferRequest)->_exceptionPtr) {
                    std::rethrow_exception((*workerInferRequest)->_exceptionPtr);
                }
                if (_multiSContext->_needPerfCounters) {
                    auto multiSyncInferRequest = std::dynamic_pointer_cast<MultiDeviceInferRequest>
                        (syncInferRequest);
                    multiSyncInferRequest->_scheduledRequest =
                        (*workerInferRequest)->_inferRequest;
                }
                INFO_RUN([workerInferRequest]() {
                   (*workerInferRequest)->_endTimes.push_back(std::move(std::chrono::steady_clock::now()));
                });
            }}
    };
    return pipeline;
}

bool BinderMultiSchedule::ScheduleToWorkerInferRequest(IE::Task inferPipelineTask, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    devices = [&] {
        std::lock_guard<std::mutex> lock(_multiSContext->_mutex);
        return _multiSContext->_devicePriorities;
    }();
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.deviceName != preferred_device)) {
            continue;
        }
        if (RunPipelineTask(inferPipelineTask, _idleWorkerRequests[device.deviceName], preferred_device)) {
            return true;
        }
    }
    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty()) {
        _inferPipelineTasksDeviceSpecific[preferred_device]->push(std::move(inferPipelineTask));
    } else {
        _inferPipelineTasks.push(std::move(inferPipelineTask));
    }
    return false;
}

bool BinderMultiSchedule::RunPipelineTask(IE::Task& inferPipelineTask,
    NotBusyWorkerRequests& idleWorkerRequests,
    const DeviceName& preferred_device) {
    WorkerInferRequest* workerRequestPtr = nullptr;
    WorkerInferRequest* headWorker = nullptr;
    bool flag = false;
    while (idleWorkerRequests.try_pop(workerRequestPtr)) {
        if (flag && workerRequestPtr == headWorker)
            break;
        if (!flag) {
            headWorker = workerRequestPtr;
            flag = true;
        }
        IdleGuard<NotBusyWorkerRequests> idleGuard{workerRequestPtr, idleWorkerRequests};
        if (_sharedRequest == workerRequestPtr->_inferRequest._ptr.get()) {
            _thisWorkerInferRequest = workerRequestPtr;
            {
                auto capturedTask = std::move(inferPipelineTask);
                capturedTask();
            }
            idleGuard.Release();
            return true;
        }
    }
    return false;
}

void BinderMultiSchedule::run(IE::Task inferPipelineTask) {
    if (_thisWorkerInferRequest) {
        auto capturedTask = std::move(inferPipelineTask);
        capturedTask();
    } else {
        ScheduleToWorkerInferRequest(std::move(inferPipelineTask), _thisPreferredDeviceName);
    }
}

BinderMultiSchedule::~BinderMultiSchedule() {
}

IInferPtr BinderMultiSchedule::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    auto num = _numRequestsCreated++;
    size_t sum = 0;
    SoInfer request_to_share_blobs_with;
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
    if (!request_to_share_blobs_with) {
        IE_THROW() <<
                    "binder mode does not allow oversubsciption of infer requests"
                    " please use optimal infer request";
    }
    auto syncImpl = std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
    return syncImpl;
}

IInferPtr BinderMultiSchedule::CreateInferRequestImpl(IE::InputsDataMap networkInputs,
    IE::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    SoInfer request_to_share_blobs_with;
    size_t sum = 0;
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
    if (!request_to_share_blobs_with) {
        IE_THROW() <<
                    "binder mode does not allow oversubsciption of infer requests"
                    " please use optimal infer request";
    }
    auto syncImpl = std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
    return syncImpl;
}

}  // namespace MultiDevicePlugin

