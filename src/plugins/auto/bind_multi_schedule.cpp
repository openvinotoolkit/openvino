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

void BinderMultiSchedule::init(const ScheduleContext::Ptr& sContext) {
    MultiSchedule::init(sContext);
    LOG_INFO_TAG("will enable bind buffer for MULTI!");
}

BinderMultiSchedule::~BinderMultiSchedule() {
}

Pipeline BinderMultiSchedule::GetPipeline(const IInferPtr& syncInferRequest, WorkerInferRequest** workerInferRequest) {
    Pipeline pipeline;
    struct RequestExecutor : ITaskExecutor {
        explicit RequestExecutor(InferenceEngine::SoIInferRequestInternal& inferRequest) : _inferRequest(inferRequest) {
            _inferRequest->SetCallback([this](std::exception_ptr exceptionPtr) mutable {
                _exceptionPtr = exceptionPtr;
                auto capturedTask = std::move(_task);
                capturedTask();
            });
        }
        void run(InferenceEngine::Task task) override {
            _task = std::move(task);
            _inferRequest->StartAsync();
        };
        InferenceEngine::SoIInferRequestInternal& _inferRequest;
        std::exception_ptr _exceptionPtr;
        InferenceEngine::Task _task;
    };
    auto requestExecutor =
        std::make_shared<RequestExecutor>(std::static_pointer_cast<MultiDeviceInferRequest>(syncInferRequest)->GetSharedRequest());
    pipeline.emplace_back(requestExecutor, [requestExecutor] {
        if (nullptr != requestExecutor->_exceptionPtr) {
            std::rethrow_exception(requestExecutor->_exceptionPtr);
        }
    });
    return pipeline;
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

