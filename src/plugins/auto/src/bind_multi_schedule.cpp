// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "async_infer_request.hpp"
#include "plugin.hpp"
#include "bind_multi_schedule.hpp"
// ------------------------------MultiSchedule----------------------------
namespace MultiDevicePlugin {

void BinderMultiSchedule::init(const ScheduleContext::Ptr& sContext) {
    AutoSchedule::init(sContext);
    LOG_INFO_TAG("enable bind buffer for AUTO");
}

Pipeline BinderMultiSchedule::GetPipeline(const IInferPtr& syncInferRequest, WorkerInferRequest** workerInferRequest) {
    Pipeline pipeline;
    struct RequestExecutor : ITaskExecutor {
        explicit RequestExecutor(InferenceEngine::SoIInferRequestInternal& inferRequest,
                                 WorkerInferRequest* workInferReq)
            : _inferRequest(inferRequest),
              _workInferReq(workInferReq) {
            _inferRequest->SetCallback([this](std::exception_ptr exceptionPtr) mutable {
                _exceptionPtr = exceptionPtr;
                auto capturedTask = std::move(_task);
                capturedTask();
                INFO_RUN([&]() {
                    if (_workInferReq) {
                        _workInferReq->_endTimes.push_back(std::chrono::steady_clock::now());
                    }
                });
            });
        }
        void run(InferenceEngine::Task task) override {
            _task = std::move(task);
            INFO_RUN([&]() {
                if (_workInferReq) {
                    _workInferReq->_startTimes.push_back(std::chrono::steady_clock::now());
                }
            });
            _inferRequest->StartAsync();
        };
        InferenceEngine::SoIInferRequestInternal& _inferRequest;
        std::exception_ptr _exceptionPtr;
        InferenceEngine::Task _task;
        WorkerInferRequest* _workInferReq;
    };
    auto &soInferReq =
        std::static_pointer_cast<MultiDeviceInferRequest>(syncInferRequest)->GetSharedRequest();
    WorkerInferRequest* workInferReq = nullptr;
    INFO_RUN([&]() {
        std::lock_guard<std::mutex> lock(m_dev_infer_mutex);
        auto iter = m_dev_infer.find(soInferReq._ptr);
        if (iter != m_dev_infer.end()) {
            workInferReq = iter->second;
        }
    });
    auto requestExecutor = std::make_shared<RequestExecutor>(soInferReq, workInferReq);
    pipeline.emplace_back(requestExecutor, [requestExecutor] {
        if (nullptr != requestExecutor->_exceptionPtr) {
            std::rethrow_exception(requestExecutor->_exceptionPtr);
        }
    });
    return pipeline;
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
    for (const auto& device : _autoSContext->_devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            INFO_RUN([&]() {
                std::lock_guard<std::mutex> lock(m_dev_infer_mutex);
                m_dev_infer.insert(std::make_pair(request_to_share_blobs_with._ptr, &dev_requests.at(num - sum)));
            });
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
    for (const auto& device : _autoSContext->_devicePrioritiesInitial) {
        auto& dev_requests = _workerRequests[device.deviceName];
        if ((num - sum) < dev_requests.size()) {
            request_to_share_blobs_with = dev_requests.at(num - sum)._inferRequest;
            INFO_RUN([&]() {
                std::lock_guard<std::mutex> lock(m_dev_infer_mutex);
                m_dev_infer.insert(std::make_pair(request_to_share_blobs_with._ptr, &dev_requests.at(num - sum)));
            });
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

