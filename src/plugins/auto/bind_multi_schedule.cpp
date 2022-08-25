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

