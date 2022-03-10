// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "base_schedule.hpp"

// ------------------------------Schedule----------------------------
namespace MultiDevicePlugin {

IInferPtr Schedule::CreateInferRequest() {
    return nullptr;
}
IInferPtr Schedule::CreateInferRequestImpl(IE::InputsDataMap networkInputs,
    IE::OutputsDataMap networkOutputs) {
    return nullptr;
}
IInferPtr Schedule::CreateInferRequestImpl(const
    std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    return nullptr;
}
void Schedule::release() {
}

void Schedule::init(const ScheduleContext::Ptr& context) {
    _sContext = context;
}

Pipeline Schedule::GetPipeline(const IInferPtr& syncRequestImpl,
    WorkerInferRequest** WorkerInferRequest) {
    return {};
}
}  // namespace MultiDevicePlugin
