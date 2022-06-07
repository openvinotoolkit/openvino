// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "common.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
using Stage = std::pair<IE::ITaskExecutor::Ptr, IE::Task>;
using Pipeline = std::vector<Stage>;
class Schedule : public std::enable_shared_from_this<Schedule>  {
public:
    using Ptr = std::shared_ptr<Schedule>;
    virtual IInferPtr CreateInferRequest() = 0;
    virtual IInferPtr CreateInferRequestImpl(IE::InputsDataMap networkInputs, IE::OutputsDataMap networkOutputs) = 0;
    virtual IInferPtr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs) = 0;
    //virtual void release() = 0;
    virtual void init(const ScheduleContext::Ptr& context) = 0;
    virtual Pipeline GetPipeline(const IInferPtr& syncRequestImpl, WorkerInferRequest** WorkerInferRequest) = 0;
    virtual ~Schedule() = default;
};

}  // namespace MultiDevicePlugin
