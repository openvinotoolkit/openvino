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
    virtual IInferPtr CreateInferRequest();
    virtual IInferPtr CreateInferRequestImpl(IE::InputsDataMap networkInputs, IE::OutputsDataMap networkOutputs);
    virtual IInferPtr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                             const std::vector<std::shared_ptr<const ov::Node>>& outputs);
    virtual void release();
    virtual void init(const ScheduleContext::Ptr& context);
    virtual Pipeline GetPipeline(const IInferPtr& syncRequestImpl, WorkerInferRequest** WorkerInferRequest);
    virtual ~Schedule() = default;

protected:
    ScheduleContext::Ptr _sContext;
    SoExecNetwork        _passthroughExeNet;
};

}  // namespace MultiDevicePlugin
