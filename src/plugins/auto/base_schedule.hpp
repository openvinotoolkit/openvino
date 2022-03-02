// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "common.hpp"
#include "threading/ie_immediate_executor.hpp"
#include "threading/ie_istreams_executor.hpp"
#include "threading/ie_itask_executor.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
using Stage = std::pair<InferenceEngine::ITaskExecutor::Ptr, InferenceEngine::Task>;
using Pipeline = std::vector<Stage>;
class Schedule : public std::enable_shared_from_this<Schedule>  {
public:
    using Ptr = std::shared_ptr<Schedule>;
    virtual IInferPtr CreateInferRequest();
    virtual IInferPtr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
            InferenceEngine::OutputsDataMap networkOutputs);
    virtual IInferPtr CreateInferRequestImpl(
            const std::vector<std::shared_ptr<const ov::Node>>& inputs,
            const std::vector<std::shared_ptr<const ov::Node>>& outputs);
    virtual void release();
    virtual void init(const Context::Ptr& context);
    virtual Pipeline GetPipeline(const IInferPtr& syncRequestImpl,
            WorkerInferRequest** WorkerInferRequest);
    virtual ~Schedule() = default;

protected:
    Context::Ptr _context;
};

}  // namespace MultiDevicePlugin
