// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class ExecutableNetwork : public
    InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<ExecutableNetwork>;
    ExecutableNetwork(const Schedule::Ptr& schedule, const ScheduleContext::Ptr& sContext);
    IInferPtr CreateInferRequest() override;
    IInferPtr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                     InferenceEngine::OutputsDataMap networkOutputs) override;
    IInferPtr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                     const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
    ~ExecutableNetwork() override;

private:
    Schedule::Ptr        _schedule;
    ScheduleContext::Ptr _sContext;
    std::once_flag       _oc;
    void SetExeNetworkForContext();
};
}  // namespace MultiDevicePlugin
