// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "executable_network.hpp"
#include "multi_schedule.hpp"
#include "bind_multi_schedule.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class MultiExecutableNetwork : public ExecutableNetwork {
    friend IInferPtr MultiSchedule::CreateInferRequest();
public:
    using Ptr = std::shared_ptr<MultiExecutableNetwork>;

    explicit MultiExecutableNetwork(MultiScheduleContext::Ptr& context, const MultiSchedule::Ptr& schedule);

    void SetConfig(const std::map<std::string, IE::Parameter>& config) override;
    IE::Parameter GetConfig(const std::string& name) const override;
    IE::Parameter GetMetric(const std::string& name) const override;
    std::shared_ptr<IE::RemoteContext> GetContext() const override;
    ~MultiExecutableNetwork() override;

private:
    MultiScheduleContext::Ptr _multiSContext;
};

}  // namespace MultiDevicePlugin
