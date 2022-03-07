// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include "threading/ie_thread_safe_containers.hpp"
#include "threading/ie_itask_executor.hpp"
#include "threading/ie_executor_manager.hpp"
#include "ie_icore.hpp"
#include <ie_performance_hints.hpp>
#include "openvino/runtime/properties.hpp"
#include "base_executable_network.hpp"
#include "auto_schedule.hpp"
#include "common.hpp"
#include <ie_performance_hints.hpp>

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class AutoExecutableNetwork : public BaseExecutableNetwork {
    friend IInferPtr AutoSchedule::CreateInferRequest();
    public:
        using Ptr = std::shared_ptr<AutoExecutableNetwork>;

        explicit AutoExecutableNetwork(AutoContext::Ptr& context,
               const AutoSchedule::Ptr& schedule);

        void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
        InferenceEngine::Parameter GetConfig(const std::string &name) const override;
        InferenceEngine::Parameter GetMetric(const std::string &name) const override;
        std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
        virtual ~AutoExecutableNetwork() = default;

    private:
        AutoContext::Ptr _autoContext;
        AutoSchedule::Ptr _autoSchedule;
};
}  // namespace MultiDevicePlugin
