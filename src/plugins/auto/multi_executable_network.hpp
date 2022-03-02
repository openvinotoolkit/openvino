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
#include "multi_schedule.hpp"
#include "common.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class MultiExecutableNetwork : public BaseExecutableNetwork {
    friend InferenceEngine::IInferRequestInternal::Ptr MultiSchedule::CreateInferRequest();
    public:
        using Ptr = std::shared_ptr<MultiExecutableNetwork>;

        explicit MultiExecutableNetwork(MultiContext::Ptr& context,
               const MultiSchedule::Ptr& schedule);

        void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
        InferenceEngine::Parameter GetConfig(const std::string &name) const override;
        InferenceEngine::Parameter GetMetric(const std::string &name) const override;
        std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
        ~MultiExecutableNetwork() override;

    private:
        MultiContext::Ptr                                           _multiContext;
};

}  // namespace MultiDevicePlugin
