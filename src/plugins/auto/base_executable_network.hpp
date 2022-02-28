// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class BaseExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
    public:
        using Ptr = std::shared_ptr<BaseExecutableNetwork>;
        BaseExecutableNetwork(Schedule::Ptr schedule) ;

        virtual void SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) override;
        virtual InferenceEngine::Parameter GetConfig(const std::string &name) const override;
        virtual InferenceEngine::Parameter GetMetric(const std::string &name) const override;
        virtual void run(InferenceEngine::Task inferTask) override;
        virtual InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
        virtual InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                InferenceEngine::OutputsDataMap networkOutputs) override;
        virtual InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;
        virtual std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;
        ~ExecutableNetwork() override;
    private :
        Schedule::Ptr _schedule;
        InferenceEngine::SoExecutableNetworkInternal _executableNetwork;
};

}  // namespace MultiDevicePlugin
