// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_core.hpp>
#include <ie_common.h>

class IECoreAdapter {
public:
    using Ptr = std::shared_ptr<IECoreAdapter>;

    IECoreAdapter(
        std::shared_ptr<InferenceEngine::Core> ieCore, std::string deviceName);

    // -----------------------------------------
    // IInferencePlugin API (deprecated). Begin.
    // - InferenceEngine::ICNNNetwork is replaced by InferenceEngine::CNNNetwork
    // -----------------------------------------

    InferenceEngine::StatusCode LoadNetwork(
        InferenceEngine::IExecutableNetwork::Ptr& ret, InferenceEngine::CNNNetwork network,
        const std::map<std::string, std::string>& config, InferenceEngine::ResponseDesc* resp) noexcept;

    InferenceEngine::StatusCode ImportNetwork(
        InferenceEngine::IExecutableNetwork::Ptr& ret, const std::string& modelFileName,
        const std::map<std::string, std::string>& config, InferenceEngine::ResponseDesc* resp) noexcept;

    // -----------------------------------------
    // IInferencePlugin API (deprecated). End.
    // -----------------------------------------

    InferenceEngine::ExecutableNetwork ImportNetwork(std::istream& networkModel,
        const std::map<std::string, std::string>& config = {});

private:
    std::shared_ptr<InferenceEngine::Core> m_ieCore;
    std::string m_deviceName;
};
