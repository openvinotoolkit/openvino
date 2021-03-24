// Copyright (C) 2018-2021 Intel Corporation
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

    InferenceEngine::ExecutableNetwork LoadNetwork(const InferenceEngine::CNNNetwork & network,
        const std::map<std::string, std::string>& config = {});

    InferenceEngine::ExecutableNetwork ImportNetwork(const std::string& modelFileName,
        const std::map<std::string, std::string>& config = {});

    InferenceEngine::ExecutableNetwork ImportNetwork(std::istream& networkModel,
        const std::map<std::string, std::string>& config = {});

    std::shared_ptr<InferenceEngine::Core>& ieCore() {
        return m_ieCore;
    }

private:
    std::shared_ptr<InferenceEngine::Core> m_ieCore;
    std::string m_deviceName;
};
