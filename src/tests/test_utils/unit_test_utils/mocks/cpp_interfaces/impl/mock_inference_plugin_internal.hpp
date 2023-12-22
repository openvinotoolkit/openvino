// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <cpp_interfaces/interface/ie_iplugin_internal.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ie_icore.hpp"
#include "openvino/runtime/iplugin.hpp"

class MockInferencePluginInternal : public InferenceEngine::IInferencePlugin {
public:
    MOCK_METHOD2(
        LoadExeNetworkImpl,
        std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>(const InferenceEngine::CNNNetwork&,
                                                                     const std::map<std::string, std::string>&));
    MOCK_METHOD1(AddExtension, void(const std::shared_ptr<InferenceEngine::IExtension>&));
    MOCK_METHOD1(SetConfig, void(const std::map<std::string, std::string>&));

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& stream,
        const std::map<std::string, std::string>&) {
        return {};
    }

    std::string importedString;
};