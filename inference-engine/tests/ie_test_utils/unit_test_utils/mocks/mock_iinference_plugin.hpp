// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "ie_plugin.hpp"
#include <gmock/gmock.h>

IE_SUPPRESS_DEPRECATED_START
class MockIInferencePlugin : public InferenceEngine :: IInferencePlugin {
public:
    MOCK_QUALIFIED_METHOD2(AddExtension, noexcept, InferenceEngine::StatusCode(InferenceEngine::IExtensionPtr,
                                                                               InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD1(GetVersion, noexcept, void(const InferenceEngine::Version *&));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void());
    MOCK_QUALIFIED_METHOD2(LoadNetwork, noexcept, InferenceEngine::StatusCode(
            const InferenceEngine::ICNNNetwork &, InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD4(LoadNetwork, noexcept, InferenceEngine::StatusCode(
            InferenceEngine::IExecutableNetwork::Ptr &,
            const InferenceEngine::ICNNNetwork &,
            const std::map<std::string, std::string> &,
            InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD4(ImportNetwork, noexcept, InferenceEngine::StatusCode(
            InferenceEngine::IExecutableNetwork::Ptr &,
            const std::string &,
            const std::map<std::string, std::string> &,
            InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(SetConfig, noexcept, InferenceEngine::StatusCode(const std::map<std::string, std::string> &,
                                                                            InferenceEngine::ResponseDesc *resp));
};
IE_SUPPRESS_DEPRECATED_END