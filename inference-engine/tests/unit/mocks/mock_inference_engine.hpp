// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include <gmock/gmock.h>

class MockInferenceEngine : public InferenceEngine :: IInferencePlugin
{
public:
    MOCK_QUALIFIED_METHOD2(AddExtension, noexcept, InferenceEngine::StatusCode(InferenceEngine::IExtensionPtr,
                                                                               InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD1(GetVersion, noexcept, void (const InferenceEngine::Version *&));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD1(SetLogCallback, noexcept, void (InferenceEngine::IErrorListener &));
    MOCK_QUALIFIED_METHOD2(LoadNetwork, noexcept, InferenceEngine::StatusCode(InferenceEngine::ICNNNetwork &, InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD4(LoadNetwork, noexcept, InferenceEngine::StatusCode(
            InferenceEngine::IExecutableNetwork::Ptr &, InferenceEngine::ICNNNetwork &, const std::map<std::string, std::string> &, InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD0(Unload, noexcept, void ());
    MOCK_QUALIFIED_METHOD4(ImportNetwork, noexcept, InferenceEngine::StatusCode(
            InferenceEngine::IExecutableNetwork::Ptr &, const std::string &,
            const std::map<std::string, std::string> &, InferenceEngine::ResponseDesc *));
    MOCK_QUALIFIED_METHOD3(Infer, noexcept,
                           InferenceEngine::StatusCode(
                               const InferenceEngine::Blob &,
                               InferenceEngine::Blob &,
                               InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD3(Infer, noexcept,
                           InferenceEngine::StatusCode(
                               const InferenceEngine::BlobMap&,
                               std::map<std::string, InferenceEngine::Blob::Ptr> &,
                               InferenceEngine::ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD2(GetPerformanceCounts,
                           const noexcept,
                           InferenceEngine::StatusCode(
                               std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> & ,
                               InferenceEngine::ResponseDesc * resp));
    MOCK_QUALIFIED_METHOD2(SetConfig, noexcept, InferenceEngine::StatusCode(const std::map<std::string, std::string> &,
                                                                            InferenceEngine::ResponseDesc *resp));
};

