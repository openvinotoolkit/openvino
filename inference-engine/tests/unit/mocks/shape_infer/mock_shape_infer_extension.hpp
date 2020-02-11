// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include "ie_extension.h"
#include <gmock/gmock.h>

class MockShapeInferExtension : public InferenceEngine::IShapeInferExtension {
 public:
    virtual ~MockShapeInferExtension() = default;

    using Ptr = std::shared_ptr<MockShapeInferExtension>;
    MOCK_QUALIFIED_METHOD1(GetVersion, const noexcept, void (const InferenceEngine::Version *&));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD1(SetLogCallback, noexcept, void (InferenceEngine::IErrorListener &));
    MOCK_QUALIFIED_METHOD0(Unload, noexcept, void ());

    MOCK_QUALIFIED_METHOD3(getShapeInferTypes, noexcept, InferenceEngine::StatusCode
                            (char**&, unsigned int&, InferenceEngine::ResponseDesc *resp));

    MOCK_QUALIFIED_METHOD3(getShapeInferImpl, noexcept, InferenceEngine::StatusCode
            (InferenceEngine::IShapeInferImpl::Ptr&, const char* type, InferenceEngine::ResponseDesc *resp));
};

