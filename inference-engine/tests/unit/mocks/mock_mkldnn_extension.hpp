// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "inference_engine.hpp"
#include <gmock/gmock.h>
#include "mkldnn/mkldnn_extension.hpp"

using MKLDNExtension = InferenceEngine::MKLDNNPlugin::IMKLDNNExtension;
using GenericPrimitive = InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive;

class MockMKLDNNExtension : public MKLDNExtension {
 public:
    MOCK_QUALIFIED_METHOD1(GetVersion, const noexcept, void (const InferenceEngine::Version *&));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD1(SetLogCallback, noexcept, void (InferenceEngine::IErrorListener &));
    MOCK_QUALIFIED_METHOD0(Unload, noexcept, void ());

    MOCK_QUALIFIED_METHOD3(CreateGenericPrimitive, const noexcept, InferenceEngine::StatusCode
                            (GenericPrimitive*& primitive,
                            const InferenceEngine::CNNLayerPtr& layer,
                            InferenceEngine::ResponseDesc *resp));
};

