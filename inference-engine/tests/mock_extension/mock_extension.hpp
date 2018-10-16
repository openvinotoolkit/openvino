// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <inference_engine.hpp>
#include "mkldnn/mkldnn_extension.hpp"

using MKLDNExtension = InferenceEngine::MKLDNNPlugin::IMKLDNNExtension;
using GenericPrimitive = InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive;

class MockMKLDNNExtensionShared : public MKLDNExtension {
    MKLDNExtension * _target = nullptr;
public:
    MockMKLDNNExtensionShared(MKLDNExtension *);

    void GetVersion(const InferenceEngine::Version *& versionInfo) const noexcept override;

    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override;

    void Unload() noexcept override;

    InferenceEngine::StatusCode CreateGenericPrimitive(GenericPrimitive*& primitive,
                                                       const InferenceEngine::CNNLayerPtr& layer,
                                                       InferenceEngine::ResponseDesc *resp) const noexcept override;

    void Release () noexcept override {
        delete this;
    };
};


