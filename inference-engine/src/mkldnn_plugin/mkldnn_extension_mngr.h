// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <ie_iextension.h>
#include <ie_layers.h>

namespace MKLDNNPlugin {

class MKLDNNExtensionManager {
public:
    using Ptr = std::shared_ptr<MKLDNNExtensionManager>;
    MKLDNNExtensionManager() = default;
    InferenceEngine::ILayerImpl::Ptr CreateImplementation(const std::shared_ptr<ngraph::Node>& op);
    IE_SUPPRESS_DEPRECATED_START
    std::shared_ptr<InferenceEngine::ILayerImplFactory> CreateExtensionFactory(const InferenceEngine::CNNLayerPtr& Layer);
    InferenceEngine::IShapeInferImpl::Ptr CreateReshaper(const InferenceEngine::CNNLayerPtr& Layer);
    IE_SUPPRESS_DEPRECATED_END
    void AddExtension(InferenceEngine::IExtensionPtr extension);

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace MKLDNNPlugin
