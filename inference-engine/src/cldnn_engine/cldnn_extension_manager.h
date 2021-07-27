// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <ie_iextension.h>

namespace CLDNNPlugin {

class GPUExtensionManager {
public:
    using Ptr = std::shared_ptr<GPUExtensionManager>;
    GPUExtensionManager() = default;
    InferenceEngine::ILayerImpl::Ptr CreateImplementation(const std::shared_ptr<ngraph::Node>& op);
    void AddExtension(const InferenceEngine::IExtensionPtr& extension);
private:
    bool IsSupportedImplType(std::string type);
    const std::vector<std::string> supportedImplTypes = {"GPU"};
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace CLDNNPlugin
