// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>

#include "mkldnn/mkldnn_extension_ptr.hpp"
#include "mkldnn/mkldnn_extension.hpp"

namespace MKLDNNPlugin {

class MKLDNNExtensionManager {
public:
    using Ptr = std::shared_ptr<MKLDNNExtensionManager>;
    MKLDNNExtensionManager() = default;
    InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive* CreateExtensionPrimitive(const InferenceEngine::CNNLayerPtr& layer);
    InferenceEngine::ILayerImplFactory* CreateExtensionFactory(const InferenceEngine::CNNLayerPtr& Layer);
    void AddExtension(InferenceEngine::IExtensionPtr extension);

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace MKLDNNPlugin
