// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <ie_iextension.h>

namespace MKLDNNPlugin {

class MKLDNNExtensionManager {
public:
    using Ptr = std::shared_ptr<MKLDNNExtensionManager>;
    MKLDNNExtensionManager() = default;
    InferenceEngine::ILayerImplFactory* CreateExtensionFactory(const InferenceEngine::CNNLayerPtr& Layer);
    InferenceEngine::IShapeInferImpl::Ptr CreateReshaper(const InferenceEngine::CNNLayerPtr& Layer);
    void AddExtension(InferenceEngine::IExtensionPtr extension);

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}  // namespace MKLDNNPlugin
