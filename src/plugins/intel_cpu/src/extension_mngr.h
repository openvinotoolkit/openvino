// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <memory>
#include <ie_iextension.h>

namespace ov {
namespace intel_cpu {

class ExtensionManager {
public:
    using Ptr = std::shared_ptr<ExtensionManager>;
    ExtensionManager() = default;
    InferenceEngine::ILayerImpl::Ptr CreateImplementation(const std::shared_ptr<ov::Node>& op);
    void AddExtension(const InferenceEngine::IExtensionPtr& extension);
    const std::vector<InferenceEngine::IExtensionPtr> & Extensions() const;

private:
    std::vector<InferenceEngine::IExtensionPtr> _extensions;
};

}   // namespace intel_cpu
}   // namespace ov
