// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>

namespace ov {
namespace intel_cpu {

class Extension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;
    void Unload() noexcept override;
    std::map<std::string, ngraph::OpSet> getOpSets() override;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ov::Node>& node) override;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ov::Node>& node, const std::string& implType) override;
};

}   // namespace intel_cpu
}   // namespace ov
