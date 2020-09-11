// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>
#include <ie_api.h>
#include <ngraph/ngraph.hpp>
#include <memory>
#include <vector>
#include <string>
#include <map>

//! [extension:header]
namespace TemplateExtension {

class Extension : public InferenceEngine::IExtension {
public:
    Extension() = default;
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;
    void Unload() noexcept override {}
    void Release() noexcept override { delete this; }

    std::map<std::string, ngraph::OpSet> getOpSets() override;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override;
};

}  // namespace TemplateExtension
//! [extension:header]
