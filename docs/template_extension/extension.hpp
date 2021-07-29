// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_iextension.h>

#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <string>
#include <vector>

//! [extension:header]
namespace TemplateExtension {

class Extension : public InferenceEngine::IExtension {
public:
    Extension();
    ~Extension();
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override;
    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) override;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) override;
};

}  // namespace TemplateExtension
//! [extension:header]
