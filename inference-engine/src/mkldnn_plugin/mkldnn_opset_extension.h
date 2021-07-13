// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_iextension.h>

namespace MKLDNNPlugin {

class OpsetExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept final;
    void Unload() noexcept final;
    std::map<std::string, ngraph::OpSet> getOpSets() final;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) final;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) final;
};

class TypeRelaxedOpsetExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept final;
    void Unload() noexcept final;
    std::map<std::string, ngraph::OpSet> getOpSets() final;
    std::vector<std::string> getImplTypes(const std::shared_ptr<ngraph::Node>& node) final;
    InferenceEngine::ILayerImpl::Ptr getImplementation(const std::shared_ptr<ngraph::Node>& node, const std::string& implType) final;
};

}  // namespace MKLDNNPlugin
