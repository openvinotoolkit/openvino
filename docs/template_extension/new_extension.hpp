// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_extension.h>

//! [extension:header]
namespace TemplateExtension {

class Extension1 : public InferenceEngine::OpsetExtension {
public:
    NGRAPH_RTTI_DECLARATION;
    Extension1();
    std::map<std::string, ngraph::OpSet> getOpSets() override {
        return {};
    }
};

class Extension2 : public InferenceEngine::OpsetExtension {
public:
    NGRAPH_RTTI_DECLARATION;
    Extension2();
    std::map<std::string, ngraph::OpSet> getOpSets() override {
        return {};
    }
};

}  // namespace TemplateExtension
//! [extension:header]
