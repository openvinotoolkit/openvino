// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>
#include <ie_extension.h>

//! [extension:header]
namespace TemplateExtension {

class Extension1 : public InferenceEngine::NewExtension {
public:
    Extension1();
};

class Extension2 : public InferenceEngine::NewExtension {
public:
    Extension2();
};

}  // namespace TemplateExtension
//! [extension:header]

