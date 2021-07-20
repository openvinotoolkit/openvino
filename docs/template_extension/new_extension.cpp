// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "new_extension.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpu_kernel.hpp"
#include "op.hpp"

using namespace TemplateExtension;

//! [extension:ctor]
Extension1::Extension1() {}
Extension2::Extension2() {}
//! [extension:ctor]

IE_CREATE_DEFAULT_CONTAINER(std::vector<InferenceEngine::NewExtension::Ptr>({std::make_shared<Extension1>(), std::make_shared<Extension2>()}));
IE_CREATE_EXTENSIONS(std::vector<InferenceEngine::NewExtension::Ptr>({std::make_shared<Extension1>(), std::make_shared<Extension2>()}));
