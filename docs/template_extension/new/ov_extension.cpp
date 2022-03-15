// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "identity.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<TemplateExtension::Identity>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::Identity>>("TemplateIdentity")
    }));
//! [ov_extension:entry_point]
// clang-format on
