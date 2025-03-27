// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov::snippets {
void mark_as_external_parameter(const std::shared_ptr<Node>& node);

bool is_external_parameter(const std::shared_ptr<const Node>& node);

// TODO: add a description
class ExternalParameterAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("ExternalParameterAttribute", "", ov::RuntimeAttribute);
    bool is_copyable() const override {
        return false;
    }
};
}  // namespace ov::snippets
