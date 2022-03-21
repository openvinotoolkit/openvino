// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/factory.h"
#include "openvino/core/node.hpp"

namespace ov {
namespace opset {
OV_CC_DOMAINS(opset_factory)

class Factory : public openvino::cc::Factory<NodeTypeInfo, ov::Node*()> {
public:
    Factory(const std::string& name) : openvino::cc::Factory<NodeTypeInfo, ov::Node*()>(name) {}

    Factory(const ov::opset::Factory& factory)
        : openvino::cc::Factory<NodeTypeInfo, ov::Node*()>(factory.name) {
        builders = factory.builders;
    }

    Factory& operator=(const ov::opset::Factory& factory) {
        name = factory.name;
        builders = factory.builders;
        return *this;
    }

    /// @brief Register specific builder for operation's type
    /// Provides limited conditional compilation (CC) capabilities
    /// due each operation will be registered with `extension` name.
    /// For full CC capabilities use `registerImplIfRequired()` inlined
    /// instead of `register_type` (see details in CCOpSet).
    void register_type(const NodeTypeInfo& type_info, const std::function<ov::Node*()>& builder) {
        registerImplIfRequired(opset_factory, extension, type_info, std::move(builder));
    }

    ov::Node* create(const NodeTypeInfo& type_info) {
        return createNodeIfRegistered(opset_factory, type_info);
    }

    std::unordered_map<NodeTypeInfo, std::function<ov::Node*()>> get_builders() {
        return builders;
    }
};
}  // namespace opset
}  // namespace ov
