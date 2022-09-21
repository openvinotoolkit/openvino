// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/factory.h"
#include "openvino/core/node.hpp"

namespace ov {
namespace opset {
OV_CC_DOMAINS(opset_factory)

class Factory : public openvino::cc::Factory<std::string, ov::Node*()> {
public:
    Factory(const std::string& name) : openvino::cc::Factory<std::string, ov::Node*()>(name) {}

    Factory(const ov::opset::Factory& factory) : openvino::cc::Factory<std::string, ov::Node*()>(factory.name) {
        builders = factory.builders;
    }

    Factory& operator=(const ov::opset::Factory& factory) {
        name = factory.name;
        builders = factory.builders;
        return *this;
    }

    /// @brief Register specific builder for operation
    /// Provides limited conditional compilation (CC) capabilities
    /// due each operation will be registered with `extension` name.
    /// For full CC capabilities use `registerImplIfRequired()` inlined
    /// instead of `register_builder` (see details in CCOpSet).
    void register_builder(const std::string& op_name, const std::function<ov::Node*()>& builder) {
        registerImplIfRequired(opset_factory, extension, op_name, builder);
    }

    ov::Node* create(const std::string& op_name) {
        return createNodeIfRegistered(opset_factory, op_name);
    }

    std::unordered_map<std::string, std::function<ov::Node*()>> get_builders() {
        return builders;
    }
};
}  // namespace opset
}  // namespace ov
