// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/runtime_attribute.hpp"

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"

namespace ov {

RuntimeAttribute::~RuntimeAttribute() = default;

std::string RuntimeAttribute::to_string() const {
    return {};
}
bool RuntimeAttribute::visit_attributes(AttributeVisitor&) {
    return false;
}

ov::Any RuntimeAttribute::init(const std::shared_ptr<ov::Node>& node) const {
    return {};
}

ov::Any RuntimeAttribute::merge(const ov::NodeVector& nodes) const {
    return {};
}

ov::Any RuntimeAttribute::merge(const ov::OutputVector& outputs) const {
    return {};
}

bool RuntimeAttribute::is_copyable() const {
    return true;
}

bool RuntimeAttribute::is_copyable(const std::shared_ptr<Node>& to) const {
    return is_copyable();
}

std::ostream& operator<<(std::ostream& os, const RuntimeAttribute& attrubute) {
    return os << attrubute.to_string();
}

}  // namespace ov
