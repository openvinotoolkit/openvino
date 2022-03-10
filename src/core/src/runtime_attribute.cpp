// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/runtime_attribute.hpp"

#include "ngraph/node.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {

std::string RuntimeAttribute::to_string() const {
    return {};
}
bool RuntimeAttribute::visit_attributes(AttributeVisitor&) {
    return false;
}

Any RuntimeAttribute::init(const std::shared_ptr<ngraph::Node>& node) const {
    return {};
}

Any RuntimeAttribute::merge(const ngraph::NodeVector& nodes) const {
    return {};
}

Any RuntimeAttribute::merge(const ngraph::OutputVector& outputs) const {
    return {};
}

bool RuntimeAttribute::is_copyable() const {
    return true;
}

std::ostream& operator<<(std::ostream& os, const RuntimeAttribute& attrubute) {
    return os << attrubute.to_string();
}

}  // namespace ov