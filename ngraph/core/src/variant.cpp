// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

#include "ngraph/node.hpp"
#include "openvino/core/attribute_visitor.hpp"

using namespace ngraph;

Variant::~Variant() = default;

std::shared_ptr<ngraph::Variant> Variant::init(const std::shared_ptr<ngraph::Node>& node) {
    return nullptr;
}

std::shared_ptr<ngraph::Variant> Variant::merge(const ngraph::NodeVector& nodes) {
    return nullptr;
}

bool Variant::is_copyable() const {
    return true;
}

template class ngraph::VariantImpl<std::string>;
template class ngraph::VariantImpl<int64_t>;

bool ov::IndexWrapper::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", m_value);
    return true;
}
