// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

#include "ngraph/node.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/attribute_visitor.hpp"

using namespace ngraph;

Variant::~Variant() = default;

ov::Any Variant::init(const std::shared_ptr<ngraph::Node>& node) {
    return {};
}

ov::Any Variant::merge(const ngraph::NodeVector& nodes) {
    return {};
}

bool Variant::is_copyable() const {
    return true;
}

template class ngraph::VariantImpl<std::string>;
template class ngraph::VariantImpl<int64_t>;
template class ngraph::VariantImpl<bool>;
