// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

using namespace ngraph;

// Define variant for std::string
constexpr VariantTypeInfo VariantWrapper<std::string>::type_info;
constexpr VariantTypeInfo VariantWrapper<int64_t>::type_info;

Variant::~Variant() {}

std::shared_ptr<ngraph::Variant> Variant::init(const std::shared_ptr<ngraph::Node>& node)
{
    return nullptr;
}

std::shared_ptr<ngraph::Variant> Variant::merge(const ngraph::NodeVector& nodes)
{
    return nullptr;
}

template class ngraph::VariantImpl<std::string>;
template class ngraph::VariantImpl<int64_t>;
