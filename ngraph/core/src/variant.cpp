// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

using namespace ov;

// Define variant for std::string
constexpr VariantTypeInfo VariantWrapper<std::string>::type_info;
constexpr VariantTypeInfo VariantWrapper<int64_t>::type_info;

Variant::~Variant() {}

std::shared_ptr<ov::Variant> Variant::init(const std::shared_ptr<ov::Node>& node)
{
    return nullptr;
}

std::shared_ptr<ov::Variant> Variant::merge(const ov::NodeVector& nodes)
{
    return nullptr;
}

bool Variant::is_copyable() const
{
    return true;
}

template class ov::VariantImpl<std::string>;
template class ov::VariantImpl<int64_t>;
