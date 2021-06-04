// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/variant.hpp"

using namespace ngraph;

Variant::~Variant() {}

std::shared_ptr<ngraph::Variant> Variant::init(const std::shared_ptr<ngraph::Node>& node)
{
    return nullptr;
}

std::shared_ptr<ngraph::Variant> Variant::merge(const ngraph::NodeVector& nodes)
{
    return nullptr;
}

NGRAPH_VARIANT_DEFINITION(std::string, "Variant::std::string", 0);
NGRAPH_VARIANT_DEFINITION(std::int64_t, "Variant::int64_t", 0);
