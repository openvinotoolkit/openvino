//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
