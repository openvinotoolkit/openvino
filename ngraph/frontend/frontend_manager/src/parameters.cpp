// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/parameters.hpp"

using namespace ngraph;

constexpr VariantTypeInfo VariantWrapper<std::istream*>::type_info;

constexpr VariantTypeInfo VariantWrapper<std::istringstream*>::type_info;

constexpr VariantTypeInfo VariantWrapper<std::shared_ptr<ngraph::runtime::AlignedBuffer>>::type_info;

constexpr VariantTypeInfo VariantWrapper<std::map<std::string, ngraph::OpSet>>::type_info;