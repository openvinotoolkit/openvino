// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/register_info.hpp"

template class ngraph::VariantImpl<std::vector<size_t>>;

BWDCMP_RTTI_DEFINITION(ngraph::VariantWrapper<std::vector<size_t>>);
