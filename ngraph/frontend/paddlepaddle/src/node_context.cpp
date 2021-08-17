// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"

#define NGRAPH_VARIANT_DEFINITION(TYPE)                        \
    constexpr VariantTypeInfo VariantWrapper<TYPE>::type_info; \
    template class ngraph::VariantImpl<TYPE>;

namespace ov {
NGRAPH_VARIANT_DEFINITION(int32_t)
NGRAPH_VARIANT_DEFINITION(std::vector<int32_t>)
NGRAPH_VARIANT_DEFINITION(float)
NGRAPH_VARIANT_DEFINITION(std::vector<float>)
NGRAPH_VARIANT_DEFINITION(bool)
NGRAPH_VARIANT_DEFINITION(ngraph::element::Type)
NGRAPH_VARIANT_DEFINITION(std::vector<int64_t>)
}  // namespace ov
