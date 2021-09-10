// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"

namespace ov {
BWDCMP_RTTI_DEFINITION(VariantWrapper<int32_t>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<std::vector<int32_t>>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<float>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<std::vector<float>>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<bool>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<ngraph::element::Type>);
BWDCMP_RTTI_DEFINITION(VariantWrapper<std::vector<int64_t>>);
}  // namespace ov
