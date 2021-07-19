// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/per_tensor_quantization_attribute.hpp"

using namespace ngraph;

template class ngraph::VariantImpl<PerTensorQuantizationAttribute>;
constexpr VariantTypeInfo VariantWrapper<PerTensorQuantizationAttribute>::type_info;