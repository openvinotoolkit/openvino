// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/variant.hpp>
#include "ngraph/op/util/variable_context.hpp"

constexpr ov::VariantTypeInfo ov::VariantWrapper<ov::VariableContext>::type_info;
