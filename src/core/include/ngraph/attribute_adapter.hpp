// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"
#include "openvino/core/attribute_adapter.hpp"

///
namespace ngraph {

using ov::ValueAccessor;

using ov::DirectValueAccessor;

using ov::IndirectScalarValueAccessor;

using ov::copy_from;

using ov::IndirectVectorValueAccessor;

using ov::AttributeAdapter;
using ov::EnumAttributeAdapterBase;

using ov::VisitorAdapter;

}  // namespace ngraph
