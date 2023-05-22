// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

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
