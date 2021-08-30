// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "openvino/core/variant.hpp"

namespace ov {
class Node;
}
namespace ngraph {
using ov::Node;
using ov::VariantTypeInfo;

using ov::Variant;
using ov::VariantImpl;
using ov::VariantWrapper;

using ov::make_variant;

using ov::RTMap;
}  // namespace ngraph
