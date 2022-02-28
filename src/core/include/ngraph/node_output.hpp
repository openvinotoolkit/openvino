// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <map>
#include <unordered_set>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/node_output.hpp"

namespace ngraph {
using ov::Node;
using ov::Output;
}  // namespace ngraph
