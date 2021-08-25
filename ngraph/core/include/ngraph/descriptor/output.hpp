// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ngraph {
// The forward declaration of Node is needed here because Node has a deque of
// Outputs, and Output is an incomplete type at this point. STL containers of
// incomplete type have undefined behavior according to the C++11 standard, and
// in practice including node.hpp here was causing compilation errors on some
// systems (namely macOS).
class Node;

namespace descriptor {
// Describes an output tensor of an op
using ov::descriptor::Output;
}  // namespace descriptor
}  // namespace ngraph
