// Copyright (C) 2018-2022 Intel Corporation
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
using ov::Node;
namespace descriptor {
// Describes an output tensor of an op
using ov::descriptor::Output;
}  // namespace descriptor
}  // namespace ngraph
