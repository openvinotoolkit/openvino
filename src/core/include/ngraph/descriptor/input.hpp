// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>

#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/variant.hpp"
#include "openvino/core/descriptor/input.hpp"

namespace ngraph {
using ov::Node;
namespace descriptor {

// Describes a tensor that is an input to an op, directly or indirectly via a tuple
using ov::descriptor::Input;
}  // namespace descriptor
}  // namespace ngraph
