// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/util/variable_value.hpp"

namespace ngraph {
using ov::op::util::VariableValue;
using VariableValuePtr = std::shared_ptr<VariableValue>;
}  // namespace ngraph
