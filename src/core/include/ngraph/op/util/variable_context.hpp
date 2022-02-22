// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <unordered_map>

#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_value.hpp"
#include "ngraph/output_vector.hpp"
#include "ngraph/variant.hpp"
#include "openvino/op/util/variable_context.hpp"

namespace ngraph {
using VariableMap = std::unordered_map<VariablePtr, VariableValuePtr>;
using ov::op::util::VariableContext;
}  // namespace ngraph
