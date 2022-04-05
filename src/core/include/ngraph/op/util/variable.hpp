// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "ngraph/partial_shape.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/op/util/variable.hpp"

namespace ngraph {
using ov::op::util::Variable;
using ov::op::util::VariableInfo;
using VariablePtr = std::shared_ptr<Variable>;
using VariableVector = std::vector<VariablePtr>;
}  // namespace ngraph
