// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <memory>
#include <unordered_map>

#include "ngraph/op/util/variable.hpp"
#include "ngraph/op/util/variable_value.hpp"
#include "ngraph/output_vector.hpp"
#include "openvino/op/util/variable_context.hpp"

namespace ngraph {
using VariableMap = std::unordered_map<VariablePtr, VariableValuePtr>;
using ov::op::util::VariableContext;
}  // namespace ngraph
