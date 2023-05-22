// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <map>
#include <memory>

#include "ngraph/descriptor/tensor.hpp"
#include "openvino/core/descriptor/input.hpp"

namespace ngraph {
using ov::Node;
namespace descriptor {

// Describes a tensor that is an input to an op, directly or indirectly via a tuple
using ov::descriptor::Input;
}  // namespace descriptor
}  // namespace ngraph
