// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <memory>
#include <vector>

#include "openvino/core/node_vector.hpp"

namespace ngraph {

using NodeVector = ov::NodeVector;
using OutputVector = ov::OutputVector;
}  // namespace ngraph
