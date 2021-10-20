// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/node_vector.hpp"

namespace ngraph {

using NodeVector = ov::NodeVector;
using ConstNodeVector = ov::ConstNodeVector;
using OutputVector = ov::OutputVector;
using ConstOutputVector = ov::ConstOutputVector;
}  // namespace ngraph
