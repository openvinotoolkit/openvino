// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ScatterElementsUpdate;
}  // namespace v3
using v3::ScatterElementsUpdate;
}  // namespace op
}  // namespace ngraph
