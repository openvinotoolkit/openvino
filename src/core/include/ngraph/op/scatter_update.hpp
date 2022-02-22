// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/scatter_base.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/scatter_update.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ScatterUpdate;
}  // namespace v3
}  // namespace op
}  // namespace ngraph
