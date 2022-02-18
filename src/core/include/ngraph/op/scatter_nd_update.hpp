// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/scatter_nd_base.hpp"
#include "openvino/op/scatter_nd_update.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ScatterNDUpdate;
}  // namespace v3
using v3::ScatterNDUpdate;
}  // namespace op
}  // namespace ngraph
