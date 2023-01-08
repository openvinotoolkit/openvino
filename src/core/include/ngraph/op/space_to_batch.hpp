// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/space_to_batch.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::SpaceToBatch;
}  // namespace v1
using v1::SpaceToBatch;
}  // namespace op
}  // namespace ngraph
