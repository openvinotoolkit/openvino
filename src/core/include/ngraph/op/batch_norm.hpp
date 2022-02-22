// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/batch_norm.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::BatchNormInference;
}  // namespace v0
namespace v5 {
using ov::op::v5::BatchNormInference;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
