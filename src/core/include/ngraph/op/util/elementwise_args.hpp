// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "openvino/op/util/elementwise_args.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::validate_and_infer_elementwise_args;
}
}  // namespace op
}  // namespace ngraph
