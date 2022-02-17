// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/op/op.hpp"
#include "openvino/op/sink.hpp"

namespace ngraph {
namespace op {
using ov::op::Sink;
}  // namespace op
using SinkVector = std::vector<std::shared_ptr<op::Sink>>;
}  // namespace ngraph
