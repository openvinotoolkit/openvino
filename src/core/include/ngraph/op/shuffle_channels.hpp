// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/shuffle_channels.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::ShuffleChannels;
}  // namespace v0
using v0::ShuffleChannels;
}  // namespace op
}  // namespace ngraph
