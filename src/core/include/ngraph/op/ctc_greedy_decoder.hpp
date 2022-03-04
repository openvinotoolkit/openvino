// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/ctc_greedy_decoder.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::CTCGreedyDecoder;
}  // namespace v0
using v0::CTCGreedyDecoder;
}  // namespace op
}  // namespace ngraph
