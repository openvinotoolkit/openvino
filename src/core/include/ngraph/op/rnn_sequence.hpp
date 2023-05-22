// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/rnn_sequence.hpp"

namespace ngraph {
namespace op {
namespace v5 {
using ov::op::v5::RNNSequence;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
