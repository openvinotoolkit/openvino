// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/gru_sequence.hpp"

namespace ngraph {
namespace op {
namespace v5 {
using ov::op::v5::GRUSequence;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
