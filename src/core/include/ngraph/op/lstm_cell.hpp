// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/lstm_cell.hpp"

namespace ngraph {
namespace op {
using ov::op::LSTMWeightsFormat;

namespace v0 {
using ov::op::v0::LSTMCell;
}  // namespace v0

namespace v4 {
using ov::op::v4::LSTMCell;
}  // namespace v4
}  // namespace op
}  // namespace ngraph
