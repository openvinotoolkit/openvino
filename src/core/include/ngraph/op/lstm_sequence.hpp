// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/lstm_cell.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/lstm_sequence.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::LSTMSequence;
}  // namespace v0

namespace v5 {
using ov::op::v5::LSTMSequence;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
