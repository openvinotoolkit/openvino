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
#include "openvino/op/util/rnn_cell_base.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::convert_lstm_node_format;
using ov::op::util::LSTMWeightsFormat;
using ov::op::util::RNNCellBase;
}  // namespace util
}  // namespace op
}  // namespace ngraph
