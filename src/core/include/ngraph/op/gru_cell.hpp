// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"
#include "openvino/op/gru_cell.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::GRUCell;
}  // namespace v3
}  // namespace op
}  // namespace ngraph
