// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
