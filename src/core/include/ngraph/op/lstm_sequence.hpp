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
