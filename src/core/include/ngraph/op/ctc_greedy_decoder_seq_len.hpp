// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef IN_OV_LIBRARY
#    warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include "ngraph/op/op.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

namespace ngraph {
namespace op {
namespace v6 {
using ov::op::v6::CTCGreedyDecoderSeqLen;
}  // namespace v6
}  // namespace op
}  // namespace ngraph
