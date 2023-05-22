// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

#include "ngraph/pass/graph_rewrite.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"

namespace ngraph {
namespace pass {
using ov::pass::ConvertFP32ToFP16;
}  // namespace pass
}  // namespace ngraph
