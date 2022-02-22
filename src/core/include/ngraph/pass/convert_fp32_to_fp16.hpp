// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/pass/graph_rewrite.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"

namespace ngraph {
namespace pass {
using ov::pass::ConvertFP32ToFP16;
}  // namespace pass
}  // namespace ngraph
