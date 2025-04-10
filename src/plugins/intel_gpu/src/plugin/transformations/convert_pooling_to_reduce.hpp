// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_gpu {

class ConvertAvgPoolingToReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertAvgPoolingToReduce");
    ConvertAvgPoolingToReduce();
};

}  // namespace ov::intel_gpu
