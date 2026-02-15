// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/core/visibility.hpp"

namespace ov::intel_gpu {

class ConvertShapeOf1To3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertShapeOf1To3");
    ConvertShapeOf1To3();
};

}  // namespace ov::intel_gpu
