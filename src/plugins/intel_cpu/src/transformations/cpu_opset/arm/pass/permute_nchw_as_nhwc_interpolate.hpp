// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

class PermuteNCHWAsNHWCInterpolate : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PermuteNCHWAsNHWCInterpolate");
    PermuteNCHWAsNHWCInterpolate();
};

}  // namespace ov::intel_cpu
