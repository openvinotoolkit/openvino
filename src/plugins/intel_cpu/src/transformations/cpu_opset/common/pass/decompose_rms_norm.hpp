// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

class DecomposeRMSNorm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DecomposeRMSNorm");
    DecomposeRMSNorm();
};

}  // namespace ov::intel_cpu
