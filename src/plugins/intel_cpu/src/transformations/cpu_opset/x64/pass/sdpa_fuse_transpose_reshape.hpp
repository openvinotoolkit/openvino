// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov::intel_cpu {
class SDPAFuseTransposeReshape : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAFuseTransposeReshape");
    SDPAFuseTransposeReshape();
};

}  // namespace ov::intel_cpu
