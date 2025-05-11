// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ov::intel_cpu {

class ConvertToInteraction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertToInteraction");
    ConvertToInteraction();
};

class FuseFQtoInteraction : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseFQtoInteraction");
    FuseFQtoInteraction();
};

class ConvertInteractionInt8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertInteractionInt8");
    ConvertInteractionInt8();
};

}  // namespace ov::intel_cpu
