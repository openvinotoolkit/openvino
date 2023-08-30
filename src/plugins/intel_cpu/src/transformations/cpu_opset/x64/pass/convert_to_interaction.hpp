// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertToInteraction: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertToInteraction", "0");
    ConvertToInteraction();
};

class FuseFQtoInteraction: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FuseFQtoInteraction", "0");
    FuseFQtoInteraction();
};

class ConvertInteractionInt8: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertInteractionInt8", "0");
    ConvertInteractionInt8();
};

}   // namespace intel_cpu
}   // namespace ov
