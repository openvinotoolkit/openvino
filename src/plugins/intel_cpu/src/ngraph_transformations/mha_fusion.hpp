// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class MHAFusion: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAFusion", "0");
    MHAFusion();
};

class MHAQuantFusion: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAQuantFusion", "0");
    MHAQuantFusion();
};

}   // namespace intel_cpu
}   // namespace ov
