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

class MHAFusion2: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAFusion2", "0");
    MHAFusion2();
};

class MHAQuantFusion: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAQuantFusion", "0");
    MHAQuantFusion();
};

class MHAQuantFusion2: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAQuantFusion2", "0");
    MHAQuantFusion2();
};

}   // namespace intel_cpu
}   // namespace ov
