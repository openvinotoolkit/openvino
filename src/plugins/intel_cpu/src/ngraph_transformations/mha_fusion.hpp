// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class MHAFloatFusion: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAFloatFusion", "0");
    MHAFloatFusion();
};

class MHAFloatFusion2: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("MHAFloatFusion2", "0");
    MHAFloatFusion2();
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

class MHAFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("MHAFusion", "0");
    MHAFusion() {
        add_matcher<MHAFloatFusion>();
        add_matcher<MHAFloatFusion2>();
        add_matcher<MHAQuantFusion>();
        add_matcher<MHAQuantFusion2>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
