// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ReshapeFullyConnectedFusion : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ReshapeFullyConnectedFusion();
};

}   // namespace intel_cpu
}   // namespace ov
