// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class MishDecomposition : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MishDecomposition();
};

}  // namespace vpu

