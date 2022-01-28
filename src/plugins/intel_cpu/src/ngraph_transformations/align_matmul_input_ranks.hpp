// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     AlignMatMulInputRanks transformation detects MatMul operations
 *     and unsqueezes one input to another to align the ranks of the inputs.
 *     The transformation is required because oneDNN library
 *     requires inputs to have equal ranks
 */

namespace MKLDNNPlugin {

class AlignMatMulInputRanks: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AlignMatMulInputRanks();
};

}  // namespace MKLDNNPlugin
