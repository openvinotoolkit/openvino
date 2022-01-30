// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 * transformation aligns elementwise input ranks by adding unsqueeze to constant node that has smaller rank than the other input
 */

namespace MKLDNNPlugin {

class AlignEltwiseInputRanks: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    AlignEltwiseInputRanks();
};

}  // namespace MKLDNNPlugin
