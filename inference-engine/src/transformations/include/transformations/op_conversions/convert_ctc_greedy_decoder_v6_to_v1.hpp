// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
    namespace pass {

        class TRANSFORMATIONS_API ConvertCTCGreedyDecoderV6ToV1;

    }  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertCTCGreedyDecoderV6ToV1: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertCTCGreedyDecoderV6ToV1();
};
