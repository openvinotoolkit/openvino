// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

class UpdateConvert : public ngraph::pass::MatcherPass {
public:
    UpdateConvert();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
