// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

class FuseLoadAndConvert: public ngraph::pass::MatcherPass {
public:
    FuseLoadAndConvert(const std::vector<element::Type> supported_types);
};

}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
